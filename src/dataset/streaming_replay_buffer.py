import zarr
import numpy as np

class ZarrReference:
    """Lazy loading image data reference"""
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape
        self.dtype = zarr_array.dtype
        self.chunks = zarr_array.chunks

    def __getitem__(self, idx):
        return self.zarr_array[idx]

    def __len__(self):
        return len(self.zarr_array)

class StreamingReplayBuffer:
    """
    Streaming replay buffer, mainly used for handling large-scale data
    """
    def __init__(self, root=None):
        self._data = None
        self._meta = None
        self._episode_ends = None
        self.zarr_path = None
        
    @classmethod
    def copy_from_path(cls, path, keys=None, lazy_load=True):
        """Create StreamingReplayBuffer from zarr file"""
        buffer = cls()
        buffer.zarr_path = path
        
        # Open zarr storage
        try: 
            src = zarr.open_consolidated(path, mode='r')
        except:
            print(f"Failed to open zarr file {path} with consolidated metadata, trying without")
            src = zarr.open(path, mode='r')
        
        # Load metadata
        buffer._meta = dict()
        if 'meta' in src:
            for key, value in src['meta'].items():
                if len(value.shape) == 0:
                    buffer._meta[key] = np.array(value)
                else:
                    buffer._meta[key] = value[:]
        
        # Set data references
        buffer._data = dict()
        if 'data' not in src:
            src_data = src
        else:
            src_data = src['data']
            
        if keys is None:
            keys = list(src_data.keys())
            
        for key in keys:
            data = src_data[key]
            # Use references for image, because image is too large to load into memory
            # Use references for all other data according to the lazy_load flag
            buffer._data = buffer._data | cls._load_zarr_recursive(
                data, prefix_key=key, lazy_load=lazy_load if key != 'image' and key != 'depth' else True
            )
                
        return buffer
    
    @staticmethod
    def _load_zarr_recursive(node, prefix_key="", lazy_load=True):
        """
        Recursively load Zarr nodes.
        - If the node is a Group, create a dictionary and recursively.
        - If the node is an Array, load the data or create a reference.
        """
        # Recursive case: if the current node is a group
        if isinstance(node, zarr.hierarchy.Group):
            data_dict = dict()
            for sub_key, sub_node in node.items():
                data_dict = data_dict | __class__._load_zarr_recursive(
                    sub_node, prefix_key=f"{prefix_key}/{sub_key}", lazy_load=lazy_load
                )
            return data_dict
        
        if lazy_load:
            return {prefix_key: ZarrReference(node)}
        else:
            return {prefix_key: node[:]}

    @property
    def data(self):
        """Get data dictionary"""
        return self._data

    @property
    def meta(self):
        """Get metadata dictionary"""
        return self._meta

    @property
    def episode_ends(self):
        """Get episode ending indices"""
        if self._episode_ends is None:
            if 'episode_ends' in self.meta:
                self._episode_ends = self.meta['episode_ends']
            else:
                # Get length of first data
                first_data = next(iter(self.data.values()))
                self._episode_ends = np.array([len(first_data)])
        return self._episode_ends

    def get_episode(self, idx, copy=False):
        """Get data for specified episode"""
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        
        result = dict()
        for key, value in self.data.items():
            x = value[start_idx:end_idx]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """Get data for specified step range"""
        _slice = slice(start, stop, step)
        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def __len__(self):
        """Get data length"""
        if len(self.episode_ends) == 0:
            return 0
        return int(self.episode_ends[-1])

    def __getitem__(self, key):
        """Get data for specified key"""
        return self.data[key]

    def keys(self):
        """Get all data keys"""
        return self.data.keys()

    def values(self):
        """Get all data values"""
        return self.data.values()

    def items(self):
        """Get all data items"""
        return self.data.items()

    def __contains__(self, key):
        """Check if contains specified key"""
        return key in self.data

    @property
    def n_episodes(self):
        """Get total number of episodes"""
        return len(self.episode_ends)

    def get_episode_lengths(self):
        """Get length of each episode"""
        ends = self.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        return lengths