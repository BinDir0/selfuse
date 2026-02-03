import os
import warnings
from typing import Any, Optional, Tuple
import numpy as np
import zarr

class ImmediateFuture:
    """
    Simple Future-like object for already-available data.

    Args:
        value (Any): Resolved value to return from result().

    Returns:
        ImmediateFuture: An object with a .result() method.
    """
    def __init__(self, value):
        self._value = value

    def result(self):
        """
        Return the stored value.

        Returns:
            Any: The value passed at construction time.
        """
        return self._value


class TensorStoreReference:
    """
    Lazy-opening TensorStore reference with async read support.

    The underlying TensorStore is opened on first read to avoid fork/thread
    conflicts when DataLoader uses num_workers > 0.

    Args:
        root_path (str): Root path to the zarr store.
        array_path (str): Relative path of the array inside the store.
        cache_pool_bytes (int): TensorStore cache pool size in bytes.

    Attributes:
        shape (Optional[Tuple[int, ...]]): Array shape (resolved lazily).
        dtype (Optional[np.dtype]): Array dtype (resolved lazily).
        chunks (Optional[Tuple[int, ...]]): Chunk shape if available.
    """
    def __init__(self, root_path: str, array_path: str, cache_pool_bytes: int):
        self.root_path = root_path
        self.array_path = array_path
        self.cache_pool_bytes = cache_pool_bytes
        self.ts_array = None
        self.shape = None
        self.dtype = None
        self.chunks = None
        self._fallback_ref = None

    def _ensure_open(self):
        if self._fallback_ref is not None:
            return
        if self.ts_array is None:
            try:
                self.ts_array = StreamingReplayBuffer._open_tensorstore(
                    self.root_path, self.array_path, self.cache_pool_bytes
                )
                self.shape = self.ts_array.shape
                self.dtype = self.ts_array.dtype
                try:
                    self.chunks = self.ts_array.chunk_layout.read_chunk_shape
                except Exception:
                    self.chunks = None
            except Exception:
                warnings.warn(f"TensorStore open failed for '{self.array_path}', falling back to zarr.")
                # Fallback to zarr when TensorStore cannot open (e.g., unsupported dtype).
                zarr_array = StreamingReplayBuffer._open_zarr_array(self.root_path, self.array_path)
                self._fallback_ref = ZarrReference(zarr_array)
                self.shape = self._fallback_ref.shape
                self.dtype = self._fallback_ref.dtype
                self.chunks = self._fallback_ref.chunks

    def read(self, idx):
        """
        Asynchronously read a slice/index.

        Args:
            idx (Any): Index/slice/array of indices accepted by TensorStore.

        Returns:
            tensorstore.Future: Future whose result is a NumPy array.
        """
        self._ensure_open()
        if self._fallback_ref is not None:
            return self._fallback_ref.read(idx)
        return self.ts_array[idx].read()

    def __getitem__(self, idx):
        """
        Alias for read() to keep indexing syntax.

        Args:
            idx (Any): Index/slice/array of indices.

        Returns:
            tensorstore.Future: Future whose result is a NumPy array.
        """
        return self.read(idx)

    def __len__(self):
        """
        Return length along the first dimension.

        Returns:
            int: Number of elements along axis 0.
        """
        self._ensure_open()
        if self._fallback_ref is not None:
            return len(self._fallback_ref)
        return int(self.ts_array.shape[0]) if self.ts_array.shape else 0


class ZarrReference:
    """
    Lazy-loading Zarr reference with a Future-like read() interface.

    Args:
        zarr_array (zarr.Array): Opened zarr array.
    """
    def __init__(self, zarr_array: zarr.Array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape
        self.dtype = zarr_array.dtype
        self.chunks = zarr_array.chunks

    def read(self, idx):
        """
        Read a slice/index and return a Future-like object.

        Args:
            idx (Any): Index/slice/array of indices accepted by zarr.

        Returns:
            ImmediateFuture: Future-like object whose result is a NumPy array.
        """
        return ImmediateFuture(self.zarr_array[idx])

    def __getitem__(self, idx):
        return self.read(idx)

    def __len__(self):
        return len(self.zarr_array)

class StreamingReplayBuffer:
    """
    Streaming replay buffer for large-scale data backed by TensorStore.

    Attributes:
        zarr_path (str): Root path to the zarr store.
        _data (Dict[str, Union[TensorStoreReference, np.ndarray]]): Data handles.
        _meta (Dict[str, np.ndarray]): Metadata arrays.
    """
    def __init__(self, root=None):
        """
        Initialize an empty buffer. Use copy_from_path to populate.

        Args:
            root (Any): Unused placeholder for legacy API compatibility.
        """
        self._data = None
        self._meta = None
        self._episode_ends = None
        self.zarr_path = None
        
    @classmethod
    def copy_from_path(cls, path, keys=None, lazy_load=True, cache_pool_bytes: int = 256 * 1024**2):
        """
        Create a StreamingReplayBuffer from a zarr directory using TensorStore.

        Args:
            path (str): Root path to the zarr store.
            keys (Optional[List[str]]): Prefix keys to include (e.g. ["image", "state"]).
            lazy_load (bool): If True, store TensorStoreReference objects; otherwise load to memory.
            cache_pool_bytes (int): TensorStore cache pool size in bytes for each worker (default: 256MB).

        Returns:
            StreamingReplayBuffer: Initialized buffer instance.
        """
        buffer = cls()
        buffer.zarr_path = path

        array_paths = cls._discover_zarr_arrays(path)
        meta_arrays = [p for p in array_paths if p == "meta" or p.startswith("meta/")]
        data_arrays = [p for p in array_paths if not (p == "meta" or p.startswith("meta/"))]
        has_data_group = any(p == "data" or p.startswith("data/") for p in data_arrays)
        data_prefix = "data/" if has_data_group else ""
        if data_prefix:
            data_arrays = [p for p in data_arrays if p.startswith(data_prefix)]
        # Load metadata (small) synchronously using zarr only.
        buffer._meta = dict()
        for meta_path in meta_arrays:
            meta_key = meta_path[len("meta/"):] if meta_path.startswith("meta/") else meta_path
            zarr_array = cls._open_zarr_array(path, meta_path)
            buffer._meta[meta_key] = cls._read_zarr_array(zarr_array)

        # Set data references (lazy by default, async on read)
        buffer._data = dict()
        for array_path in data_arrays:
            key = array_path[len(data_prefix):] if data_prefix else array_path
            if keys is not None:
                if not any((key == k or key.startswith(f"{k}/")) for k in keys):
                    continue

            # Use references for image/depth regardless of lazy_load to avoid huge memory.
            force_lazy = key.split("/")[0] in ("image", "depth")
            if lazy_load or force_lazy:
                buffer._data[key] = TensorStoreReference(path, array_path, cache_pool_bytes)
            else:
                zarr_array = cls._open_zarr_array(path, array_path)
                buffer._data[key] = cls._read_zarr_array(zarr_array)

        return buffer
    
    @staticmethod
    def _open_tensorstore(root_path: str, array_path: str, cache_pool_bytes: int) -> Any:
        """
        Open a TensorStore array for a given zarr array path.

        Args:
            root_path (str): Root path to the zarr store.
            array_path (str): Relative path of the array inside the store.
            cache_pool_bytes (int): TensorStore cache pool size in bytes, for each worker.

        Returns:
            tensorstore.TensorStore: Opened TensorStore array.
        """
        # Import tensorstore lazily to avoid initializing its thread pools before fork.
        import tensorstore as ts
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": root_path},
            "path": array_path,
        }
        context = ts.Context({
            "cache_pool": {"total_bytes_limit": int(cache_pool_bytes)},
            # Limit default TensorStore concurrency to avoid oversubscription.
            "file_io_concurrency": {"limit": 2},
            "data_copy_concurrency": {"limit": 2},
        })
        return ts.open(spec, open=True, read=True, context=context).result()

    @staticmethod
    def _open_zarr_array(root_path: str, array_path: str) -> zarr.Array:
        """
        Open a zarr array directly from the store.

        Args:
            root_path (str): Root path to the zarr store.
            array_path (str): Relative path of the array inside the store.

        Returns:
            zarr.Array: Opened zarr array.
        """
        return zarr.open(os.path.join(root_path, array_path), mode="r")

    @staticmethod
    def _read_zarr_array(zarr_array: zarr.Array):
        """
        Read a zarr array into memory, handling scalar arrays.

        Args:
            zarr_array (zarr.Array): Opened zarr array.

        Returns:
            np.ndarray: Array contents.
        """
        if zarr_array.shape is None or len(zarr_array.shape) == 0:
            return np.array(zarr_array[()])
        return zarr_array[:]

    @staticmethod
    def _discover_zarr_arrays(root_path: str):
        """
        Discover all array paths by finding .zarray files under a zarr store.

        Args:
            root_path (str): Root path to the zarr store.

        Returns:
            List[str]: Relative paths of arrays in the store.
        """
        array_paths = []
        for dirpath, _, filenames in os.walk(root_path):
            if ".zarray" in filenames:
                rel_path = os.path.relpath(dirpath, root_path)
                if rel_path == ".":
                    rel_path = ""
                array_paths.append(rel_path)
        return array_paths

    @property
    def data(self):
        """
        Get the data dictionary.

        Returns:
            Dict[str, Union[TensorStoreReference, np.ndarray]]: Data handles.
        """
        return self._data

    @property
    def meta(self):
        """
        Get the metadata dictionary.

        Returns:
            Dict[str, np.ndarray]: Metadata arrays.
        """
        return self._meta

    @property
    def episode_ends(self):
        """
        Get episode ending indices.

        Returns:
            np.ndarray: 1D array of episode end indices.
        """
        if self._episode_ends is None:
            if 'episode_ends' in self.meta:
                self._episode_ends = self.meta['episode_ends']
            else:
                # Get length of first data
                first_data = next(iter(self.data.values()))
                self._episode_ends = np.array([len(first_data)])
        return self._episode_ends

    def get_episode(self, idx, copy=False):
        """
        Get data for a specified episode (synchronous).

        Args:
            idx (int): Episode index.
            copy (bool): If True, returns a copy of NumPy arrays.

        Returns:
            Dict[str, np.ndarray]: Data for the episode.
        """
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        
        result = dict()
        for key, value in self.data.items():
            if isinstance(value, TensorStoreReference):
                x = value.read(slice(start_idx, end_idx)).result()
            else:
                x = value[start_idx:end_idx]
            if copy and isinstance(x, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """
        Get data for a step range (synchronous).

        Args:
            start (int): Start index (inclusive).
            stop (int): Stop index (exclusive).
            step (Optional[int]): Step size.
            copy (bool): If True, returns a copy of NumPy arrays.

        Returns:
            Dict[str, np.ndarray]: Data for the slice.
        """
        _slice = slice(start, stop, step)
        result = dict()
        for key, value in self.data.items():
            if isinstance(value, TensorStoreReference):
                x = value.read(_slice).result()
            else:
                x = value[_slice]
            if copy and isinstance(x, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def __len__(self):
        """
        Get total number of steps across all episodes.

        Returns:
            int: Number of steps.
        """
        if len(self.episode_ends) == 0:
            return 0
        return int(self.episode_ends[-1])

    def __getitem__(self, key):
        """
        Get data handle by key.

        Args:
            key (str): Data key.

        Returns:
            Union[TensorStoreReference, np.ndarray]: Data handle or array.
        """
        return self.data[key]

    def keys(self):
        """
        Get all data keys.

        Returns:
            KeysView[str]: Keys view of the data dict.
        """
        return self.data.keys()

    def values(self):
        """
        Get all data values.

        Returns:
            ValuesView[Union[TensorStoreReference, np.ndarray]]: Values view of the data dict.
        """
        return self.data.values()

    def items(self):
        """
        Get all data items.

        Returns:
            ItemsView[str, Union[TensorStoreReference, np.ndarray]]: Items view of the data dict.
        """
        return self.data.items()

    def __contains__(self, key):
        """
        Check if a key exists in the buffer.

        Args:
            key (str): Data key.

        Returns:
            bool: True if key exists.
        """
        return key in self.data

    def read_async(self, key, idx):
        """
        Read data asynchronously from a key.

        Args:
            key (str): Data key.
            idx (Any): Index/slice/array of indices.

        Returns:
            Union[tensorstore.Future, ImmediateFuture]: Future-like object with .result().
        """
        value = self.data[key]
        if isinstance(value, (TensorStoreReference, ZarrReference)):
            return value.read(idx)
        return ImmediateFuture(value[idx])

    @property
    def n_episodes(self):
        """
        Get total number of episodes.

        Returns:
            int: Number of episodes.
        """
        return len(self.episode_ends)

    def get_episode_lengths(self):
        """
        Get length of each episode.

        Returns:
            np.ndarray: Per-episode lengths.
        """
        ends = self.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        return lengths
