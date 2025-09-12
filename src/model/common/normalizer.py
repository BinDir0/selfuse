from typing import Union, Dict, Optional

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from src.utils.pytorch_util import dict_apply
from .dict_of_tensor_mixin import DictOfTensorMixin

class StreamingStats:
    """
    class for streaming statistics calculation, supporting dynamic data addition and updating statistics
    """
    
    def __init__(self, last_n_dims=1):
        self.last_n_dims = last_n_dims
        self.n_samples = 0
        self.sum = None
        self.sum_sq = None
        self.min_vals = None
        self.max_vals = None
        self.dim = None
        
    def update(self, data: Union[torch.Tensor, np.ndarray]):
        """
        add new data and update statistics
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        if self.dim is None:
            if self.last_n_dims > 0:
                self.dim = np.prod(data.shape[-self.last_n_dims:])
            else:
                self.dim = 1
            data = data.reshape(-1, self.dim)
            
            self.sum = np.zeros(self.dim, dtype=np.float64)
            self.sum_sq = np.zeros(self.dim, dtype=np.float64)
            self.min_vals = np.full(self.dim, np.inf)
            self.max_vals = np.full(self.dim, -np.inf)
        else:
            data = data.reshape(-1, self.dim)
        
        n_new = data.shape[0]
        self.n_samples += n_new
        
        self.sum += np.sum(data, axis=0)
        self.sum_sq += np.sum(data ** 2, axis=0)
        
        self.min_vals = np.minimum(self.min_vals, np.min(data, axis=0))
        self.max_vals = np.maximum(self.max_vals, np.max(data, axis=0))
    
    def get_stats(self):
        """
        get current statistics
        """
        if self.n_samples == 0:
            return None
            
        mean = self.sum / self.n_samples
        if self.n_samples > 1:
            variance = (self.sum_sq - self.n_samples * mean ** 2) / (self.n_samples - 1)
            std = np.sqrt(np.maximum(variance, 0))
        else:
            std = np.zeros_like(mean)
            
        return {
            'min': self.min_vals.copy(),
            'max': self.max_vals.copy(),
            'mean': mean,
            'std': std,
            'n_samples': self.n_samples
        }
    
    def reset(self):
        """
        reset statistics
        """
        self.n_samples = 0
        self.sum = None
        self.sum_sq = None
        self.min_vals = None
        self.max_vals = None
        self.dim = None


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    def __init__(self):
        super().__init__()
        self.streaming_stats = {}
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def start_streaming_fit(self, 
                           keys: Optional[list] = None,
                           last_n_dims=1,
                           dtype=torch.float32,
                           mode='limits',
                           output_max=1.,
                           output_min=-1.,
                           range_eps=1e-4,
                           fit_offset=True):
        """
        start streaming fit, initialize streaming statistics
        """
        self.streaming_config = {
            'last_n_dims': last_n_dims,
            'dtype': dtype,
            'mode': mode,
            'output_max': output_max,
            'output_min': output_min,
            'range_eps': range_eps,
            'fit_offset': fit_offset
        }
        
        if keys is None:
            keys = ['_default']
        
        for key in keys:
            self.streaming_stats[key] = StreamingStats(last_n_dims=last_n_dims)
    
    def update_streaming_fit(self, data: Union[Dict, torch.Tensor, np.ndarray]):
        """
        update streaming statistics
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key in self.streaming_stats:
                    self.streaming_stats[key].update(value)
        else:
            if '_default' in self.streaming_stats:
                self.streaming_stats['_default'].update(data)
    
    def finish_streaming_fit(self):
        """
        finish streaming fit, calculate final normalizer parameters
        """
        if not hasattr(self, 'streaming_config'):
            raise RuntimeError("Must call start_streaming_fit first")
        
        config = self.streaming_config
        
        for key, stats in self.streaming_stats.items():
            stats_dict = stats.get_stats()
            if stats_dict is None:
                continue
                
            self.params_dict[key] = _fit_from_stats(
                stats_dict,
                last_n_dims=config['last_n_dims'],
                dtype=config['dtype'],
                mode=config['mode'],
                output_max=config['output_max'],
                output_min=config['output_min'],
                range_eps=config['range_eps'],
                fit_offset=config['fit_offset']
            )
        
        self.streaming_stats = {}
        delattr(self, 'streaming_config')

    def ignore_dim(self, key: str, dim: slice):
        """
        ignore some dimensions when normalizing, e.g. the wrist rotation
        """
        if key not in self.params_dict:
            raise RuntimeError(f"Not initialized with key: {key}")
        params = self.params_dict[key]
        params['scale'][dim] = 1.0
        params['offset'][dim] = 0.0
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                if key not in self.params_dict:
                    raise RuntimeError(f"Not initialized with key: {key}")
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _fit_from_stats(stats_dict: Dict,
                   last_n_dims=1,
                   dtype=torch.float32,
                   mode='limits',
                   output_max=1.,
                   output_min=-1.,
                   range_eps=1e-4,
                   fit_offset=True):
    """
    calculate normalizer parameters from statistics dictionary
    """
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # extract data from statistics dictionary and convert to torch
    input_min = torch.from_numpy(stats_dict['min'].astype(np.float32))
    input_max = torch.from_numpy(stats_dict['max'].astype(np.float32))
    input_mean = torch.from_numpy(stats_dict['mean'].astype(np.float32))
    input_std = torch.from_numpy(stats_dict['std'].astype(np.float32))

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x
