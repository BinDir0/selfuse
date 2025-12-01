import torch
import numpy as np
from typing import Dict, Union

class BaseVLPreprocessor:
    def __call__(
        self, 
        image: Union[torch.Tensor, np.ndarray], 
        instruction: str,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        '''
        Preprocess the input images and instruction.
        Args:
            image: Union[torch.Tensor, np.ndarray]
            instruction: str
            **kwargs: Additional arguments
        Returns:
            Dict[str, np.ndarray]
        '''
        raise NotImplementedError("Subclasses must implement this method")
