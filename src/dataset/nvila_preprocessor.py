import torch
import numpy as np
from typing import Dict, Union, List

from .base_vl_preprocessor import BaseVLPreprocessor

# Preprocessor for only one image and one instruction
# Language padding and batching are handled in the collator
class NVILAPreprocessor(BaseVLPreprocessor):

    def __init__(self, image_preprocessor, tokenizer):
        self.image_preprocessor = image_preprocessor
        self.tokenizer = tokenizer

    def _build_input_ids(self, instruction: str) -> torch.Tensor:
        """
        Vectorized construction of input_ids:
        [ encoded instruction ]
        No per-sample Python loops.
        """
        messages = [
            {
                "role": "user",
                "content": instruction
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = np.array(prompt)  # [L_lang]

        return input_ids

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess images for NVILA model.
        """
        image = self.image_preprocessor.preprocess(image, return_tensors="pt")['pixel_values']
        return image.cpu().numpy()

    def __call__(
        self, 
        image: np.ndarray, 
        instruction: str, 
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Tokenize the instruction. 
        Args:
            image: np.ndarray [B, H, W, C]
            instruction: str
            **kwargs: Additional arguments
        Returns:
          - input_ids: [L_lang]
          - image: np.ndarray [B, 3, H, W]
        """
        return {
            "input_ids": self._build_input_ids(instruction),
            "image": self._preprocess_image(image),
        }
    
