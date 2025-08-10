import torch
import numpy as np
from PIL import Image
from typing import Dict, Union, List
from transformers import AutoConfig, AutoModel
from llava.model.language_model.llava_llama import LlavaLlamaModel

from egovla.model.preprocessing.base_vl_preprocessor import BaseVLPreprocessor

# Preprocessor for only one image and one instruction
# Language padding and batching are handled in the collator
class NVILAPreprocessor(BaseVLPreprocessor):

    def __init__(self, shape_meta: dict, model_config: dict):
        self.shape_meta = shape_meta
        self.model_config = model_config
        self.model_path = model_config["local_weights_path"]
        if self.model_path is None:
            # Load config (trust remote code to trigger local class registration)
            self.config = AutoConfig.from_pretrained(model_config["model_type"], trust_remote_code=True)
            self.model: LlavaLlamaModel = AutoModel.from_pretrained(
                model_config["model_type"],
                trust_remote_code=True,
                device_map="auto",
            )
        else : 
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.model: LlavaLlamaModel = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
            )

        # Tokenizer and vision processor
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"Tokenizer is not found on the loaded model. Please ensure {self.model_config['model_type']} provides tokenizer.")

        # Safe vocabulary sizing
        tokenizer_vocab_size = len(self.tokenizer.get_vocab()) if hasattr(self.tokenizer, 'get_vocab') else self.tokenizer.vocab_size
        model_vocab_size = self.model.llm.config.vocab_size
        embedding_vocab_size = self.model.llm.model.embed_tokens.num_embeddings
        print(f"[NVILA] Vocabulary sizes - Tokenizer: {tokenizer_vocab_size}, Model config: {model_vocab_size}, Embeddings: {embedding_vocab_size}")

        self.vocab_size = min(tokenizer_vocab_size, model_vocab_size, embedding_vocab_size)

    def _build_input_ids(self, instruction: str) -> torch.Tensor:
        """
        Vectorized construction of input_ids:
        [ encoded instruction ]
        No per-sample Python loops.
        """
        # TODO: fit chat template
        image_prompt = ["<image>\n"] * self.shape_meta["obs"]["rgb"]["horizon"]
        image_prompt = "".join(image_prompt)
        instruction = image_prompt + instruction
        messages = [
            {
                "role": "user",
                "content": instruction
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = np.array(prompt)  # [L_lang]
        input_ids = input_ids.clamp_(0, self.vocab_size - 1)  # safety

        return input_ids
    
    def __call__(
        self, 
        image: Union[torch.Tensor, np.ndarray], 
        instruction: str, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize the instruction. 
        Args:
            image: [H, W, C]
            instruction: str
            **kwargs: Additional arguments
        Returns:
          - input_ids: [L_lang]
          - image: [H, W, C]
        """
        return {
            "input_ids": self._build_input_ids(instruction),
            "image": image.numpy(),
        }
    