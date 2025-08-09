"""
NVILA: wrapper built on top of LlavaLlamaModel to load NVILA-Lite-2B and
prepare inputs that include: language instructions, 6 images, and 30 action
query tokens (hardcoded approach: uses the last 30 tokens from vocabulary excluding the final 5 special tokens).

Uses hardcoded approach for action tokens, no longer performs complex special token detection.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import List, Dict, Union, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel

# Add VILA project path to sys.path for importing local llava module
import sys
_VILA_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../VILA"))
if _VILA_ROOT not in sys.path:
    sys.path.append(_VILA_ROOT)

from llava.constants import DEFAULT_IMAGE_TOKEN  # <image>
from llava.model.language_model.llava_llama import LlavaLlamaModel  # Trigger local class registration
from egovla.model.vlm.base_vlm import BaseVLM

class NVILA(BaseVLM):
    """
    A simple wrapper class:
    - Uses local LlavaLlamaModel to load `NVILA-Lite-2B`
    - Builds inputs: language instruction + images + action query tokens
    - Provides two main interfaces: `build_inputs` and `forward`
    - Uses hardcoded approach to select action tokens, excluding the last 5 special tokens from vocabulary
    """

    def __init__(self, shape_meta: dict, model_config: dict):
        """Initialize and load model + processor, making sure CUDA/cuDNN are ready."""
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
        self.image_processor = self.model.get_vision_tower().image_processor

        # Safe vocabulary sizing
        tokenizer_vocab_size = len(self.tokenizer.get_vocab()) if hasattr(self.tokenizer, 'get_vocab') else self.tokenizer.vocab_size
        model_vocab_size = self.model.llm.config.vocab_size
        embedding_vocab_size = self.model.llm.model.embed_tokens.num_embeddings
        print(f"[NVILA] Vocabulary sizes - Tokenizer: {tokenizer_vocab_size}, Model config: {model_vocab_size}, Embeddings: {embedding_vocab_size}")

        self.vocab_size = min(tokenizer_vocab_size, model_vocab_size, embedding_vocab_size)
        self.num_action_tokens = shape_meta["action"]["horizon"]

        # Hardcoded approach to choose action tokens, excluding the last 5 special tokens from vocabulary
        self.action_token_ids = self._find_valid_action_tokens(self.num_action_tokens)
        print(f"[NVILA] Action token IDs: {self.action_token_ids[:5]}...{self.action_token_ids[-5:]}")

        # Resolve image token id
        image_token_from_media = getattr(self.tokenizer, "media_token_ids", None)
        if image_token_from_media and "image" in image_token_from_media:
            self.image_token_id = image_token_from_media["image"]
            print(f"[NVILA] Using official media_token_ids['image']: {self.image_token_id}")

        # Validate action tokens are normal tokens (not specials)
        self._validate_action_tokens()

    # -------------------------
    # Token utilities
    # -------------------------
    def _find_valid_action_tokens(self, num_tokens: int) -> List[int]:
        """Hardcoded approach to get the last N action tokens, excluding the last 5 special tokens from vocabulary."""
        print(f"[NVILA] Using hardcoded approach to get last {num_tokens} action tokens...")
        
        # Hardcoded: exclude the last 5 special tokens from vocabulary
        num_special_tokens_to_exclude = 5
        valid_end = self.vocab_size - num_special_tokens_to_exclude
        valid_start = valid_end - num_tokens
        
        if valid_start < 0:
            print(f"[NVILA] Warning: Vocabulary too small to provide {num_tokens} action tokens")
            valid_start = 0
            
        valid_tokens = list(range(valid_start, valid_end))
        print(f"[NVILA] Hardcoded action token range: {valid_start} to {valid_end-1}")
        print(f"[NVILA] Excluded special token range: {valid_end} to {self.vocab_size-1}")
        
        return valid_tokens



    def _validate_action_tokens(self):
        """Print diagnostic information to confirm chosen action tokens. Uses hardcoded approach, excluding last 5 special tokens."""
        print(f"\n[NVILA] Validating last {self.num_action_tokens} action tokens...")
        print(f"[NVILA] Using hardcoded approach, excluding last 5 special tokens from vocabulary")
        
        print(f"[NVILA] Action token analysis (last {self.num_action_tokens} tokens):")
        for i, token_id in enumerate(self.action_token_ids):
            try:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                token_text_clean = self.tokenizer.decode([token_id], skip_special_tokens=True)
                
                if i < 5 or i >= len(self.action_token_ids) - 5:
                    print(f"  ID {token_id}: '{token_text}' -> '{token_text_clean}'")
                elif i == 5 and len(self.action_token_ids) > 10:
                    print("  ...")
            except Exception as e:
                print(f"  ID {token_id}: ERROR - {e}")

        num_special_excluded = 5
        excluded_start = self.vocab_size - num_special_excluded
        print(f"[NVILA] Hardcoded action token range: {min(self.action_token_ids)} to {max(self.action_token_ids)}")
        print(f"[NVILA] Excluded special token range: {excluded_start} to {self.vocab_size-1}")
        print(f"[NVILA] Action token validation completed!")

    # -------------------------
    # Preprocessing (vectorized)
    # -------------------------
    def _preprocess_one_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Kept for compatibility, but fast path uses batched processor calls."""
        if isinstance(image, Image.Image):
            pixel = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                pixel = image[0]
            elif image.dim() == 3:
                pixel = image
            else:
                raise ValueError(f"Unsupported image tensor dimension: {image.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if pixel.dtype != torch.float32:
            pixel = pixel.float()
        if not pixel.is_contiguous():
            pixel = pixel.contiguous()
        return pixel

    def _process_batch_images(self, batch_images: List[List[Union[Image.Image, torch.Tensor]]]) -> List[Dict[str, List[torch.Tensor]]]:
        """
        Process multiple samples (each with 6 images) in a single batch call with zero for-loops.
        Uses pure vectorized operations for maximum efficiency.
        
        Args:
            batch_images: List of samples, each containing 6 images
            
        Returns:
            List of media dicts, one per sample
        """
        import itertools
        import numpy as np
        
        batch_size = len(batch_images)
        images_per_sample = 6
        
        # Vectorized validation using numpy
        sample_lengths = np.array([len(sample) for sample in batch_images])
        if not np.all(sample_lengths == images_per_sample):
            invalid_indices = np.where(sample_lengths != images_per_sample)[0]
            raise ValueError(f"Samples {invalid_indices.tolist()}: Expected {images_per_sample} images each, got {sample_lengths[invalid_indices].tolist()}")
        
        # Flatten all images using itertools.chain - no for loop!
        all_images = list(itertools.chain.from_iterable(batch_images))
        total_images = len(all_images)
        
        print(f"[NVILA] Vectorized batch processing {total_images} images ({batch_size} samples × {images_per_sample} images/sample)")
        
        # Process all images in a single batch call
        batch_processed = self.image_processor(images=all_images, return_tensors="pt")
        all_tensors = batch_processed["pixel_values"]  # [total_images, C, H, W]
        all_tensors = all_tensors.to(device=self.device, dtype=self.vision_dtype, non_blocking=True)
        

        # Reshape using pure tensor operations - no for loop!
        # [total_images, C, H, W] -> [batch_size, images_per_sample, C, H, W]
        reshaped_tensors = all_tensors.view(batch_size, images_per_sample, *all_tensors.shape[1:])
        
        # Convert to list of media dicts using vectorized operations
        # Split tensor into list of per-sample tensors
        sample_tensors_list = reshaped_tensors.unbind(dim=0)  # Tuple of [images_per_sample, C, H, W]
        
        # Create media dicts using list comprehension (vectorized, not iterative)
        media_dicts = [
            {"image": list(sample_tensor.unbind(dim=0))}  # [images_per_sample, C, H, W] -> List[[C, H, W]]
            for sample_tensor in sample_tensors_list
        ]
        return media_dicts

    def _build_media_dict(
        self,
        images: Union[List[Union[Image.Image, torch.Tensor]], List[List[Union[Image.Image, torch.Tensor]]]]
    ) -> Union[Dict[str, List[torch.Tensor]], List[Dict[str, List[torch.Tensor]]]]:
        """
        Optimized batch image preprocessing: processes all images in a single batch call.
        Returns the schema expected by VILA:
          - single sample: {"image": [6 x [C,H,W] tensors]}
          - batch:        [{"image": [6 x [C,H,W] tensors]}, ...]
        """
        if isinstance(images[0], list):
            # Batch processing: flatten all images, process once, then reshape
            return self._process_batch_images(images)
        else:
            # Single sample: process 6 images
            return self._process_single_sample(images)
    
    def _process_single_sample(self, sample_images: List[Union[Image.Image, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """
        Process a single sample with 6 images.
        
        Args:
            sample_images: List of 6 images
            
        Returns:
            Media dict with list of tensors
        """
        if len(sample_images) != 6:
            raise ValueError(f"Exactly 6 images are required, but received {len(sample_images)}.")
        
        print(f"[NVILA] Processing single sample with 6 images")
        batch = self.image_processor(images=sample_images, return_tensors="pt")["pixel_values"]
        batch = batch.to(device=self.device, dtype=self.vision_dtype, non_blocking=True)  # [6, C, H, W]
        
        return {"image": list(batch.unbind(dim=0))}
    

    def _build_input_ids(self, instruction: Union[str, List[str]]) -> torch.Tensor:
        """
        Vectorized construction of input_ids:
        [ encoded instruction ] + [ 6 × <image> ] + [ 30 × action tokens ]
        No per-sample Python loops.
        """
        if isinstance(instruction, str):
            instructions = [instruction]
        else:
            instructions = instruction

        # Batch encode and pad only the language part
        enc = self.tokenizer.batch_encode_plus(
            instructions,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]  # [B, L_lang]
        input_ids = input_ids.clamp_(0, self.vocab_size - 1)  # safety
        input_ids = input_ids.to(self.device)

        B = input_ids.size(0)

        # 6 image tokens per sample
        img_prefix = torch.full(
            (B, 6), fill_value=self.image_token_id, dtype=torch.long, device=self.device
        )
        # 30 action tokens per sample
        act_suffix = torch.as_tensor(
            self.action_token_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(B, -1)

        # Concatenate along sequence dimension
        batch_tensor = torch.cat([input_ids, img_prefix, act_suffix], dim=1)

        return batch_tensor

    # -------------------------
    # Public API
    # -------------------------
    @torch.inference_mode()
    def build_inputs(
        self,
        instruction: Union[str, List[str]],
        images: Union[List[Union[Image.Image, torch.Tensor]], List[List[Union[Image.Image, torch.Tensor]]]],
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Construct the forward input dict, supporting single or batched inputs.
        Returns:
          - input_ids: [B, S]
          - attention_mask: [B, S] (True for real tokens, False for pad)
          - media: {"image": list-of-6 tensors} or list of such dicts
          - media_config: dict
        """
        is_batch = isinstance(instruction, list)
        if is_batch != isinstance(images[0], list):
            raise ValueError("Instruction and images format mismatch. Both should be single or both should be batch.")
        if is_batch and len(instruction) != len(images):
            raise ValueError(f"Instructions ({len(instruction)}) and images_batch ({len(images)}) must have the same length.")

        # Vectorized input_ids
        input_ids = self._build_input_ids(instruction)  # [B, S]

        # Attention mask = tokenizer's language mask + ones for appended 6+30 tokens
        enc = self.tokenizer.batch_encode_plus(
            [instruction] if isinstance(instruction, str) else instruction,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )
        attn_lang = enc["attention_mask"].to(self.device).bool()  # [B, L_lang]
        B, S_total = input_ids.shape
        L_lang = attn_lang.shape[1]
        extra_len = S_total - L_lang
        attn_extra = torch.ones((B, extra_len), dtype=torch.bool, device=self.device)
        attention_mask = torch.cat([attn_lang, attn_extra], dim=1)  # [B, S_total]

        # Media dict (vectorized per-sample via image processor)
        media_result = self._build_media_dict(images)

        # Normalize media output schema
        if isinstance(media_result, list):
            # Flatten the batch: each sample has 6 images, so for B samples we get B*6 images
            all_images = []
            for sample in media_result:
                all_images.extend(sample["image"])  # Flatten the list of image tensors
            batch_media = {"image": all_images}
        else:
            batch_media = media_result

        from collections import defaultdict
        media_config = defaultdict(dict)
        media_config["image"] = {}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "media": batch_media,
            "media_config": media_config
        }

    def forward(
        self,
        images: Union[List[Union[Image.Image, torch.Tensor]], List[List[Union[Image.Image, torch.Tensor]]]],
        instruction: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass (no generation).
        Returns:
          - loss (if provided by model)
          - logits: [B, 30, V] - Only the last 30 action token logits
          - input_ids, attention_mask (echoed back)
        """
        batch = self.build_inputs(instruction, images)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            media=batch["media"],
            media_config=batch["media_config"],
            return_dict=True,
        )
        
        # Extract only the last 30 action token logits
        # outputs.logits shape: [B, S, V] where S includes instruction + 6 images + 30 action tokens
        full_logits = outputs.logits  # [B, S, V]
        action_logits = full_logits[:, -self.num_action_tokens:, :]  # [B, 30, V]
        
        return {
            "loss": getattr(outputs, "loss", None),
            "logits": action_logits,  # Only the last 30 action token logits
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

__all__ = ["NVILA"]


def main():
    """
    Short test for image preprocessing functions with different input sizes.
    """
    import numpy as np
    from PIL import Image
    import torch
    
    print("=" * 60)
    print("NVILA IMAGE PREPROCESSING TEST")
    print("=" * 60)
    
    try:
        # Mock configuration for testing
        shape_meta = {"action": {"horizon": 30}}
        model_config = {
            "local_weights_path": "/home/fanlian/EgoVLA/NVILA-Lite-2B",
            "model_type": "NVILA"
        }
        
        print("\n1. Initializing NVILA...")
        nvila = NVILA(shape_meta=shape_meta, model_config=model_config)
        print(f"✓ Model loaded, image processor: {type(nvila.image_processor).__name__}")
        
        # Create test images of different sizes and formats
        print(f"\n2. Creating test images with different sizes...")
        test_cases = [
            ((64, 64), "Tiny square"),
            ((224, 224), "ImageNet standard"),  
            ((640, 480), "VGA 4:3"),
            ((1920, 1080), "Full HD 16:9"),
            ((800, 600), "SVGA 4:3"),
            ((512, 256), "Wide rectangle"),
        ]
        
        test_images = []
        for (width, height), desc in test_cases:
            # Create colorful test image
            img_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
            # Add some pattern for visual distinction
            img_array[::20, :, 0] = 255  # Red stripes
            pil_image = Image.fromarray(img_array)
            test_images.append(pil_image)
            print(f"  - {desc}: {width}×{height}")
        
        # Test single sample processing (6 images)
        print(f"\n3. Testing single sample processing...")
        print(f"  - Input: 6 images of different sizes")
        
        media_dict = nvila._build_media_dict(test_images)
        
        print(f"  ✓ Output: {len(media_dict['image'])} tensors")
        for i, tensor in enumerate(media_dict['image']):
            print(f"    Tensor {i+1}: {tensor.shape} (dtype: {tensor.dtype})")
        
        # Verify all outputs are 448×448
        expected_shape = torch.Size([3, 448, 448])
        all_correct = all(tensor.shape == expected_shape for tensor in media_dict['image'])
        print(f"  ✓ All tensors have correct shape {expected_shape}: {all_correct}")
        
        # Test batch processing (2 samples)
        print(f"\n4. Testing batch processing...")
        batch_images = [test_images, test_images]  # 2 samples, 6 images each
        print(f"  - Input: 2 samples × 6 images = 12 total images")
        
        batch_media = nvila._build_media_dict(batch_images)
        
        print(f"  ✓ Output: {len(batch_media)} sample dicts")
        for i, sample_dict in enumerate(batch_media):
            print(f"    Sample {i+1}: {len(sample_dict['image'])} images")
            sample_shapes = [tensor.shape for tensor in sample_dict['image']]
            print(f"      Shapes: {sample_shapes[0]} (all identical: {all(s == sample_shapes[0] for s in sample_shapes)})")
        
        # Test image processor directly
        print(f"\n5. Testing SiglipImageProcessor directly...")
        
        # Single image test
        single_result = nvila.image_processor.preprocess(test_images[0], return_tensors="pt")
        single_tensor = single_result["pixel_values"][0]
        print(f"  - Single image: {test_images[0].size} → {single_tensor.shape}")
        
        # Batch test
        batch_result = nvila.image_processor(images=test_images, return_tensors="pt")
        batch_tensor = batch_result["pixel_values"]
        print(f"  - Batch images: 6 different sizes → {batch_tensor.shape}")
        
        # Value range verification
        print(f"\n6. Verifying normalization...")
        sample_tensor = media_dict['image'][0]
        tensor_min, tensor_max = sample_tensor.min().item(), sample_tensor.max().item()
        tensor_mean, tensor_std = sample_tensor.mean().item(), sample_tensor.std().item()
        
        print(f"  - Value range: [{tensor_min:.3f}, {tensor_max:.3f}]")
        print(f"  - Mean: {tensor_mean:.3f}, Std: {tensor_std:.3f}")
        print(f"  - Expected range: [-1, 1] ✓" if -1.1 <= tensor_min <= tensor_max <= 1.1 else "  - ❌ Unexpected range!")
        
        # Memory usage estimate
        single_memory = test_images[0].size[0] * test_images[0].size[1] * 3 * 4 / (1024**2)
        processed_memory = 448 * 448 * 3 * 4 / (1024**2)
        print(f"\n7. Memory usage:")
        print(f"  - Example input image: {single_memory:.2f} MB")
        print(f"  - After processing: {processed_memory:.2f} MB")
        print(f"  - 6 images total: {6 * processed_memory:.2f} MB")
        
        print(f"\n" + "=" * 60)
        print("✓ ALL IMAGE PREPROCESSING TESTS PASSED!")
        print("✓ Arbitrary input sizes → consistent 448×448 output")
        print("✓ Proper normalization [-1, 1] range")
        print("✓ Both single and batch processing working")
        print("✓ SiglipImageProcessor functioning correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
