from typing import Dict, List, Tuple
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.utils.pytorch_util import dict_apply

IMAGENET_STANDARD_MEAN = np.array([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = np.array([0.5, 0.5, 0.5])


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    eos_token,
    image_seq_len,
    image_token,
    suffix_target, 
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n{suffix_target}{eos_token}"


def add_image_action_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    eos_token,
    image_seq_len,
    image_token,
    action_begin_token,
    action_end_token,
    action_token,
    action_seq_len,
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n{action_begin_token}{action_token * action_seq_len}{action_end_token}{eos_token}"


def rescale(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Rescale image pixel values by a factor.
    
    Args:
        image: np.ndarray [B, H, W, C] or [B, C, H, W]
        scale: float - scaling factor
        
    Returns:
        np.ndarray - rescaled image
    """
    rescaled_image = image.astype(np.float32) * scale
    return rescaled_image


def resize(
    image: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """Highly optimized batch resize using PIL.Image with memory efficiency.
    
    Args:
        image: np.ndarray [B, C, H, W]
        size: Tuple[int, int] - (height, width)
        
    Returns:
        np.ndarray - resized image
    """
    height, width = size
    batch_size, channels, orig_height, orig_width = image.shape
    
    resized_image = np.zeros((batch_size, channels, height, width), dtype=np.float32)
    temp_hwc = np.zeros((orig_height, orig_width, channels), dtype=np.uint8)
    
    for b in range(batch_size):
        # Convert from [C, H, W] to [H, W, C] for PIL
        img_hwc = np.transpose(image[b], (1, 2, 0))
        
        # Handle different input data types efficiently
        if img_hwc.dtype == np.uint8:
            pil_image = Image.fromarray(img_hwc)
        elif img_hwc.dtype in [np.float32, np.float64]:
            pil_image = Image.fromarray((img_hwc * 255).astype(np.uint8))
        
        resized_pil = pil_image.resize((width, height), Image.Resampling.BILINEAR)
        
        resized_hwc = np.array(resized_pil, dtype=np.float32) / 255.0
        
        # Convert back to [C, H, W]
        resized_image[b] = np.transpose(resized_hwc, (2, 0, 1))
    
    return resized_image


def normalize(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize image using mean and std.
    
    Args:
        image: np.ndarray [B, C, H, W]
        mean: np.ndarray [C] - mean values for each channel
        std: np.ndarray [C] - std values for each channel
        
    Returns:
        np.ndarray - normalized image
    """
    assert image.ndim == 4, f"Expected 4D array, got {image.ndim}D array."
    assert (
        image.shape[1] == 3
    ), f"Expected 3 channels at axis 1, got {image.shape[1]} channels."
    
    # Add batch and spatial dimensions for broadcasting
    mean = mean[None, :, None, None]  # [1, C, 1, 1]
    std = std[None, :, None, None]    # [1, C, 1, 1]
    
    # Normalize
    normalized_image = (image - mean) / std
    return normalized_image


def process_images(
    images: np.ndarray,
    size: Tuple[int, int],
    rescale_factor: float,
    image_mean: np.ndarray,
    image_std: np.ndarray,
) -> np.ndarray:
    """Process images using numpy operations for CPU-based preprocessing.
    
    Args:
        images: np.ndarray [B, H, W, C] or [B, C, H, W]
        size: Dict[str, int] - target size with 'height' and 'width' keys
        rescale_factor: float - scaling factor for pixel values
        image_mean: np.ndarray [C] - mean values for normalization
        image_std: np.ndarray [C] - std values for normalization
        
    Returns:
        np.ndarray [B, C, H, W] - processed images
    """
    # Convert to numpy if input is torch tensor
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    # Ensure images are in [B, C, H, W] format
    if images.shape[-1] == 3:
        images = np.transpose(images, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
    
    # Rescale the pixel values to be in the range [0, 1]
    images = rescale(images, scale=rescale_factor)
    
    # Resize the images to the desired size using PIL for high quality
    images = resize(images, size=size)
    
    # Normalize the images to have mean 0 and standard deviation 1
    images = normalize(images, mean=image_mean, std=image_std)
    
    return images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [
            self.IMAGE_TOKEN, 
        ]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer('\n', max_length=1, padding="max_length", truncation=True)['input_ids'][0]

    def __call__(
        self,
        text: str,
        images: np.ndarray,
        target: str, 
        truncation: bool = True,
    ) -> dict:
        '''
        Args: 
            text: str
            images: np.ndarray [T, C, H, W] or [T, H, W, C]
            target: str
            truncation: bool

        Returns:
            dict:
                - pixel_values: torch.FloatTensor [T, C, H, W]
                - input_ids: torch.LongTensor [L]
                - labels: torch.LongTensor [L]
                - attention_mask: torch.LongTensor [L]
        '''
        if images.dtype == np.uint8:
            scale_factor = 1 / 255.0
        else:
            scale_factor = 1.0

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            rescale_factor=scale_factor,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        text = text.replace('\n', '')
        target = target.replace('\n', '')
        input_string = add_image_tokens_to_prompt(
            prefix_prompt=text,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            image_seq_len=self.image_seq_length * images.shape[0],
            image_token=self.IMAGE_TOKEN,
            suffix_target=target,
        )

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_string,
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        inputs = dict_apply(inputs, lambda x: np.array(x))

        labels = inputs['input_ids'].copy()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        condition = (labels == self.sep_token_id)
        assert np.any(condition), "The separator token is not found in the input_ids"
        sep_idx = np.argmax(condition)
        labels[:sep_idx+1] = self.ignore_index
        inputs['labels'] = labels
        inputs['answer_start_idx'] = np.array(sep_idx+1)
        
        output = {"pixel_values": pixel_values, **inputs}
        return output


class PaliGemmaVLAProcessor:
    IMAGE_TOKEN = "<image>"
    STATE_BEGIN_TOKEN = "<state_begin>"
    STATE_TOKEN = "<state>"
    STATE_END_TOKEN = "<state_end>"
    HUMAN_ACTION_BEGIN_TOKEN = "<human_action_begin>"
    HUMAN_ACTION_TOKEN = "<human_action>"
    HUMAN_ACTION_END_TOKEN = "<human_action_end>"

    def __init__(
        self,
        tokenizer,
        fast_tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        tokenizer_padding: str = "max_length",  #  # instead of truncating to longest
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [
            self.IMAGE_TOKEN, 
            self.STATE_BEGIN_TOKEN, 
            self.STATE_TOKEN,
            self.STATE_END_TOKEN, 
            self.HUMAN_ACTION_BEGIN_TOKEN, 
            self.HUMAN_ACTION_TOKEN, 
            self.HUMAN_ACTION_END_TOKEN,
        ]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.state_token_id = tokenizer.convert_tokens_to_ids(self.STATE_TOKEN)
        self.human_action_token_id = tokenizer.convert_tokens_to_ids(self.HUMAN_ACTION_TOKEN)
        self.human_action_begin_token_id = tokenizer.convert_tokens_to_ids(self.HUMAN_ACTION_BEGIN_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        self.fast_tokenizer = fast_tokenizer

        self.fast_token_id2gemma_token_id = {
            k: {} for k in self.fast_tokenizer.keys()
        }
        sum_fast_vocab_size = sum([self.fast_tokenizer[k].vocab_size for k in self.fast_tokenizer.keys()])
        vocab_size = self.tokenizer.vocab_size
        special_tokens = self.tokenizer.all_special_ids
        token_id_replace = []
        replace_size = 0
        for i in range(vocab_size - 1, -1, -1):
            if i in special_tokens:
                continue
            token_id_replace.append(i)
            replace_size += 1
            if replace_size >= sum_fast_vocab_size:
                break
        assert replace_size == sum_fast_vocab_size, "The replace size is not equal to the sum of the fast tokenizer vocab size"

        token_id_replace = token_id_replace[::-1]
        replace_id = 0
        for k in self.fast_tokenizer.keys():
            tk = self.fast_tokenizer[k]
            for i in range(tk.vocab_size):
                self.fast_token_id2gemma_token_id[k][i] = token_id_replace[replace_id]
                replace_id += 1
        self.gemma_token_id2fast_token_id = {}
        for key, id_map in self.fast_token_id2gemma_token_id.items():
            self.gemma_token_id2fast_token_id[key] = {v: k for k, v in id_map.items()}

    def __call__(
        self,
        text: str,
        images: np.ndarray,
        states: np.ndarray,
        human_actions: np.ndarray,
        truncation: bool = True,
    ) -> dict:
        '''
        Args: 
            text: str
            images: np.ndarray [T_image, C, H, W] or [T_image, H, W, C]
            state: np.ndarray [T_state, state_dim]
            human_action: np.ndarray [Horizon, human_action_dim]
            truncation: bool

        Returns:
            dict:
                - pixel_values: torch.FloatTensor [T_image, C, H, W]
                - input_ids: torch.LongTensor [L]
                - labels: torch.LongTensor [L]
                - attention_mask: torch.LongTensor [L]
        '''
        if images.dtype == np.uint8:
            scale_factor = 1 / 255.0
        else:
            scale_factor = 1.0

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            rescale_factor=scale_factor,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # We assume the states and human actions are in [-1, 1]
        discrete_states = self.fast_tokenizer['states'](states)[0]
        discrete_human_actions = self.fast_tokenizer['human_actions'](human_actions)[0]

        text = text.replace('.', '')
        text = text.lower()
        text = f"What should the robot do to {text} with the state {self.STATE_BEGIN_TOKEN}{self.STATE_TOKEN * len(discrete_states)}{self.STATE_END_TOKEN}?"
        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_string = add_image_action_tokens_to_prompt(
            prefix_prompt=text,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            image_seq_len=self.image_seq_length * images.shape[0],
            image_token=self.IMAGE_TOKEN,
            action_begin_token=self.HUMAN_ACTION_BEGIN_TOKEN,
            action_end_token=self.HUMAN_ACTION_END_TOKEN,
            action_token=self.HUMAN_ACTION_TOKEN,
            action_seq_len=len(discrete_human_actions),
        )

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_string,
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        inputs = dict_apply(inputs, lambda x: np.array(x))
        
        discrete_states = np.array([self.fast_token_id2gemma_token_id['states'][id] for id in discrete_states])
        discrete_human_actions = np.array([self.fast_token_id2gemma_token_id['human_actions'][id] for id in discrete_human_actions])
        input_ids = inputs['input_ids']
        input_ids = set_token_id(input_ids, self.state_token_id, discrete_states)
        input_ids = set_token_id(input_ids, self.human_action_token_id, discrete_human_actions)
        inputs['input_ids'] = input_ids

        labels = input_ids.copy()
        answer_start_idx = np.argmax(labels == self.human_action_begin_token_id) + 1
        labels[:answer_start_idx] = self.ignore_index
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        inputs['labels'] = labels
        inputs['answer_start_idx'] = np.array(answer_start_idx)

        output = {"pixel_values": pixel_values, **inputs}
        return output


def set_token_id(input_ids, token_id, discrete_tokens):
    condition = (input_ids == token_id)
    available_tokens = np.sum(condition)
    if available_tokens != len(discrete_tokens):
        warnings.warn(f"The number of tokens to set is not equal to the number of discrete tokens. {np.sum(condition)} != {len(discrete_tokens)}")
    
    input_ids[condition] = discrete_tokens[:available_tokens]
    return input_ids
