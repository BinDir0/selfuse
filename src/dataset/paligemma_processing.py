from math import e
from typing import Tuple
import warnings
import re

import torch
import numpy as np
import cv2

from src.utils.pytorch_util import dict_apply

IMAGENET_STANDARD_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGENET_STANDARD_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    eos_token,
    image_seq_len,
    image_token,
    suffix_target, 
    need_target = True,
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    if not need_target: 
        return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
    else:
        return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n{suffix_target}{eos_token}"


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
    is_depth: bool = False,
) -> np.ndarray:
    """Batch resize, supports RGB image, depth image and grayscale image.
    
    Args:
        image: np.ndarray - input image
            - [B, C, H, W]: RGB image (C=3) or grayscale image (C=1)
            - [B, H, W]: depth image
        size: Tuple[int, int] - (height, width)
        
    Returns:
        np.ndarray - resized image, keep the same dimension format as the input
    """
    height, width = size
    
    if image.ndim == 3:  # [B, H, W]
        resized_image = np.zeros((image.shape[0], height, width), dtype=image.dtype)
    elif image.ndim == 4:  # [B, C, H, W]
        resized_image = np.zeros((image.shape[0], image.shape[1], height, width), dtype=image.dtype)
    else: 
        raise ValueError(f"Invalid input dimension: {image.ndim}")
    is_gray = (image.shape[1] == 1)
    
    for b in range(image.shape[0]):
        img = image[b]
        if img.ndim == 3: # [C, H, W] -> [H, W, C]
            img = img.transpose(1, 2, 0)
        
        # Resize
        if is_depth: 
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        if is_gray: # [H, W] -> [H, W, 1]
            resized_img = resized_img[..., np.newaxis]
        # [H, W, C] -> [C, H, W]
        if img.ndim == 3:
            resized_image[b] = resized_img.transpose(2, 0, 1)
        else:
            resized_image[b] = resized_img
    
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


def process_depth_images(
    depth_images: np.ndarray,
    size: Tuple[int, int],
    rescale_factor: float = 1.0,
    image_mean: np.ndarray = IMAGENET_MEAN,
    image_std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Process depth images using numpy operations for CPU-based preprocessing.
    
    Args:
        depth_images: np.ndarray [T, H, W]
        size: Tuple[int, int] - target size (height, width)
        rescale_factor: float - scaling factor for pixel values (default 1.0)
        
    Returns:
        np.ndarray [T, 3, H, W] - processed depth images
    """
    # Convert to numpy if input is torch tensor
    if isinstance(depth_images, torch.Tensor):
        depth_images = depth_images.cpu().numpy()
    
    # Rescale the pixel values if needed
    if rescale_factor != 1.0:
        depth_images = rescale(depth_images, scale=rescale_factor)
    
    # Resize the depth images to the desired size using PIL
    depth_images = resize(depth_images, size=size, is_depth=True)
    
    # Normalize the depth images to have mean 0 and standard deviation 1
    depth_images = depth_images[:, np.newaxis, :, :] # [T, H, W] -> [T, 1, H, W]
    depth_images = np.tile(depth_images, (1, 3, 1, 1)) # [T, 1, H, W] -> [T, 3, H, W]
    depth_images = normalize(depth_images, mean=image_mean, std=image_std)
    return depth_images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"
    LOCALIZATION_TOKEN_NUM = 1024
    SEGMENTATION_TOKEN_NUM = 128

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        tokenizer_padding: str = "longest",  # longest or max_length
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
            f"<loc{i:04d}>" for i in range(self.LOCALIZATION_TOKEN_NUM)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(self.SEGMENTATION_TOKEN_NUM)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer('\n', max_length=1, padding="max_length", truncation=True)['input_ids'][0]

    def float2localization_tokens(self, text: str):
        pattern = r"\d+\.\d+"

        def replacer(match: re.Match) -> str:
            # match.group(1) is the captured floating point string, e.g. "0.123"
            float_str = match.group(0)
            
            # Convert the string to a floating point number
            float_val = float(float_str)
            
            # Clip the floating point number to [0, 1] range, then scale and discretize to an integer
            token_id = int(np.clip(float_val, 0, 1) * self.LOCALIZATION_TOKEN_NUM)
            
            # Build and return the formatted token string
            return f"<loc{token_id:04d}>"

        # Use re.sub to globally find and replace
        return re.sub(pattern, replacer, text)

    def localization_tokens2float(self, text: str):
        pattern = r"<loc\d{4}>"
        def replacer(match: re.Match) -> str:
            loc_token = match.group(0)
            loc_token_id = int(loc_token.split('<loc')[-1].split('>')[0])
            return f"{loc_token_id / self.LOCALIZATION_TOKEN_NUM:.4f}"
        return re.sub(pattern, replacer, text)

    def __call__(
        self,
        text: str,
        images: np.ndarray,
        target: str, 
        truncation: bool = True,
        mode = 'train',
    ) -> dict:
        '''
        Args: 
            text: str
            images: np.ndarray [T, C, H, W] or [T, H, W, C]
            target: str
            truncation: bool
            mode: str
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
        text = text.replace('<image>', '')
        target = target.replace('<image>', '')
        # We assume the floating point numbers in the text and target are localization tokens
        text = self.float2localization_tokens(text)
        target = self.float2localization_tokens(target)
        input_string = add_image_tokens_to_prompt(
            prefix_prompt=text,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            image_seq_len=self.image_seq_length * images.shape[0],
            image_token=self.IMAGE_TOKEN,
            suffix_target=target,
            need_target=True,
        )

        if mode == 'infer': # Use left padding for inference
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.padding_side = "right"
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
        if not np.any(condition):
            if mode != 'infer': # Do not warn in inference mode
                warnings.warn("The separator token is not found in the input_ids")
            sep_idx = len(labels) - 1
        else : 
            sep_idx = np.argmax(condition)
        labels[:sep_idx+1] = self.ignore_index
        inputs['labels'] = labels
        inputs['answer_start_idx'] = np.array(sep_idx+1)
        
        output = {"pixel_values": pixel_values, **inputs}
        return output

    def decode(self, output_ids):
        output_str = self.tokenizer.decode(output_ids)
        return self.localization_tokens2float(output_str)


class PaliGemmaVLAProcessor(PaliGemmaProcessor):
    STATE_TOKEN = "<state>"
    ACTION_TOKEN = "<action>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        depth_image_size: int = 224, 
        tokenizer_padding: str = "longest", # longest or max_length
    ):
        super().__init__(
            tokenizer = tokenizer,
            num_image_tokens = num_image_tokens,
            max_seq_len = max_seq_len,
            ignore_index = ignore_index,
            image_size = image_size,
            tokenizer_padding = tokenizer_padding,
        )
        self.depth_image_size = depth_image_size
        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [
            self.STATE_TOKEN,
            self.ACTION_TOKEN, 
        ]}
        tokenizer.add_special_tokens(tokens_to_add)
        self.state_token_id = tokenizer.convert_tokens_to_ids(self.STATE_TOKEN)
        self.action_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_TOKEN)
        self.eos_token_id = tokenizer.eos_token_id
    
    def __call__(
        self,
        text: str,
        images: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        intrinsic: np.ndarray,
        objective: str = None,
        truncation: bool = True,
        depth_images: np.ndarray = None,
        mode: str = 'train',
    ) -> dict:
        '''
        Args: 
            text: str
            images: np.ndarray [T_image, C, H, W] or [T_image, H, W, C]
            state: np.ndarray [T_state, state_dim]
            action: np.ndarray [Horizon, action_dim]
            intrinsic: np.ndarray [4]
            objective: str, 'ar' or 'flow' or None, None means both
            truncation: bool
            depth_images: np.ndarray [T_image, 3, H, W]
            mode: str
        Returns:
            dict:
                - pixel_values: torch.FloatTensor [T_image, C, H, W]
                - depth_values: torch.FloatTensor [T_image, 1, H, W] (if depth_images provided)
                - input_ids: torch.LongTensor [L]
                - labels: torch.LongTensor [L]
                - attention_mask: torch.LongTensor [L]
        '''
        if images.dtype == np.uint8:
            scale_factor = 1 / 255.0
        else:
            scale_factor = 1.0

        if images.shape[-1] == 3: 
            original_height, original_width = images.shape[1], images.shape[2]
        else:
            original_height, original_width = images.shape[2], images.shape[3]
        intrinsic = get_resized_intrinsic(intrinsic, original_width, original_height, self.image_size)
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            rescale_factor=scale_factor,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        
        # Process depth images if provided
        depth_values = None
        if depth_images is not None:
            # Determine scale factor for depth (typically depth is in meters, normalize to [0, 1])
            # Adjust this based on your depth data range
            depth_scale_factor = 1.0  # No rescaling by default, adjust if needed
            depth_values = process_depth_images(
                depth_images=depth_images,
                size=(self.depth_image_size, self.depth_image_size),
                rescale_factor=depth_scale_factor,
                image_mean=IMAGENET_MEAN,
                image_std=IMAGENET_STD,
            )

        intrinsic_str = f"fx:{intrinsic[0]:.2f} fy:{intrinsic[1]:.2f} cx:{intrinsic[2]:.2f} cy:{intrinsic[3]:.2f}"
        text = text.replace('.', '')
        text = text.lower()
        prefix = (
            f"Task: {text}, Camera intrinsic: {intrinsic_str}, "
            f"States: {self.STATE_TOKEN * len(states)}"
            f"Action: "
        )
        suffix = f"{self.ACTION_TOKEN * len(actions)}"
        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_string = add_image_tokens_to_prompt(
            prefix_prompt=prefix,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            image_seq_len=self.image_seq_length * images.shape[0],
            image_token=self.IMAGE_TOKEN,
            suffix_target=suffix,
            need_target=(mode != 'infer' and objective != "train_flow"),
        )

        if mode == 'infer': # Use left padding for inference
            self.tokenizer.padding_side = "left" 
        else:
            self.tokenizer.padding_side = "right"
        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_string,
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )
        inputs = dict_apply(inputs, lambda x: np.array(x))
        
        input_ids = inputs['input_ids'] # [L]
        assert input_ids.ndim == 1, f"The input_ids should be 1D array, got {input_ids.ndim}D array."

        labels = np.ones_like(input_ids) * self.ignore_index # Dummy labels for continuous action prediction
        inputs['labels'] = labels
        assert np.any(input_ids == self.sep_token_id), "The separator token is not found in the input_ids"
        sep_idx = np.argmax(input_ids == self.sep_token_id)
        inputs['answer_start_idx'] = np.array(sep_idx+1)

        output = {"pixel_values": pixel_values, **inputs}
        if depth_values is not None:
            output["depth_values"] = depth_values
        return output


def get_resized_intrinsic(intrinsic, original_width: int, original_height: int, img_size: int = 224): 
    '''
    Return intrinsic after resizing the images. 
    Args: 
        intrinsic: np.ndarray [..., 4]
        original_width: int
        original_height: int
        img_size: int = 224
    Returns: 
        intrinsic: np.ndarray [..., 4]
    '''
    scale_x = img_size / original_width
    scale_y = img_size / original_height
    intrinsic = intrinsic * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
    return intrinsic
