from math import e
from typing import Tuple
import warnings
import re

import torch
import numpy as np
import cv2

from src.utils.pytorch_util import dict_apply

IMAGENET_STANDARD_MEAN = np.array([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = np.array([0.5, 0.5, 0.5])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


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
    need_action = True, # if True, the action tokens will be added, otherwise only the image tokens will be added
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    if need_action:
        return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n{action_begin_token}{action_token * action_seq_len}{action_end_token}{eos_token}"
    else:
        return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n{eos_token}"


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
    depth_images = resize(depth_images, size=size)
    
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
        if not np.any(condition):
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


class PaliGemmaVLAProcessor:
    IMAGE_TOKEN = "<image>"
    STATE_BEGIN_TOKEN = "<state_begin>"
    STATE_TOKEN = "<state>"
    STATE_END_TOKEN = "<state_end>"
    ACTION_BEGIN_TOKEN = "<action_begin>"
    ACTION_TOKEN = "<action>"
    ACTION_END_TOKEN = "<action_end>"

    def __init__(
        self,
        tokenizer,
        motion_tokenizer,
        num_image_tokens: int,
        num_depth_image_tokens: int,
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        depth_image_size: int = 224, 
        tokenizer_padding: str = "longest", # longest or max_length
        last_skip_tokens: int = 128, # the number of tokens to skip at the end of the vocabulary
    ):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.depth_image_seq_length = num_depth_image_tokens
        self.image_size = image_size
        self.depth_image_size = depth_image_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.tokenizer_padding = tokenizer_padding
        self.wrist_dim = 9
        self.hand_dim = 15
        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [
            self.IMAGE_TOKEN,
            self.STATE_BEGIN_TOKEN,
            self.STATE_TOKEN,
            self.STATE_END_TOKEN,
            self.ACTION_BEGIN_TOKEN, 
            self.ACTION_TOKEN, 
            self.ACTION_END_TOKEN,
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
        self.state_begin_token_id = tokenizer.convert_tokens_to_ids(self.STATE_BEGIN_TOKEN)
        self.state_end_token_id = tokenizer.convert_tokens_to_ids(self.STATE_END_TOKEN)
        self.action_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_TOKEN)
        self.action_begin_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_BEGIN_TOKEN)
        self.action_end_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_END_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        self.motion_tokenizer = motion_tokenizer
        
        # Unified tokenization (both states and actions)
        total_vocab_size = self.motion_tokenizer.vocab_size

        vocab_size = tokenizer.vocab_size
        added_tokens = list(tokenizer.added_tokens_encoder.values())
        token_id_replace = []
        replace_size = 0
        for i in range(vocab_size - 1 - last_skip_tokens, -1, -1):
            if i in added_tokens:
                continue
            token_id_replace.append(i)
            replace_size += 1
            if replace_size >= total_vocab_size:
                break
        assert replace_size == total_vocab_size, "The replace size is not equal to the total vocab size"
        token_id_replace = token_id_replace[::-1]
        
        
        forward, reverse, replace_idx = motion_tokenizer.setup_tokenizer_gemma_mappings(
            token_id_replace, 0
        )
        # For VQ tokenizer, forward is a nested dict: {part_name: {vq_id: gemma_id}}
        self.motion_token_id2gemma_token_id = forward
        self.gemma_token_id2motion_token_id = reverse
        assert replace_idx == total_vocab_size
        self.total_motion_token_list = token_id_replace[:total_vocab_size]
    
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

        # We assume the states and actions are in [-1, 1]
        # Jointly encode states and actions using unified tokenization
        discrete_states, discrete_actions, _ = self.motion_tokenizer(states, actions)
        discrete_states = discrete_states[0]  # Extract first batch item
        discrete_actions = discrete_actions[0] if objective != "train_flow" else None

        intrinsic_str = f"fx:{intrinsic[0]:.2f} fy:{intrinsic[1]:.2f} cx:{intrinsic[2]:.2f} cy:{intrinsic[3]:.2f}"
        text = text.replace('.', '')
        text = text.lower()
        text = (
            f"Task: {text}, Camera intrinsic: {intrinsic_str}, "
            f"States: {self.STATE_BEGIN_TOKEN}{self.STATE_TOKEN * len(discrete_states)}{self.STATE_END_TOKEN};"
            f"Action: "
        )
        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        depth_image_seq_len = 0
        if depth_images is not None:
            depth_image_seq_len = self.depth_image_seq_length * depth_images.shape[0]
        input_string = add_image_action_tokens_to_prompt(
            prefix_prompt=text,
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            image_seq_len=self.image_seq_length * images.shape[0],
            image_token=self.IMAGE_TOKEN,
            action_begin_token=self.ACTION_BEGIN_TOKEN,
            action_end_token=self.ACTION_END_TOKEN,
            action_token=self.ACTION_TOKEN,
            action_seq_len=len(discrete_actions) if objective != "train_flow" else 0,
            need_action=objective != "train_flow",
        )

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

        mapped_states_1d = self.motion_tokenizer.map_motion_tokens2gemma(
            np.array(discrete_states), 
            mapping=self.motion_token_id2gemma_token_id,
        )
        input_ids = set_token_id(input_ids, self.state_token_id, mapped_states_1d)
        
        if objective != "train_flow":
            # Map VQ tokens to Gemma space (discrete_actions already encoded above)
            mapped_actions_1d = self.motion_tokenizer.map_motion_tokens2gemma(
                np.array(discrete_actions),
                mapping=self.motion_token_id2gemma_token_id,
            )
            input_ids = set_token_id(input_ids, self.action_token_id, mapped_actions_1d)

        if objective != "train_flow":
            labels = input_ids.copy()
            if not np.any(labels == self.action_begin_token_id):
                warnings.warn("The action begin token is not found in the input_ids")
                answer_start_idx = len(labels)
            else : 
                answer_start_idx = np.argmax(labels == self.action_begin_token_id) + 1
            labels[:answer_start_idx] = self.ignore_index
            labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
            inputs['labels'] = labels
            inputs['answer_start_idx'] = np.array(answer_start_idx)
        else: 
            attention_mask = inputs['attention_mask']
            inputs['answer_start_idx'] = np.array(np.sum(attention_mask))

        output = {"pixel_values": pixel_values, **inputs}
        if depth_values is not None:
            output["depth_values"] = depth_values
        return output

    def extract_partial_tokens(self, input_ids, begin_id, end_id): 
        if begin_id not in input_ids:
            warnings.warn(f"The begin token {begin_id} is not found in the input_ids")
            return None
        if end_id not in input_ids:
            warnings.warn(f"The end token {end_id} is not found in the input_ids")
            return None
        start_idx = np.argmax(input_ids == begin_id) + 1
        end_idx = np.argmax(input_ids == end_id)
        return input_ids[start_idx:end_idx].copy()

    def map_gemma_tokens2motion_tokens(self, gemma_tokens):
        motion_tokens = []
        for token in gemma_tokens:
            if token not in self.gemma_token_id2motion_token_id:
                warnings.warn(f"The token {token} is not found in the gemma_token_id2motion_token_id")
                return {}
            motion_tokens.append(self.gemma_token_id2motion_token_id[token])
        return motion_tokens

    def decode(self, output_ids, T_state_original, T_action_original):
        '''
        Decode the output_ids of VLM back to states and actions.
        Args:
            output_ids: np.ndarray [L]
            T_state_original: int
            T_action_original: int

        Returns:
            dict:
                - states: np.ndarray [T_state_original, D_total]
                - actions: np.ndarray [T_action_original, D_total]
        '''
        state_tokens = self.extract_partial_tokens(
            output_ids, self.state_begin_token_id, self.state_end_token_id
        )
        action_tokens = self.extract_partial_tokens(
            output_ids, self.action_begin_token_id, self.action_end_token_id
        )
        
        state_tokens = self.map_gemma_tokens2motion_tokens(state_tokens)
        action_tokens = self.map_gemma_tokens2motion_tokens(action_tokens)
        
        if state_tokens is None or action_tokens is None:
            return {}
        states, actions = self.motion_tokenizer.decode(
            state_tokens, action_tokens, T_state_original, T_action_original
        )
        return {'states': states[0], 'actions': actions[0]}


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
    intrinsic = intrinsic * np.array([scale_x, scale_y, scale_x, scale_y])
    return intrinsic

def set_token_id(input_ids, token_id, discrete_tokens):
    condition = (input_ids == token_id)
    available_tokens = np.sum(condition)
    if available_tokens != len(discrete_tokens):
        warnings.warn(f"The number of tokens to set is not equal to the number of discrete tokens. {np.sum(condition)} != {len(discrete_tokens)}")
    
    input_ids[condition] = discrete_tokens[:available_tokens]
    return input_ids


def test_encode_decode_consistency():
    """测试 encode/decode 的往返一致性"""
    from omegaconf import OmegaConf
    import hydra
    
    print("=" * 80)
    print("测试 encode/decode 往返一致性")
    print("=" * 80)
    
    # 允许在配置中使用 ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # 加载配置
    cfg = OmegaConf.load("src/config/experiment/pretrain_legendvla_deepspeed.yaml")
    
    # 实例化 processor
    print("\n1. 实例化 VLA Processor...")
    vla_processor = hydra.utils.instantiate(cfg.vla_processor)
    print(f"   ✓ Processor 初始化成功")
    print(f"   - vocab_size: {vla_processor.tokenizer.vocab_size}")
    print(f"   - motion_tokenizer vocab_size: {vla_processor.motion_tokenizer.vocab_size}")
    
    # 准备测试数据
    print("\n2. 准备测试数据...")
    T_state = cfg.n_obs_state_steps  # 16
    T_action = cfg.n_action_steps  # 32
    state_dim = 48  # wrist_dim(18) + hand_dim(30)
    action_dim = 48
    
    # 生成随机测试数据（在 [-1, 1] 范围内，模拟归一化后的数据）
    np.random.seed(42)
    states = np.random.uniform(-1, 1, size=(T_state, state_dim)).astype(np.float32)
    actions = np.random.uniform(-1, 1, size=(T_action, action_dim)).astype(np.float32)
    
    print(f"   ✓ 测试数据准备完成")
    print(f"   - states shape: {states.shape}")
    print(f"   - actions shape: {actions.shape}")
    print(f"   - states range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"   - actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # 测试 1: 直接使用 motion_tokenizer 进行 encode/decode
    print("\n3. 测试 motion_tokenizer encode/decode...")
    try:
        # Encode
        discrete_states, discrete_actions, metadata = vla_processor.motion_tokenizer(states, actions)
        discrete_states = discrete_states[0]  # 提取第一个 batch
        discrete_actions = discrete_actions[0]
        
        print(f"   ✓ Encode 成功")
        print(f"   - discrete_states length: {len(discrete_states)}")
        print(f"   - discrete_actions length: {len(discrete_actions)}")
        print(f"   - T_state_original: {metadata['T_state_original']}")
        print(f"   - T_action_original: {metadata['T_action_original']}")
        
        # Decode
        decoded_states, decoded_actions = vla_processor.motion_tokenizer.decode(
            discrete_states,
            discrete_actions,
            T_state_original=metadata['T_state_original'],
            T_action_original=metadata['T_action_original']
        )
        decoded_states = decoded_states[0]  # 提取第一个 batch
        decoded_actions = decoded_actions[0]
        
        print(f"   ✓ Decode 成功")
        print(f"   - decoded_states shape: {decoded_states.shape}")
        print(f"   - decoded_actions shape: {decoded_actions.shape}")
        
        # 计算误差
        state_error = np.abs(states - decoded_states)
        action_error = np.abs(actions - decoded_actions)
        
        print(f"\n   📊 误差统计:")
        print(f"   - State MAE: {state_error.mean():.6f}")
        print(f"   - State Max Error: {state_error.max():.6f}")
        print(f"   - Action MAE: {action_error.mean():.6f}")
        print(f"   - Action Max Error: {action_error.max():.6f}")
        
        # 检查形状是否一致
        assert states.shape == decoded_states.shape, \
            f"State shape mismatch: {states.shape} vs {decoded_states.shape}"
        assert actions.shape == decoded_actions.shape, \
            f"Action shape mismatch: {actions.shape} vs {decoded_actions.shape}"
        
        print(f"   ✓ 形状检查通过")
        
    except Exception as e:
        print(f"   ✗ motion_tokenizer encode/decode 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 2: 完整的 processor pipeline (encode -> map to gemma -> map back -> decode)
    print("\n4. 测试完整的 processor pipeline...")
    try:
        # 使用 processor 的 __call__ 方法进行编码
        # 创建虚拟的 images 和 intrinsic
        images = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
        intrinsic = np.array([320.0, 240.0, 160.0, 120.0], dtype=np.float32)
        text = "test task"
        
        # 先获取 metadata（通过直接调用 motion_tokenizer）
        _, _, metadata_full = vla_processor.motion_tokenizer(states, actions)
        
        # Encode
        result = vla_processor(
            text=text,
            images=images,
            states=states,
            actions=actions,
            intrinsic=intrinsic,
            objective=None,  # 使用默认值
            truncation=False,
        )
        
        print(f"   ✓ Processor encode 成功")
        print(f"   - input_ids length: {len(result['input_ids'])}")
        
        # 提取 state 和 action tokens
        state_tokens_gemma = vla_processor.extract_partial_tokens(
            result['input_ids'],
            vla_processor.state_begin_token_id,
            vla_processor.state_end_token_id
        )
        action_tokens_gemma = vla_processor.extract_partial_tokens(
            result['input_ids'],
            vla_processor.action_begin_token_id,
            vla_processor.action_end_token_id
        )
        
        print(f"   ✓ Token 提取成功")
        print(f"   - state_tokens_gemma length: {len(state_tokens_gemma) if state_tokens_gemma is not None else 0}")
        print(f"   - action_tokens_gemma length: {len(action_tokens_gemma) if action_tokens_gemma is not None else 0}")
        
        # Decode
        decoded_result = vla_processor.decode(
            result['input_ids'],
            T_state_original=metadata_full['T_state_original'],
            T_action_original=metadata_full['T_action_original']
        )
        
        if decoded_result:
            decoded_states_full = decoded_result['states']
            decoded_actions_full = decoded_result['actions']
            
            print(f"   ✓ Processor decode 成功")
            print(f"   - decoded_states_full shape: {decoded_states_full.shape}")
            print(f"   - decoded_actions_full shape: {decoded_actions_full.shape}")
            
            # 计算误差
            state_error_full = np.abs(states - decoded_states_full)
            action_error_full = np.abs(actions - decoded_actions_full)
            
            print(f"\n   📊 完整 pipeline 误差统计:")
            print(f"   - State MAE: {state_error_full.mean():.6f}")
            print(f"   - State Max Error: {state_error_full.max():.6f}")
            print(f"   - Action MAE: {action_error_full.mean():.6f}")
            print(f"   - Action Max Error: {action_error_full.max():.6f}")
            
            # 检查形状是否一致
            assert states.shape == decoded_states_full.shape, \
                f"State shape mismatch: {states.shape} vs {decoded_states_full.shape}"
            assert actions.shape == decoded_actions_full.shape, \
                f"Action shape mismatch: {actions.shape} vs {decoded_actions_full.shape}"
            
            print(f"   ✓ 形状检查通过")
        else:
            print(f"   ⚠ Decode 返回空结果")
            
    except Exception as e:
        print(f"   ✗ Processor pipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)
    return True


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra
    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # 运行测试
    success = test_encode_decode_consistency()
    
    # if not success:
    #     print("\n运行原始测试...")
    #     cfg = OmegaConf.load("src/config/experiment/pretrain_legendvla_deepspeed.yaml")
    #     config = OmegaConf.to_yaml(cfg.vla_processor, resolve=True)
    #     print(config)
    #     vla_processor = hydra.utils.instantiate(cfg.vla_processor)
    #     print(f"vocab_size: {vla_processor.tokenizer.vocab_size}")
    #     print(f"len(tokenizer): {len(vla_processor.tokenizer)}")
    #     print(f"image_token_id: {vla_processor.image_token_id}")
    #     print(f"state_token_id: {vla_processor.state_token_id}")
    #     print(f"state_begin_token_id: {vla_processor.state_begin_token_id}")
    #     print(f"state_end_token_id: {vla_processor.state_end_token_id}")
    #     print(f"action_token_id: {vla_processor.action_token_id}")
    #     print(f"action_begin_token_id: {vla_processor.action_begin_token_id}")
    #     print(f"action_end_token_id: {vla_processor.action_end_token_id}")
    #     print(f"total_motion_token_list: {vla_processor.total_motion_token_list}")
    #     if hasattr(vla_processor, 'gemma_token_id2motion_token_id'):
    #         if isinstance(vla_processor.gemma_token_id2motion_token_id, dict):
    #             if 'wrist' in vla_processor.gemma_token_id2motion_token_id:
    #                 print(f"wrist gemma_token_id2motion_token_id: {np.min(list(vla_processor.gemma_token_id2motion_token_id['wrist'].keys()))} {np.max(list(vla_processor.gemma_token_id2motion_token_id['wrist'].keys()))}")
    #             if 'hand' in vla_processor.gemma_token_id2motion_token_id:
    #                 print(f"hand gemma_token_id2motion_token_id: {np.min(list(vla_processor.gemma_token_id2motion_token_id['hand'].keys()))} {np.max(list(vla_processor.gemma_token_id2motion_token_id['hand'].keys()))}")
    #         else:
    #             keys = list(vla_processor.gemma_token_id2motion_token_id.keys())
    #             if len(keys) > 0:
    #                 print(f"gemma_token_id2motion_token_id range: {np.min(keys)} {np.max(keys)}")

    # vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
    # print(f"image_token_id: {vlm_processor.image_token_id}")
    # print(f"tokenizer.all_special_ids: {vlm_processor.tokenizer.all_special_ids}")
    # print(f"tokenizer.all_special_tokens: {vlm_processor.tokenizer.all_special_tokens}")
