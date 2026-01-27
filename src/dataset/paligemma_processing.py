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
    mode = 'train',
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    if mode == 'infer': 
        return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
    else:
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
    mode = 'train', # if True, the eos token will be added
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    if mode == 'infer': 
        prompt = f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
        if need_action:
            prompt += f"{action_begin_token}"
        return prompt
    elif need_action:
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
            mode=mode,
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
        self.eos_token_id = tokenizer.eos_token_id
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
            mode=mode,
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

        mapped_states_1d = self.motion_tokenizer.map_motion_tokens2gemma(
            np.array(discrete_states), 
            mapping=self.motion_token_id2gemma_token_id,
        )
        input_ids = set_token_id(input_ids, self.state_token_id, mapped_states_1d)
        
        if objective != "train_flow" and mode != 'infer':
            # Map VQ tokens to Gemma space (discrete_actions already encoded above)
            mapped_actions_1d = self.motion_tokenizer.map_motion_tokens2gemma(
                np.array(discrete_actions),
                mapping=self.motion_token_id2gemma_token_id,
            )
            input_ids = set_token_id(input_ids, self.action_token_id, mapped_actions_1d)

        if objective != "train_flow":
            labels = input_ids.copy()
            if not np.any(labels == self.action_begin_token_id):
                if mode != 'infer':
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


def test_encode_decode_consistency(batch_size: int = 4, output_file: str = None):
    """
    测试 encode/decode 的往返一致性，使用真实数据集中的 batch
    
    Args:
        batch_size: Batch 大小，默认 4
        output_file: 输出文件路径，如果为 None 则不保存到文件
    """
    from omegaconf import OmegaConf
    import hydra
    import torch
    from torch.utils.data import DataLoader
    import sys
    from datetime import datetime
    
    # 如果指定了输出文件，创建一个文件输出流
    original_stdout = sys.stdout
    file_handle = None
    if output_file is not None:
        file_handle = open(output_file, 'w', encoding='utf-8')
        sys.stdout = file_handle
        print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Batch size: {batch_size}")
        print(f"输出文件: {output_file}")
    
    try:
        print("=" * 80)
        print("测试 encode/decode 往返一致性（使用真实数据集）")
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
        
        # 实例化数据集
        print("\n2. 实例化数据集...")
        dataset = hydra.utils.instantiate(cfg.dataset)
        print(f"   ✓ 数据集初始化成功")
        print(f"   - VLA dataset size: {len(dataset.vla_dataset)}")
        if dataset.vlm_dataset is not None:
            print(f"   - VLM dataset size: {len(dataset.vlm_dataset)}")
        
        # 设置 preprocessor（必需，否则无法调用 _sample_to_data）
        print("\n3. 设置 preprocessor...")
        dataset.vla_dataset.set_preprocessor(vla_processor)
        vlm_processor = None
        if dataset.vlm_dataset is not None:
            vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
            dataset.vlm_dataset.set_preprocessor(vlm_processor)
        print(f"   ✓ Preprocessor 设置成功")
        if vlm_processor is not None:
            print(f"   - VLM Processor 已加载")
        
        # 设置 normalizer（如果需要）
        print("\n4. 设置 normalizer...")
        import pickle
        if hasattr(cfg.training, 'normalizer_path') and cfg.training.normalizer_path is not None:
            normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
            dataset.vla_dataset.set_normalizer(normalizer)
            print(f"   ✓ Normalizer 从文件加载成功: {cfg.training.normalizer_path}")
        else:
            # 如果没有提供 normalizer 路径，尝试计算
            print(f"   ⚠ 未提供 normalizer_path，尝试计算 normalizer...")
            try:
                normalizer = dataset.vla_dataset.get_normalizer()
                dataset.vla_dataset.set_normalizer(normalizer)
                print(f"   ✓ Normalizer 计算成功")
            except Exception as e:
                print(f"   ⚠ 无法获取 normalizer: {e}")
                print(f"   ⚠ 继续测试（normalizer 是可选的，但可能影响数据处理）")
        
        # 创建 DataLoader 并加载一个 batch
        print(f"\n5. 创建 DataLoader 并加载 batch (batch_size={batch_size})...")
        sampler = dataset.get_sampler(
            batch_size=batch_size,
            vla_ratio=7/8,  # 5:1 ratio of VLA and VLM samples
            shuffle=True,  # 不 shuffle，便于调试
            seed=42,
            drop_last=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=dataset.get_collator(),
            num_workers=0,  # 使用单进程，避免多进程问题
        )
        
        # 获取第一个 batch
        batch = next(iter(dataloader))
        print(f"   ✓ Batch 加载成功")
        print(f"   - Batch keys: {list(batch.keys())}")
        
        # 打印 batch 详细信息
        print("\n6. Batch 详细信息:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"   - {key}: length={len(value)}, type={type(value[0]) if len(value) > 0 else 'empty'}")
            else:
                print(f"   - {key}: type={type(value)}")
        
        # 提取 batch 数据
        print("\n7. 提取 batch 数据...")
        if 'input_ids' not in batch:
            raise ValueError(f"找不到 input_ids key，可用的 keys: {list(batch.keys())}")
        
        if 'actions' not in batch:
            raise ValueError(f"找不到 actions key，可用的 keys: {list(batch.keys())}")
        
        # 获取 batch 数据
        input_ids_batch = batch['input_ids']
        if isinstance(input_ids_batch, torch.Tensor):
            input_ids_batch = input_ids_batch.numpy()
        
        actions_batch = batch['actions']
        if isinstance(actions_batch, torch.Tensor):
            actions_batch = actions_batch.numpy()
        
        states_batch = None
        if 'states' in batch:
            states_batch = batch['states']
            if isinstance(states_batch, torch.Tensor):
                states_batch = states_batch.numpy()
        
        # 获取 pad_token_id
        pad_token_id = None
        if hasattr(vla_processor.tokenizer, 'pad_token_id') and vla_processor.tokenizer.pad_token_id is not None:
            pad_token_id = vla_processor.tokenizer.pad_token_id
        
        # 获取 T_state_original 和 T_action_original
        T_action_original = actions_batch.shape[1] if len(actions_batch.shape) > 2 else actions_batch.shape[0]
        T_state_original = cfg.n_obs_state_steps if hasattr(cfg, 'n_obs_state_steps') else T_action_original
        
        # 获取 normalizer
        normalizer = dataset.vla_dataset.normalizer
        
        print(f"   ✓ Batch 数据提取完成")
        print(f"   - input_ids_batch shape: {input_ids_batch.shape}")
        print(f"   - actions_batch shape: {actions_batch.shape}")
        if states_batch is not None:
            print(f"   - states_batch shape: {states_batch.shape}")
        print(f"   - T_state_original: {T_state_original}")
        print(f"   - T_action_original: {T_action_original}")
        print(f"   - Normalizer available: {normalizer is not None}")
        
        # 遍历每个样本进行测试
        print(f"\n8. 开始测试每个样本 (共 {batch_size} 个样本)...")
        print("=" * 80)
        
        all_decode_actions_success = []
        all_decode_text_success = []
        all_action_errors = []
        all_state_errors = []
        all_action_errors_unnormalized = []
        all_state_errors_unnormalized = []
        all_vlm_decode_success = []
        all_sample_types = []  # 'vla' or 'vlm'
        
        for sample_idx in range(batch_size):
            print(f"\n{'='*80}")
            print(f"样本 {sample_idx + 1}/{batch_size}")
            print(f"{'='*80}")
            
            # 提取当前样本的 input_ids
            input_ids = input_ids_batch[sample_idx]
            # 移除 padding
            if pad_token_id is not None:
                non_pad_mask = input_ids != pad_token_id
                if np.any(non_pad_mask):
                    input_ids = input_ids[non_pad_mask]
            
            # 提取当前样本的 actions
            actions = actions_batch[sample_idx] if len(actions_batch.shape) > 2 else actions_batch
            
            # 提取当前样本的 states
            states = None
            if states_batch is not None:
                states = states_batch[sample_idx] if len(states_batch.shape) > 2 else states_batch
            
            # 判断是否为 VLM 数据：VLM 数据的 actions 是全零的 dummy 数据
            is_vlm_sample = False
            if actions is not None:
                actions_sum = np.abs(actions).sum()
                if actions_sum < 1e-6:  # actions 全零或接近全零
                    is_vlm_sample = True
            
            all_sample_types.append('vlm' if is_vlm_sample else 'vla')
            
            print(f"\n样本 {sample_idx + 1} 基本信息:")
            print(f"   - 数据类型: {'VLM' if is_vlm_sample else 'VLA'}")
            print(f"   - input_ids length: {len(input_ids)}")
            if not is_vlm_sample:
                print(f"   - actions shape: {actions.shape}")
                print(f"   - actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                if states is not None:
                    print(f"   - states shape: {states.shape}")
                    print(f"   - states range: [{states.min():.3f}, {states.max():.3f}]")
            else:
                print(f"   - VLM 数据（无 actions/states）")
            
            # 根据数据类型选择不同的测试
            if is_vlm_sample:
                # VLM 数据：只测试文本 decode
                print(f"\n样本 {sample_idx + 1} - VLM 数据测试: Decode 文本...")
                decode_text_success = False
                try:
                    # 使用 VLM processor 解码 input_ids
                    if vlm_processor is None:
                        print(f"   ⚠ VLM Processor 不可用，使用 VLA Processor")
                        processor_to_use = vla_processor
                    else:
                        processor_to_use = vlm_processor
                    
                    decoded_text = processor_to_use.tokenizer.decode(
                        input_ids,
                        skip_special_tokens=False  # 保留特殊 token 以便查看完整结构
                    )
                    
                    print(f"   ✓ 文本 Decode 成功")
                    print(f"   - Decoded text length: {len(decoded_text)}")
                    print(f"   - Decoded text: {decoded_text}")
                    
                    # 也尝试跳过特殊 token 的版本
                    decoded_text_clean = processor_to_use.tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True
                    )
                    print(f"   - Decoded text (clean): {decoded_text_clean}")
                    
                    # 检查是否包含预期的特殊 token（VLM 可能没有 state/action tokens）
                    if hasattr(processor_to_use, 'image_token_id'):
                        image_token = processor_to_use.tokenizer.decode([processor_to_use.image_token_id])
                        if image_token in decoded_text:
                            print(f"   ✓ 找到 image token: '{image_token}'")
                    
                    decode_text_success = True
                    decode_actions_success = None  # VLM 数据没有 actions
                except Exception as e:
                    print(f"   ✗ Decode 文本失败: {e}")
                    import traceback
                    traceback.print_exc()
                    decode_text_success = False
                    decode_actions_success = None
                
                all_vlm_decode_success.append(decode_text_success)
                all_decode_text_success.append(decode_text_success)
                all_decode_actions_success.append(None)  # VLM 数据没有 actions
                
                # 总结当前样本的测试结果
                if decode_text_success:
                    print(f"\n样本 {sample_idx + 1}: ✅ VLM 测试通过")
                else:
                    print(f"\n样本 {sample_idx + 1}: ⚠️ VLM 测试失败")
                
                # VLM 数据只测试文本，跳过 actions/states 测试
                # 但继续执行后续的文本测试（如果有的话）
                # 实际上 VLM 数据已经测试完文本了，所以这里可以跳过后续测试
                continue  # 跳过 VLA 的 actions/states 测试，但文本测试已经在上面完成了
            
            # VLA 数据：测试 actions/states decode 和文本 decode
            # 测试 1: Decode input_ids 中的 actions
            print(f"\n样本 {sample_idx + 1} - 测试 1: Decode actions...")
            decode_actions_success = False
            try:
                # 从 input_ids 中提取 state 和 action tokens
                state_tokens_gemma = vla_processor.extract_partial_tokens(
                    input_ids,
                    vla_processor.state_begin_token_id,
                    vla_processor.state_end_token_id
                )
                action_tokens_gemma = vla_processor.extract_partial_tokens(
                    input_ids,
                    vla_processor.action_begin_token_id,
                    vla_processor.action_end_token_id
                )
                
                if state_tokens_gemma is None or action_tokens_gemma is None:
                    print(f"   ⚠ 无法提取 state 或 action tokens")
                    print(f"   - state_tokens_gemma: {state_tokens_gemma is not None}")
                    print(f"   - action_tokens_gemma: {action_tokens_gemma is not None}")
                else:
                    print(f"   ✓ Token 提取成功")
                    print(f"   - state_tokens_gemma length: {len(state_tokens_gemma)}")
                    print(f"   - action_tokens_gemma length: {len(action_tokens_gemma)}")
                    
                    # Decode
                    decoded_result = vla_processor.decode(
                        input_ids,
                        T_state_original=T_state_original,
                        T_action_original=T_action_original
                    )
                    
                    if decoded_result:
                        decoded_actions = decoded_result['actions']
                        decoded_states = decoded_result.get('states', None)
                        
                        print(f"   ✓ Decode 成功")
                        print(f"   - decoded_actions shape: {decoded_actions.shape}")
                        print(f"   - original actions shape: {actions.shape}")
                        if decoded_states is not None:
                            print(f"   - decoded_states shape: {decoded_states.shape}")
                            if states is not None:
                                print(f"   - original states shape: {states.shape}")
                        
                        # 计算 normalized 误差
                        action_error = np.abs(actions - decoded_actions)
                        action_mae = action_error.mean()
                        action_max_error = action_error.max()
                        
                        print(f"   📊 Normalized 误差统计:")
                        print(f"   - Action MAE: {action_mae:.6f}")
                        print(f"   - Action Max Error: {action_max_error:.6f}")
                        
                        # 计算 state 误差（如果有）
                        state_mae = None
                        state_max_error = None
                        if decoded_states is not None and states is not None:
                            state_error = np.abs(states - decoded_states)
                            state_mae = state_error.mean()
                            state_max_error = state_error.max()
                            print(f"   - State MAE: {state_mae:.6f}")
                            print(f"   - State Max Error: {state_max_error:.6f}")
                        
                        # 计算 unnormalized 误差
                        if normalizer is not None:
                            try:
                                # Unnormalize 原始数据
                                original_dict = {'actions': torch.from_numpy(actions)}
                                if states is not None:
                                    original_dict['states'] = torch.from_numpy(states)
                                
                                original_unnormalized = normalizer.unnormalize(original_dict)
                                original_actions_unnorm = original_unnormalized['actions'].numpy()
                                original_states_unnorm = original_unnormalized.get('states', None)
                                if original_states_unnorm is not None:
                                    original_states_unnorm = original_states_unnorm.numpy()
                                
                                # Unnormalize decoded 数据
                                decoded_dict = {'actions': torch.from_numpy(decoded_actions)}
                                if decoded_states is not None:
                                    decoded_dict['states'] = torch.from_numpy(decoded_states)
                                
                                decoded_unnormalized = normalizer.unnormalize(decoded_dict)
                                decoded_actions_unnorm = decoded_unnormalized['actions'].numpy()
                                decoded_states_unnorm = decoded_unnormalized.get('states', None)
                                if decoded_states_unnorm is not None:
                                    decoded_states_unnorm = decoded_states_unnorm.numpy()
                                
                                # 计算 unnormalized 误差
                                action_error_unnorm = np.abs(original_actions_unnorm - decoded_actions_unnorm)
                                action_mae_unnorm = action_error_unnorm.mean()
                                action_max_error_unnorm = action_error_unnorm.max()
                                
                                print(f"   📊 Unnormalized 误差统计:")
                                print(f"   - Action MAE: {action_mae_unnorm:.6f}")
                                print(f"   - Action Max Error: {action_max_error_unnorm:.6f}")
                                
                                if original_states_unnorm is not None and decoded_states_unnorm is not None:
                                    state_error_unnorm = np.abs(original_states_unnorm - decoded_states_unnorm)
                                    state_mae_unnorm = state_error_unnorm.mean()
                                    state_max_error_unnorm = state_error_unnorm.max()
                                    print(f"   - State MAE: {state_mae_unnorm:.6f}")
                                    print(f"   - State Max Error: {state_max_error_unnorm:.6f}")
                                    
                                    all_state_errors_unnormalized.append(state_mae_unnorm)
                                
                                all_action_errors_unnormalized.append(action_mae_unnorm)
                                
                            except Exception as e:
                                print(f"   ⚠ Unnormalize 计算失败: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"   ⚠ Normalizer 不可用，跳过 unnormalized 误差计算")
                        
                        # 检查形状是否一致
                        if actions.shape == decoded_actions.shape:
                            print(f"   ✓ Actions 形状检查通过")
                            decode_actions_success = True
                            all_action_errors.append(action_mae)
                            if state_mae is not None:
                                all_state_errors.append(state_mae)
                        else:
                            print(f"   ✗ Actions 形状不匹配: {actions.shape} vs {decoded_actions.shape}")
                    else:
                        print(f"   ⚠ Decode 返回空结果")
            except Exception as e:
                print(f"   ✗ Decode actions 失败: {e}")
                import traceback
                traceback.print_exc()
            
            all_decode_actions_success.append(decode_actions_success)
            
            # 测试 2: Decode input_ids 中的文本
            print(f"\n样本 {sample_idx + 1} - 测试 2: Decode 文本...")
            decode_text_success = False
            try:
                # 使用 tokenizer 解码 input_ids
                decoded_text = vla_processor.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False  # 保留特殊 token 以便查看完整结构
                )
                
                print(f"   ✓ 文本 Decode 成功")
                print(f"   - Decoded text length: {len(decoded_text)}")
                print(f"   - Decoded text: {decoded_text}")
                
                # 也尝试跳过特殊 token 的版本
                decoded_text_clean = vla_processor.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=True
                )
                print(f"   - Decoded text (clean): {decoded_text_clean}")
                
                # 检查是否包含预期的特殊 token
                has_state_token = False
                has_action_token = False
                if hasattr(vla_processor, 'state_begin_token_id'):
                    state_begin_token = vla_processor.tokenizer.decode([vla_processor.state_begin_token_id])
                    if state_begin_token in decoded_text:
                        has_state_token = True
                        print(f"   ✓ 找到 state begin token: '{state_begin_token}'")
                
                if hasattr(vla_processor, 'action_begin_token_id'):
                    action_begin_token = vla_processor.tokenizer.decode([vla_processor.action_begin_token_id])
                    if action_begin_token in decoded_text:
                        has_action_token = True
                        print(f"   ✓ 找到 action begin token: '{action_begin_token}'")
                
                decode_text_success = True
            except Exception as e:
                print(f"   ✗ Decode 文本失败: {e}")
                import traceback
                traceback.print_exc()
            
            all_decode_text_success.append(decode_text_success)
            
            # 总结当前样本的测试结果
            if decode_actions_success and decode_text_success:
                print(f"\n样本 {sample_idx + 1}: ✅ 所有测试通过")
            else:
                print(f"\n样本 {sample_idx + 1}: ⚠️ 部分测试失败")
                print(f"   - Decode actions: {'✓' if decode_actions_success else '✗'}")
                print(f"   - Decode text: {'✓' if decode_text_success else '✗'}")
        
        # 总结所有样本的测试结果
        print(f"\n{'='*80}")
        print("总体测试结果统计")
        print(f"{'='*80}")
        
        # 统计 VLA 和 VLM 样本数量
        num_vla_samples = sum(1 for t in all_sample_types if t == 'vla')
        num_vlm_samples = sum(1 for t in all_sample_types if t == 'vlm')
        print(f"\n样本类型统计:")
        print(f"   - VLA 样本: {num_vla_samples}/{batch_size}")
        print(f"   - VLM 样本: {num_vlm_samples}/{batch_size}")
        
        # 统计 VLA 测试结果
        num_decode_actions_success = sum(1 for s in all_decode_actions_success if s is True)
        num_decode_text_success_vla = sum(1 for i, s in enumerate(all_decode_text_success) if all_sample_types[i] == 'vla' and s)
        
        # 统计 VLM 测试结果
        num_vlm_decode_success = sum(all_vlm_decode_success) if all_vlm_decode_success else 0
        
        if num_vla_samples > 0:
            print(f"\nVLA 数据测试结果:")
            print(f"\n  Decode Actions 测试:")
            print(f"     - 成功: {num_decode_actions_success}/{num_vla_samples}")
            print(f"     - 失败: {num_vla_samples - num_decode_actions_success}/{num_vla_samples}")
        if all_action_errors:
            print(f"   Normalized 误差:")
            print(f"     - 平均 MAE: {np.mean(all_action_errors):.6f}")
            print(f"     - 最大 MAE: {np.max(all_action_errors):.6f}")
            print(f"     - 最小 MAE: {np.min(all_action_errors):.6f}")
        if all_action_errors_unnormalized:
            print(f"   Unnormalized 误差:")
            print(f"     - 平均 MAE: {np.mean(all_action_errors_unnormalized):.6f}")
            print(f"     - 最大 MAE: {np.max(all_action_errors_unnormalized):.6f}")
            print(f"     - 最小 MAE: {np.min(all_action_errors_unnormalized):.6f}")
        
        if all_state_errors:
            print(f"\nDecode States 测试:")
            print(f"   - 成功: {len(all_state_errors)}/{batch_size}")
            print(f"   Normalized 误差:")
            print(f"     - 平均 MAE: {np.mean(all_state_errors):.6f}")
            print(f"     - 最大 MAE: {np.max(all_state_errors):.6f}")
            print(f"     - 最小 MAE: {np.min(all_state_errors):.6f}")
        if all_state_errors_unnormalized:
            print(f"   Unnormalized 误差:")
            print(f"     - 平均 MAE: {np.mean(all_state_errors_unnormalized):.6f}")
            print(f"     - 最大 MAE: {np.max(all_state_errors_unnormalized):.6f}")
            print(f"     - 最小 MAE: {np.min(all_state_errors_unnormalized):.6f}")
        
            print(f"\n  Decode Text 测试:")
            print(f"     - 成功: {num_decode_text_success_vla}/{num_vla_samples}")
            print(f"     - 失败: {num_vla_samples - num_decode_text_success_vla}/{num_vla_samples}")
        
        if num_vlm_samples > 0:
            print(f"\nVLM 数据测试结果:")
            print(f"  Decode Text 测试:")
            print(f"     - 成功: {num_vlm_decode_success}/{num_vlm_samples}")
            print(f"     - 失败: {num_vlm_samples - num_vlm_decode_success}/{num_vlm_samples}")
        
        # 最终结果
        all_success = True
        if num_vla_samples > 0:
            all_success = all_success and (num_decode_actions_success == num_vla_samples and num_decode_text_success_vla == num_vla_samples)
        if num_vlm_samples > 0:
            all_success = all_success and (num_vlm_decode_success == num_vlm_samples)
        
        if all_success:
            print("\n" + "=" * 80)
            print("✅ 所有样本的所有测试都通过！")
            print("=" * 80)
            result = True
        else:
            print("\n" + "=" * 80)
            print("⚠️ 部分样本测试失败")
            print("=" * 80)
            result = False
            
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        result = False
    finally:
        # 恢复 stdout
        if file_handle is not None:
            sys.stdout = original_stdout
            file_handle.close()
            print(f"\n测试结果已保存到: {output_file}")
            print(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra
    import argparse
    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试 encode/decode 一致性')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch 大小，默认 4')
    parser.add_argument('--output_file', type=str, default=None, help='输出文件路径，如果指定则保存结果到文件')
    args = parser.parse_args()
    
    # 运行测试
    success = test_encode_decode_consistency(batch_size=args.batch_size, output_file=args.output_file)
    
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
