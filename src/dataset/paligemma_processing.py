from typing import Tuple
import warnings
import re

import torch
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


def process_depth_images(
    depth_images: np.ndarray,
    size: Tuple[int, int],
    rescale_factor: float = 1.0,
) -> np.ndarray:
    """Process depth images using numpy operations for CPU-based preprocessing.
    
    Args:
        depth_images: np.ndarray [B, H, W] or [B, 1, H, W] or [B, H, W, 1]
        size: Tuple[int, int] - target size (height, width)
        rescale_factor: float - scaling factor for pixel values (default 1.0)
        
    Returns:
        np.ndarray [B, 1, H, W] - processed depth images
    """
    # Convert to numpy if input is torch tensor
    if isinstance(depth_images, torch.Tensor):
        depth_images = depth_images.cpu().numpy()
    
    # Handle different input formats
    if depth_images.ndim == 2:
        # Single image: [H, W] -> [1, 1, H, W]
        depth_images = depth_images[np.newaxis, np.newaxis, :, :]
    elif depth_images.ndim == 3:
        # Batch of images: [B, H, W] or [H, W, 1]
        if depth_images.shape[-1] == 1:
            # [B, H, W, 1] -> [B, 1, H, W]
            depth_images = np.transpose(depth_images, (0, 3, 1, 2))
        else:
            # [B, H, W] -> [B, 1, H, W]
            depth_images = depth_images[:, np.newaxis, :, :]
    elif depth_images.ndim == 4:
        # [B, 1, H, W] or [B, H, W, 1]
        if depth_images.shape[-1] == 1:
            # [B, H, W, 1] -> [B, 1, H, W]
            depth_images = np.transpose(depth_images, (0, 3, 1, 2))
        # else: already [B, 1, H, W]
    
    # Rescale the pixel values if needed
    if rescale_factor != 1.0:
        depth_images = rescale(depth_images, scale=rescale_factor)
    
    # Resize the depth images to the desired size using PIL
    depth_images = resize(depth_images, size=size)
    
    # Depth images are not normalized (keep raw values)
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
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        tokenizer_padding: str = "longest", # longest or max_length
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
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
        self.action_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_TOKEN)
        self.action_begin_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_BEGIN_TOKEN)
        self.action_end_token_id = tokenizer.convert_tokens_to_ids(self.ACTION_END_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
        self.motion_tokenizer = motion_tokenizer
        
        total_vocab_size = sum([self.motion_tokenizer[k].vocab_size for k in self.motion_tokenizer.keys()])

        vocab_size = tokenizer.vocab_size
        added_tokens = list(tokenizer.added_tokens_encoder.values())
        token_id_replace = []
        replace_size = 0
        for i in range(vocab_size - 1, -1, -1):
            if i in added_tokens:
                continue
            token_id_replace.append(i)
            replace_size += 1
            if replace_size >= total_vocab_size:
                break
        assert replace_size == total_vocab_size, "The replace size is not equal to the total vocab size"
        token_id_replace = token_id_replace[::-1]
        
        # Step 3: Build mappings using each processor's setup method
        # Use unified naming - structure will differ based on tokenizer type
        self.motion_token_id2gemma_token_id = {}
        self.gemma_token_id2motion_token_id = {}
        
        replace_idx = 0
        for group_name in sorted(motion_tokenizer.keys()):
            processor = motion_tokenizer[group_name]
            forward, reverse, replace_idx = processor.setup_tokenizer_gemma_mappings(
                token_id_replace, replace_idx
            )
            # motion_token_id2gemma_token_id[group_name] = gemma_id (FAST), motion_token_id2gemma_token_id[group_name][part_name] = gemma_id (VQ)
            self.motion_token_id2gemma_token_id[group_name] = forward
            
            # Initialize inner dict for this group
            self.gemma_token_id2motion_token_id[group_name] = {}
            
            for gemma_id, token_id in reverse.items():
                self.gemma_token_id2motion_token_id[group_name][gemma_id] = token_id
        
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
            depth_images: np.ndarray [T_image, H, W] or [T_image, 1, H, W] (optional)

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
                depth_images,
                size=(self.image_size, self.image_size),
                rescale_factor=depth_scale_factor,
            )

        # We assume the states and actions are in [-1, 1]
        # Encode states and actions to get token counts for prompt construction
        discrete_states = self.motion_tokenizer['states'](states)[0]

        if objective != "train_flow":
            discrete_actions = self.motion_tokenizer['actions'](actions)[0]

        intrinsic_str = f"fx:{intrinsic[0]:.2f} fy:{intrinsic[1]:.2f} cx:{intrinsic[2]:.2f} cy:{intrinsic[3]:.2f}"
        text = text.replace('.', '')
        text = text.lower()
        text = f"{text} using camera {intrinsic_str}{self.STATE_BEGIN_TOKEN}{self.STATE_TOKEN * len(discrete_states)}{self.STATE_END_TOKEN}?"
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

        mapped_states_1d = self.motion_tokenizer['states'].map_motion_tokens2gemma(
            discrete_states, 
            mapping=self.motion_token_id2gemma_token_id['states'],
        )
        input_ids = set_token_id(input_ids, self.state_token_id, mapped_states_1d)
        
        if objective != "train_flow":
            # Map VQ tokens to Gemma space (discrete_actions already encoded above)
            mapped_actions_1d = self.motion_tokenizer['actions'].map_motion_tokens2gemma(
                discrete_actions,
                mapping=self.motion_token_id2gemma_token_id['actions'],
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

    def decode(self, output_ids):
        if self.action_begin_token_id not in output_ids:
            warnings.warn("The action begin token is not found in the output_ids")
            return {}
        if self.action_end_token_id not in output_ids:
            warnings.warn("The action end token is not found in the output_ids")
            return {}
        start_idx = np.argmax(output_ids == self.action_begin_token_id) + 1
        end_idx = np.argmax(output_ids == self.action_end_token_id)
        action_tokens = output_ids[start_idx:end_idx].copy()
        
        for idx, token in enumerate(action_tokens):
            if token not in self.gemma_token_id2motion_token_id['actions']:
                warnings.warn(f"The token {token} is not found in the actions")
                return {}
            action_tokens[idx] = self.gemma_token_id2motion_token_id['actions'][token]
        
        actions = self.motion_tokenizer['actions'].decode([action_tokens])
        return {'actions': actions}


def get_resized_intrinsic(intrinsic, original_width: int, original_height: int, img_size: int = 384): 
    '''
    Return intrinsic after resizing the images. 
    Args: 
        intrinsic: np.ndarray [..., 4]
        original_width: int
        original_height: int
        img_size: int = 384
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


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra
    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load("src/config/experiment/pretrain_legendvla_deepspeed.yaml")
    config = OmegaConf.to_yaml(cfg.vla_processor, resolve=True)
    print(config)
    vla_processor = hydra.utils.instantiate(cfg.vla_processor)
    print(f"vocab_size: {vla_processor.tokenizer.vocab_size}")
    print(f"len(tokenizer): {len(vla_processor.tokenizer)}")
    print(f"image_token_id: {vla_processor.image_token_id}")
    print(f"state_token_id: {vla_processor.state_token_id}")
    print(f"action_token_id: {vla_processor.action_token_id}")
    print(f"action_begin_token_id: {vla_processor.action_begin_token_id}")
    print(f"action_end_token_id: {vla_processor.action_end_token_id}")
    print(f"total_motion_token_list: {vla_processor.total_motion_token_list}")
    if 'states' in vla_processor.gemma_token_id2motion_token_id:
        print(f"states gemma_token_id2motion_token_id: {np.min(list(vla_processor.gemma_token_id2motion_token_id['states'].keys()))} {np.max(list(vla_processor.gemma_token_id2motion_token_id['states'].keys()))}")
    if 'actions' in vla_processor.gemma_token_id2motion_token_id:
        print(f"actions gemma_token_id2motion_token_id: {np.min(list(vla_processor.gemma_token_id2motion_token_id['actions'].keys()))} {np.max(list(vla_processor.gemma_token_id2motion_token_id['actions'].keys()))}")


    vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
    print(f"image_token_id: {vlm_processor.image_token_id}")
    print(f"tokenizer.all_special_ids: {vlm_processor.tokenizer.all_special_ids}")
    print(f"tokenizer.all_special_tokens: {vlm_processor.tokenizer.all_special_tokens}")
