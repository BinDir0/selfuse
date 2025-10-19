from typing import Dict, List, Tuple
import warnings
import re

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
            # print(f"text length: {len(text.split(' '))}, target length: {len(target.split(' '))}")
            # print(f"text: {text}, target: {target}")
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
        hand_tokenizer,
        num_image_tokens: int,
        max_seq_len: int,
        ignore_index: int = -100,
        image_size: int = 224,
        tokenizer_padding: str = "longest", # longest or max_length
        hand_tokenizer_type: str = "fast", # "fast" or "vq"
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.tokenizer_padding = tokenizer_padding
        self.hand_tokenizer_type = hand_tokenizer_type
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
        if self.hand_tokenizer_type == "fast":
            self.fast_tokenizer = hand_tokenizer
            self.setup_fast_tokenizer_mappings()
        elif self.hand_tokenizer_type == "vq":
            self.vq_tokenizer = hand_tokenizer
            self.setup_vq_tokenizer_mappings()
        else:
            raise ValueError(f"Unknown hand_tokenizer_type: {hand_tokenizer_type}")

    def setup_fast_tokenizer_mappings(self):
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

    def setup_vq_tokenizer_mappings(self, vq_tokenizer):
        """
        Build mappings between VQ action tokens (nested dict: group -> part -> vq_id)
        and base VLM (Gemma) tokenizer token ids (excluding special ids).

        Produces:
        - self.vq_token_id2gemma_token_id[group][part][vq_id] = gemma_id
        - self.gemma_token_id2vq_token_id[gemma_id] = (group, part, vq_id)
        """

        # ---------- count total VQ tokens ----------
        total_vq_vocab_size = 0
        for group_name in vq_tokenizer.keys():
            for part_name, processor in vq_tokenizer[group_name].items():
                total_vq_vocab_size += processor.vocab_size

        # ---------- collect a pool of usable Gemma ids (skip specials) ----------
        base_vocab_size = self.tokenizer.vocab_size
        special_ids = set(self.tokenizer.all_special_ids or [])
        usable = []
        for tid in range(base_vocab_size - 1, -1, -1):  # from high to low
            if tid in special_ids:
                continue
            usable.append(tid)
            if len(usable) >= total_vq_vocab_size:
                break
        assert len(usable) == total_vq_vocab_size, (
            f"Not enough free Gemma ids: need {total_vq_vocab_size}, got {len(usable)}"
        )
        usable.reverse()  # low→high for determinism

        # ---------- make mappings (nested forward; flat reverse) ----------
        self.vq_token_id2gemma_token_id = {
            group_name: {part_name: {} for part_name in vq_tokenizer[group_name].keys()}
            for group_name in vq_tokenizer.keys()
        }
        self.gemma_token_id2vq_token_id = {}

        replace_idx = 0

        # To keep deterministic order across runs, sort the keys
        for group_name in sorted(vq_tokenizer.keys()):
            sub = vq_tokenizer[group_name]
            for part_name in sorted(sub.keys()):
                processor = sub[part_name]
                vocab_sz = processor.vocab_size
                # Map vq_id = 0..vocab_sz-1 → consecutive Gemma ids from `usable`
                for vq_id in range(vocab_sz):
                    gemma_id = usable[replace_idx]
                    replace_idx += 1
                    self.vq_token_id2gemma_token_id[group_name][part_name][vq_id] = gemma_id
                    self.gemma_token_id2vq_token_id[gemma_id] = (group_name, part_name, vq_id)

        # (Optional) sanity checks
        assert replace_idx == total_vq_vocab_size
    
    def __call__(
        self,
        text: str,
        images: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        objective: str = None,
        truncation: bool = True,
    ) -> dict:
        '''
        Args: 
            text: str
            images: np.ndarray [T_image, C, H, W] or [T_image, H, W, C]
            state: np.ndarray [T_state, state_dim]
            action: np.ndarray [Horizon, action_dim]
            objective: str, 'ar' or 'flow' or None, None means both
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

        # We assume the states and actions are in [-1, 1]
        if self.hand_tokenizer_type == "fast":
            discrete_states = self.fast_tokenizer['states'](states)[0]
            if objective != "train_flow":
                discrete_actions = self.fast_tokenizer['actions'](actions)[0]
        elif self.hand_tokenizer_type == "vq":
            states_wrist_raw = self.vq_tokenizer['states']['wrist'](states[:, :18])  # [G,B,T,L]
            states_hand_raw  = self.vq_tokenizer['states']['hand'](states[:, 18:])   # [G,B,T,L]
            discrete_states_wrist = self.flatten_groups(states_wrist_raw)  # 1D
            discrete_states_hand  = self.flatten_groups(states_hand_raw)   # 1D
            discrete_states = np.concatenate([discrete_states_wrist, discrete_states_hand], axis=0)  # 1D
            if objective != "train_flow":
                actions_wrist_raw = self.vq_tokenizer['actions']['wrist'](actions[:, :18])  # [G,B,T,L]
                actions_hand_raw  = self.vq_tokenizer['actions']['hand'](actions[:, 18:])   # [G,B,T,L]
                discrete_actions_wrist = self.flatten_groups(actions_wrist_raw)  # 1D
                discrete_actions_hand  = self.flatten_groups(actions_hand_raw)   # 1D
                discrete_actions = np.concatenate([discrete_actions_wrist, discrete_actions_hand], axis=0)  # 1D
        else:
            raise ValueError(f"Unknown hand_tokenizer_type: {self.hand_tokenizer_type}")

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

        if self.hand_tokenizer_type == "fast":
            discrete_states = np.array([self.fast_token_id2gemma_token_id['states'][id] for id in discrete_states])
            input_ids = set_token_id(input_ids, self.state_token_id, discrete_states)
            if objective != "train_flow":
                discrete_actions = np.array([self.fast_token_id2gemma_token_id['actions'][id] for id in discrete_actions])
                input_ids = set_token_id(input_ids, self.action_token_id, discrete_actions)
        elif self.hand_tokenizer_type == "vq":
            states_wrist_raw = self.vq_tokenizer['states']['wrist'](states[:, :18])  # [G,B,T,L]
            states_hand_raw  = self.vq_tokenizer['states']['hand'](states[:, 18:])   # [G,B,T,L]
            vq_ids_1d, (G, T, Lw, Lh) = self.flatten_time_interleave_wrist_hand(
                states_wrist_raw, states_hand_raw
            )
            self.states_vq_meta = {"G": G, "T": T, "Lw": Lw, "Lh": Lh}
            wrist_len = G*Lw*T
            mapped_states_1d = np.concatenate([
                self.map_vq_to_gemma_1d(vq_ids_1d[:wrist_len], group="states", part="wrist", mapping=self.vq_token_id2gemma_token_id),
                self.map_vq_to_gemma_1d(vq_ids_1d[wrist_len:], group="states", part="hand", mapping=self.vq_token_id2gemma_token_id)
            ], axis=0)
            input_ids = set_token_id(input_ids, self.state_token_id, mapped_states_1d)
            if objective != "train_flow":
                actions_wrist_raw = self.vq_tokenizer['actions']['wrist'](actions[:, :18])  # [G,B,T,L]
                actions_hand_raw  = self.vq_tokenizer['actions']['hand'](actions[:, 18:])   # [G,B,T,L]
                vq_ids_1d, (G, T, Lw, Lh) = self.flatten_time_interleave_wrist_hand(
                    actions_wrist_raw, actions_hand_raw
                )
                self.actions_vq_meta = {"G": G, "T": T, "Lw": Lw, "Lh": Lh}
                wrist_len = G*Lw*T
                mapped_actions_1d = np.concatenate([
                    self.map_vq_to_gemma_1d(vq_ids_1d[:wrist_len], group="actions", part="wrist", mapping=self.vq_token_id2gemma_token_id),
                    self.map_vq_to_gemma_1d(vq_ids_1d[wrist_len:], group="actions", part="hand", mapping=self.vq_token_id2gemma_token_id)
                ], axis=0)
                input_ids = set_token_id(input_ids, self.action_token_id, mapped_actions_1d)
        else:
            raise ValueError(f"Unknown hand_tokenizer_type: {self.hand_tokenizer_type}")
        inputs['input_ids'] = input_ids

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
        return output

    def flatten_groups(self, tokens_by_group_batchTL):  
        # Input shape: [G, B, T, L]
        x = np.asarray(tokens_by_group_batchTL)
        if x.ndim == 4:
            # Get [G, T, L] by batch=0
            x = x[:, 0] 
        assert x.ndim == 3  # [G, T, L]
        G, T, L = x.shape
        # Flatten the tokens by the fixed order: first time, then group, then L dimension
        out = []
        for t in range(T):
            for g in range(G):
                out.extend(x[g, t].tolist())   # L tokens
        return np.asarray(out, dtype=np.int64)  # 1D
    def flatten_time_interleave_wrist_hand(self, wrist_raw, hand_raw):
        """
        Flatten the wrist and hand tokens by the fixed order: first time, then group, then L dimension
        """
        W = wrist_raw.squeeze(1)# [G, T, L]
        H = hand_raw.squeeze(1)   # [G, T, L]
        G, T, Lw = W.shape[0], W.shape[1], W.shape[2]
        Lh = H.shape[2]
        seq = []
        for t in range(T):
            # First wrist@t (by g then by l)
            for g in range(G):
                seq.extend(W[g, t].tolist())   # L vq_ids
            # Then hand@t
            for g in range(G):
                seq.extend(H[g, t].tolist())
        return np.asarray(seq, dtype=np.int64), (G, T, Lw, Lh)

    def map_vq_to_gemma_1d(self, vq_ids_1d, group, part, mapping):
        """Map the 1D vq_id sequence to the gemma_id 1D sequence."""
        # If different groups use different vocab, you can write part as f"{part}.g{g}", and map separately after splitting by the outer layer.
        return np.asarray([mapping[group][part][int(vq)] for vq in vq_ids_1d], dtype=np.int64)

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
        if self.hand_tokenizer_type == "fast":
            for idx, token in enumerate(action_tokens):
                if token not in self.gemma_token_id2fast_token_id['actions']:
                    warnings.warn(f"The token {token} is not found in the actions")
                    return {}
                action_tokens[idx] = self.gemma_token_id2fast_token_id['actions'][token]
            
            actions = self.fast_tokenizer['actions'].decode([action_tokens])
        elif self.hand_tokenizer_type == "vq":
            # -------- VQ branch: reconstruct time-interleaved wrist/hand --------
            meta = getattr(self, "actions_vq_meta", None)
            if not meta:
                warnings.warn("Missing actions_vq_meta; cache (G,T,Lw,Lh) during encode.")
                return {}
            G, T, Lw, Lh = int(meta["G"]), int(meta["T"]), int(meta["Lw"]), int(meta["Lh"])
            per_part = lambda L: G * L
            per_t = per_part(Lw) + per_part(Lh)

            # gemma -> (group_name, part_name, vq_id)
            triples = [self.gemma_token_id2vq_token_id.get(int(tok)) for tok in action_tokens]
            triples = [t for t in triples if t and t[0] == "actions" and t[1] in ("wrist","hand")]

            expected = T * per_t
            if len(triples) < expected:
                warnings.warn(f"Short sequence: {len(triples)} < {expected}")
                T = len(triples) // per_t

            wrist_ids = np.zeros((G, T, Lw), dtype=np.int64)
            hand_ids  = np.zeros((G, T, Lh), dtype=np.int64)

            idx = 0
            for t in range(T):
                block = triples[idx: idx + per_t]; idx += per_t
                wrist_block = block[: per_part(Lw)]
                hand_block  = block[per_part(Lw):]

                for g in range(G):
                    w_slice = wrist_block[g*Lw : (g+1)*Lw]
                    h_slice = hand_block [g*Lh : (g+1)*Lh]
                    wrist_ids[g, t] = [vv for (grp,part,vv) in w_slice if part=="wrist"][:Lw] or [0]*Lw
                    hand_ids [g, t] = [vv for (grp,part,vv) in h_slice if part=="hand" ][:Lh] or [0]*Lh

            acts_wrist = self.vq_tokenizer["actions"]["wrist"].decode(wrist_ids)
            acts_hand  = self.vq_tokenizer["actions"]["hand" ].decode(hand_ids)
            actions = np.concatenate([acts_wrist, acts_hand], axis=-1)
        return {'actions': actions}


def set_token_id(input_ids, token_id, discrete_tokens):
    condition = (input_ids == token_id)
    available_tokens = np.sum(condition)
    if available_tokens != len(discrete_tokens):
        warnings.warn(f"The number of tokens to set is not equal to the number of discrete tokens. {np.sum(condition)} != {len(discrete_tokens)}")
    
    input_ids[condition] = discrete_tokens[:available_tokens]
    return input_ids
