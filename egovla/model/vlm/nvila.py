import torch
from einops import rearrange
from llava.model import LlavaLlamaModel
from llava.constants import DEFAULT_IMAGE_TOKEN

from egovla.model.common.module_attr_mixin import ModuleAttrMixin


class NVILA(ModuleAttrMixin):
    """
    NVILA model used in EgoVLA.
    
    Architecture:
        - Vision encoder: Processes multi-frame RGB observations
        - Language encoder: Processes text instructions  
        - Action query tokens: Learnable tokens for action prediction
        - Multimodal fusion: Combines vision, language, and action representations
    """
    
    def __init__(self, model_name_or_path: str, shape_meta: dict):
        """
        Initialize NVILA model.
        
        Args:
            model_name_or_path: Path to the pre-trained VLM model
            shape_meta: Dictionary containing shape metadata for observations and actions
                Expected keys: 'action', 'obs' with nested 'horizon' information
        """
        super().__init__()
        self.shape_meta = shape_meta
        self.n_action_steps = shape_meta['action']['horizon']
        self.n_obs_steps = shape_meta['obs']['rgb']['horizon']

        # Load pre-trained vision-language model
        self.vlm = LlavaLlamaModel.from_pretrained(model_name_or_path)
        self.tokenizer = self.vlm.tokenizer
        self.image_processor = self.vlm.get_vision_tower().image_processor

        # Define action query token IDs (last n_action_steps tokens in vocabulary)
        self.action_query_token_ids = list(
            range(self.tokenizer.vocab_size - self.n_action_steps, self.tokenizer.vocab_size)
        )
        
        # Get image token ID for multimodal sequence construction
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NVILA model.
        
        Processes multi-frame RGB observations and text instructions to generate
        action query representations. The sequence structure follows:
        [text_prefix] + [img1 + sep + sep + img2 + sep + sep + ... + img6 + sep + sep] + [text_suffix] + [action_queries]
        
        Args:
            images: Multi-frame RGB observations, shape [B, n_obs_steps, H, W, 3]
            input_ids: Tokenized text instructions, shape [B, L]  
            attention_masks: Attention masks for text tokens, shape [B, L]
            
        Returns:
            action_query: Action query representations, shape [B, n_action_steps, D]
        """
        B = images.shape[0]
        L = input_ids.shape[1]
        D = self.vlm.llm.config.hidden_size
        
        # ===== Vision Processing =====
        # Flatten multi-frame images for batch processing: (B, n_obs_steps, H, W, 3) -> (B*n_obs_steps, H, W, 3)
        flattened_images = rearrange(images, 'b n h w c -> (b n) h w c')
        
        # Process images through vision pipeline
        processed_images = self.image_processor.preprocess(flattened_images)  # [B*n_obs_steps, 3, 448, 448]
        
        # Extract vision features and apply multimodal projection
        vision_features = self.vlm.get_vision_tower()(processed_images)  # Vision encoder output
        projected_vision_features = self.vlm.get_mm_projector()(vision_features)  # [B*n_obs_steps, 121, D]
        
        # Reshape to separate batch and frame dimensions: (B*n_obs_steps, 121, D) -> (B, n_obs_steps, 121, D)
        vision_features_per_frame = rearrange(projected_vision_features, '(b n) t d -> b n t d', b=B)
        
        # ===== Language Processing =====
        # Get text token embeddings
        text_embeds = self.vlm.llm.model.embed_tokens(input_ids)  # [B, L, D]
        
        # Prepare separator token embeddings for multimodal sequence
        sep_token_id = self.tokenizer.convert_tokens_to_ids('\n')
        sep_token_embedding = self.vlm.llm.model.embed_tokens(
            torch.tensor([sep_token_id], device=input_ids.device)
        )  # [1, D]
        
        # ===== Action Query Processing =====
        # Get action query token embeddings
        action_query_tokens = self.vlm.llm.model.embed_tokens(
            torch.tensor(self.action_query_token_ids, device=input_ids.device)
        )  # [n_action_steps, D]
        
        # ===== Multimodal Sequence Construction =====
        # Split text embeddings: first 13 tokens + remaining tokens
        text_prefix = text_embeds[:, :13]  # [B, 13, D] - Instruction prefix
        text_suffix = text_embeds[:, 13:]  # [B, L-13, D] - Instruction suffix
        
        # Create separator tokens batch: 2 separators per frame, 6 frames total = 12 separators
        sep_tokens = sep_token_embedding.expand(B, 2 * self.n_obs_steps, D)  # [B, 12, D]
        
        # Construct interleaved vision-separator sequence
        combined_seq = [text_prefix]
        
        for i in range(self.n_obs_steps):
            # Current frame's vision features: [B, 121, D]
            img_i = vision_features_per_frame[:, i]  # [B, 121, D]
            
            # Current frame's separator tokens: [B, 2, D]  
            sep_i = sep_tokens[:, i*2:(i+1)*2]  # [B, 2, D]
            
            # Append vision features followed by separators
            combined_seq.append(torch.cat([img_i, sep_i], dim=1))  # [B, 123, D]
        
        # Add text suffix and action query tokens
        combined_seq.extend([text_suffix, action_query_tokens])
        
        # Concatenate all sequence parts
        combined_embeddings = torch.cat(combined_seq, dim=1)  # [B, L + n_obs_steps*123 + n_action_steps, D]
        
        # ===== Attention Mask Construction =====
        # Split original attention masks
        prefix_mask = attention_masks[:, :13]  # [B, 13]
        suffix_mask = attention_masks[:, 13:]  # [B, L-13]
        
        # Create attention masks for vision and action tokens (all attend)
        vision_sep_mask = torch.ones(B, self.n_obs_steps * 123, dtype=attention_masks.dtype, device=attention_masks.device)
        action_query_mask = torch.ones(B, self.n_action_steps, dtype=attention_masks.dtype, device=attention_masks.device)
        
        # Concatenate all attention mask components
        complete_attention_mask = torch.cat([
            prefix_mask,
            vision_sep_mask, 
            suffix_mask,
            action_query_mask
        ], dim=1)
        
        # ===== VLM Forward Pass =====
        # Run complete sequence through vision-language model
        outputs = self.vlm.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=complete_attention_mask,
            return_dict=True,
        )
        
        # ===== Action Query Extraction =====
        # Extract hidden states corresponding to action query tokens (last n_action_steps positions)
        action_query = outputs.last_hidden_state[:, -self.n_action_steps:, :]  # [B, n_action_steps, D]
        
        return action_query