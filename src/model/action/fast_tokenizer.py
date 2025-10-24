# https://huggingface.co/physical-intelligence/fast/blob/main/processing_action_tokenizer.py

import logging
from typing import ClassVar
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import os

import numpy as np
from scipy.fft import dct
from scipy.fft import idct
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin


def _process_dct_chunk(action_chunk: np.array) -> np.array:
    """Helper function to process DCT for a single action chunk."""
    return dct(action_chunk, axis=0, norm="ortho").flatten()


def _process_single_dct_to_string(tokens, scale, min_token):
    """Helper function to process a single dct_token array"""
    rounded_tokens = np.around(tokens * scale) - min_token
    rounded_tokens = rounded_tokens.astype(int)
    return "".join(map(chr, rounded_tokens))


class UniversalActionProcessor(ProcessorMixin):
    attributes: ClassVar[list[str]] = ["bpe_tokenizer"]
    bpe_tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        bpe_tokenizer: PreTrainedTokenizerFast,
        scale: float = 10,
        vocab_size: int = 1024,
        min_token: int = 0,
        max_token: int = 1024,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ):
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token
        self.max_token = max_token
        
        # Action horizon and dimension needed during decoding. These can be specified
        # in three ways (in order of priority):
        # 1. passed in as kwargs to decode()
        # 2. in the constructor
        # 3. cached from the last time decode() was called
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

        super().__init__(bpe_tokenizer)

    def __call__(self, action_chunk: np.array) -> np.array:
        assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]

        # Cache the time horizon and action dimension for decoding
        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]

        dct_coeff = dct(action_chunk, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)
        tokens = []
        for elem in dct_coeff:
            elem_clipped = np.clip(elem.flatten(), self.min_token, self.max_token)
            token_str = "".join(map(chr, (elem_clipped - self.min_token).astype(int)))
            tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
        return tokens

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.array:
        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert (
            self.time_horizon is not None and self.action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.min_token
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert (
                    decoded_dct_coeff.shape
                    == (
                        self.time_horizon,
                        self.action_dim,
                    )
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)

    @classmethod
    def fit(
        cls,
        action_data: list[np.array],
        scale: float = 10,
        vocab_size: int = 1024,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        num_workers: int | None = None,
    ) -> "UniversalActionProcessor":
        """
        Fit the UniversalActionProcessor with parallel DCT computation.
        
        Args:
            action_data: List of action arrays to process
            scale: Scaling factor for quantization
            vocab_size: Size of the vocabulary
            time_horizon: Number of time steps (inferred from data if None)
            action_dim: Action dimension (inferred from data if None)
            num_workers: Number of parallel workers (defaults to CPU count if None)
        """
        if time_horizon is None:
            time_horizon = action_data[0].shape[0]
        if action_dim is None:
            action_dim = action_data[0].shape[1]
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 10) # leave 10 cores for other tasks
        num_workers = min(num_workers, len(action_data))  # Don't use more workers than data chunks
        
        print(f"Processing {len(action_data)} action chunks using {num_workers} workers...")
        
        # Run DCT over all inputs in parallel
        if num_workers > 1 and len(action_data) > 1:
            with Pool(processes=num_workers) as pool:
                # Use imap for progress tracking
                dct_tokens = list(tqdm(
                    pool.imap(_process_dct_chunk, action_data),
                    total=len(action_data),
                    desc="Computing DCT",
                    unit="sequence"
                ))
        else:
            # Fallback to sequential processing for small datasets or single worker
            dct_tokens = []
            for i, action_chunk in enumerate(tqdm(action_data, desc="Computing DCT", unit="sequence")):
                dct_tokens.append(_process_dct_chunk(action_chunk))

        # Quantize and find min token
        max_token = int(np.around(np.concatenate(dct_tokens) * scale).max())
        min_token = int(np.around(np.concatenate(dct_tokens) * scale).min())
        min_vocab_size = max_token - min_token
        
        print(f"Min token: {min_token}, Max token: {max_token}, Min vocab size: {min_vocab_size}")

        assert (
            min_vocab_size <= vocab_size
        ), f"Vocab size {vocab_size} is too small for the range of tokens {min_vocab_size}"
        if min_vocab_size + 100 > vocab_size:
            logging.warning(
                f"Initial alphabet size {min_vocab_size} is almost as large as the vocab"
                f"size {vocab_size}, consider increasing vocab size"
            )

        # Make token iterator for BPE training
        def _token_iter():
            for tokens in dct_tokens:
                rounded_tokens = np.around(tokens * scale) - min_token
                rounded_tokens = rounded_tokens.astype(int)
                string = "".join(map(chr, rounded_tokens))
                yield string

        # Train BPE tokenizer
        bpe = ByteLevelBPETokenizer()

        # Set up the entire range of possible tokens as the initial alphabet
        alphabet = [chr(i) for i in range(max_token - min_token + 1)]
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=10000,
        )

        # Train the inner tokenizer (don't use ByteLevelBPETokenizer.train_from_iterator()
        # because it doesn't support custom alphabets)
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # set this to enable parallelism
        bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)

        return cls(
            PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
            scale=scale,
            vocab_size=vocab_size,
            min_token=min_token,
            max_token=max_token,
            time_horizon=time_horizon,
            action_dim=action_dim,
        )

    def setup_tokenizer_gemma_mappings(self, usable_token_ids: list, start_idx: int = 0):
        """
        Build mappings between Fast action tokens and Gemma tokenizer token IDs for this processor.
        
        Args:
            usable_token_ids: List of usable Gemma token IDs
            start_idx: Starting index in usable_token_ids to use
            
        Returns:
            Tuple of (token_id2gemma_token_id, gemma_token_id2token_id, end_idx)
            - token_id2gemma_token_id: {token_id: gemma_id}
            - gemma_token_id2token_id: {gemma_id: token_id}
            - end_idx: Next index to use in usable_token_ids
        """
        # Initialize mappings (one-level for Fast tokenizer)
        token_id2gemma_token_id = {}
        gemma_token_id2token_id = {}
        
        replace_idx = start_idx
        for i in range(self.vocab_size):
            gemma_id = usable_token_ids[replace_idx]
            replace_idx += 1
            token_id2gemma_token_id[i] = gemma_id
            gemma_token_id2token_id[gemma_id] = i
        
        return token_id2gemma_token_id, gemma_token_id2token_id, replace_idx

    def map_hand_tokens2gemma(self, hand_tokens_1d, mapping):
        """
        Map VQ token IDs to Gemma token IDs while preserving time-interleaved order.
        
        Args:
            vq_ids_1d: 1D array of VQ token IDs [t0_wrist, t0_hand, t1_wrist, t1_hand, ...]
            mapping: vq_token_id2gemma_token_id dict
            
        Returns:
            1D array of Gemma token IDs [t0_wrist, t0_hand, t1_wrist, t1_hand, ...]
        """

        
        return np.array([mapping[id] for id in hand_tokens_1d])