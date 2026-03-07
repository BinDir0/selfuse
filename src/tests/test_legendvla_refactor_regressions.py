import importlib
import random
import sys
import types

import torch
from torch import nn

from src.policy.legendvla_utils import build_causal_mask_and_position_ids


def _install_bitsandbytes_stub() -> None:
    if "bitsandbytes" in sys.modules:
        return

    bitsandbytes = types.ModuleType("bitsandbytes")

    class Params4bit(torch.nn.Parameter):
        pass

    class Linear4bit(torch.nn.Linear):
        pass

    bitsandbytes.nn = types.SimpleNamespace(
        Params4bit=Params4bit,
        Linear4bit=Linear4bit,
    )
    bitsandbytes.functional = types.SimpleNamespace(
        quantize_4bit=lambda tensor, *args, **kwargs: (tensor, None),
        dequantize_4bit=lambda tensor, *args, **kwargs: tensor,
    )
    sys.modules["bitsandbytes"] = bitsandbytes


def _load_legendvla_class():
    _install_bitsandbytes_stub()
    module_name = "src.policy.legendvla"
    if module_name in sys.modules:
        return sys.modules[module_name].LegendVLA

    original_compile = torch.compile
    torch.compile = lambda fn=None, **kwargs: fn if fn is not None else (lambda inner: inner)
    try:
        module = importlib.import_module(module_name)
    finally:
        torch.compile = original_compile
    return module.LegendVLA


def _reference_build_causal_mask_and_position_ids(
    attention_mask: torch.Tensor,
    answer_start_idx: torch.Tensor,
    n_actions: torch.Tensor,
    num_action_tokens: int,
    dtype: torch.dtype,
):
    bsz = attention_mask.size(0)
    device = attention_mask.device
    max_vlm_tokens = attention_mask.shape[-1]
    total_num_tokens = max_vlm_tokens + num_action_tokens
    action_start = max_vlm_tokens
    vlm_token_cnts = torch.sum(attention_mask, dim=1)
    causal_mask = torch.full(
        (bsz, total_num_tokens, total_num_tokens),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    for idx in range(bsz):
        cnt = int(vlm_token_cnts[idx].item())
        start = int(answer_start_idx[idx].item())
        answer_len = cnt - start
        n_action = int(n_actions[idx].item())
        causal_mask[idx, :cnt, :start] = 0
        mask = torch.tril(torch.ones((answer_len, answer_len), dtype=torch.bool, device=device))
        causal_mask[idx, start:cnt, start:cnt] = torch.where(
            mask,
            torch.zeros(1, dtype=dtype, device=device),
            torch.full((1,), torch.finfo(dtype).min, dtype=dtype, device=device),
        )
        causal_mask[idx, action_start:action_start + n_action, :start] = 0
        causal_mask[idx, action_start:action_start + n_action, action_start:action_start + n_action] = 0

    causal_mask = causal_mask.unsqueeze(1)
    vlm_position_ids = torch.arange(1, max_vlm_tokens + 1, device=device).repeat(bsz, 1)
    action_position_ids = (
        torch.arange(0, num_action_tokens, device=device).unsqueeze(0)
        + answer_start_idx.unsqueeze(1)
        + 1
    )
    return causal_mask, vlm_position_ids, action_position_ids


class _FlattenToSinglePatch(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(values.shape[0], 1, -1).to(torch.float32)


class _LinearEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values.to(torch.float32))


class _BaseDummyLegendVLA:
    def __init__(self) -> None:
        self.vlm_hidden_size = 4
        self.image_token_index = 99
        self.pad_token_id = 0
        self.state_token_index = 77
        self.action_token_index = 88
        self.ar_action_noise_std = 0.0
        self.training = False
        self.embed_tokens = nn.Embedding(128, self.vlm_hidden_size, padding_idx=self.pad_token_id)
        self.action_encoder_ar = _LinearEncoder()

        with torch.no_grad():
            token_weights = torch.arange(128 * self.vlm_hidden_size, dtype=torch.float32)
            token_weights = token_weights.reshape(128, self.vlm_hidden_size)
            self.embed_tokens.weight.copy_(token_weights / 100.0)
            self.embed_tokens.weight[self.pad_token_id].zero_()


class _DummyLegendVLANoDepthModel(_BaseDummyLegendVLA):
    def __init__(self) -> None:
        super().__init__()
        self.use_depth = False
        self.vision_tower = _FlattenToSinglePatch()
        self.multi_modal_projector = nn.Identity()


class _DummyLegendVLADepthModel(_BaseDummyLegendVLA):
    def __init__(self) -> None:
        super().__init__()
        self.use_depth = True
        self.depth_dropout = 0.0
        self.vision_tower = _FlattenToSinglePatch()
        self.depth_encoder = _FlattenToSinglePatch()
        self.depth_missing_embeddings = nn.Parameter(
            torch.tensor([[0.3, -0.7]], dtype=torch.float32)
        )
        self.multi_modal_projector = nn.Identity()


def _reference_forward_siglip_and_text_embedding(
    model,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor = None,
    depth_values: torch.FloatTensor = None,
    has_depth_values: torch.LongTensor = None,
    states: torch.FloatTensor = None,
    actions: torch.FloatTensor = None,
    n_states: torch.LongTensor = None,
    n_actions: torch.LongTensor = None,
    is_vla_data=None,
    dtype: torch.dtype = torch.float32,
) -> torch.FloatTensor:
    inputs_embeds = model.embed_tokens(input_ids)
    device = inputs_embeds.device

    if pixel_values is not None:
        if pixel_values.ndim == 5:
            batch_size, frame_count, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * frame_count, channels, height, width)
        else:
            frame_count = None
            batch_size = pixel_values.shape[0]

        rgb_image_features = model.vision_tower(pixel_values)

        if model.use_depth and depth_values is not None:
            if depth_values.ndim == 5:
                depth_batch, frame_count, depth_channels, depth_height, depth_width = depth_values.shape
                depth_values = depth_values.view(depth_batch * frame_count, depth_channels, depth_height, depth_width)
            else:
                depth_batch = depth_values.shape[0]
                frame_count = None
            depth_image_features = model.depth_encoder(depth_values)
        else:
            depth_image_features = None
            depth_batch = batch_size

        if frame_count is not None:
            rgb_image_features = rgb_image_features.view(batch_size, -1, rgb_image_features.shape[-1])
            if depth_image_features is not None:
                depth_image_features = depth_image_features.view(depth_batch, -1, depth_image_features.shape[-1])

    bsz, seq_len = input_ids.shape
    final_embedding = torch.full(
        (bsz, seq_len, model.vlm_hidden_size), 0, dtype=dtype, device=device
    )

    text_mask = (input_ids != model.image_token_index) & (input_ids != model.pad_token_id)
    final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
    state_mask = input_ids == model.state_token_index
    action_mask = input_ids == model.action_token_index

    if n_states is not None:
        assert torch.all(n_states == state_mask.sum(dim=1))
    if n_actions is not None:
        assert torch.all(n_actions == action_mask.sum(dim=1))

    if states is not None:
        state_features = model.action_encoder_ar(states) / (model.vlm_hidden_size ** 0.5)
    if actions is not None:
        actions_input = actions
        noise = torch.randn_like(actions_input) * model.ar_action_noise_std
        actions_input = actions_input + noise
        action_features = model.action_encoder_ar(actions_input) / (model.vlm_hidden_size ** 0.5)
    if pixel_values is not None:
        image_mask = input_ids == model.image_token_index

    for batch_idx in range(bsz):
        if pixel_values is not None:
            image_indices = image_mask[batch_idx].nonzero(as_tuple=True)[0]
            if depth_image_features is None:
                depth_image_feature = None
            elif has_depth_values is not None and has_depth_values[batch_idx] and not (
                model.training and random.random() < model.depth_dropout
            ):
                depth_image_feature = depth_image_features[batch_idx]
            else:
                if frame_count is not None:
                    depth_image_feature = model.depth_missing_embeddings.repeat(frame_count, 1)
                else:
                    depth_image_feature = model.depth_missing_embeddings
            if depth_image_feature is not None:
                paired_image_features = torch.cat(
                    [rgb_image_features[batch_idx], depth_image_feature], dim=-1
                )
            else:
                paired_image_features = rgb_image_features[batch_idx]
            paired_image_features = paired_image_features.view(-1, paired_image_features.shape[-1])
            paired_image_features = model.multi_modal_projector(paired_image_features)
            scaled_image_features = paired_image_features / (model.vlm_hidden_size ** 0.5)
            final_embedding[batch_idx, image_indices] = scaled_image_features
        if is_vla_data is not None and is_vla_data[batch_idx]:
            if n_states is not None:
                final_embedding[batch_idx, state_mask[batch_idx]] = state_features[
                    batch_idx, : n_states[batch_idx]
                ].to(final_embedding.dtype)
            if n_actions is not None:
                final_embedding[batch_idx, action_mask[batch_idx]] = action_features[
                    batch_idx, : n_actions[batch_idx]
                ].to(final_embedding.dtype)
    return final_embedding


def test_build_causal_mask_and_position_ids_matches_original_logic():
    LegendVLA = _load_legendvla_class()
    dtype = torch.float32
    num_action_tokens = 3

    cases = [
        (
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long),
            torch.tensor([2, 1], dtype=torch.long),
            torch.tensor([2, 1], dtype=torch.long),
        ),
        (
            torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
            torch.tensor([0, 4], dtype=torch.long),
            torch.tensor([0, 3], dtype=torch.long),
        ),
    ]

    generator = torch.Generator().manual_seed(0)
    max_vlm_tokens = 6
    for _ in range(6):
        counts = torch.randint(1, max_vlm_tokens + 1, (3,), generator=generator)
        attention_mask = torch.zeros(3, max_vlm_tokens, dtype=torch.long)
        answer_start_idx = torch.zeros(3, dtype=torch.long)
        for batch_idx, count in enumerate(counts.tolist()):
            attention_mask[batch_idx, :count] = 1
            answer_start_idx[batch_idx] = torch.randint(0, count + 1, (1,), generator=generator)
        n_actions = torch.randint(0, num_action_tokens + 1, (3,), generator=generator)
        cases.append((attention_mask, answer_start_idx, n_actions))

    dummy_self = types.SimpleNamespace(num_action_tokens=num_action_tokens)
    for attention_mask, answer_start_idx, n_actions in cases:
        expected = _reference_build_causal_mask_and_position_ids(
            attention_mask,
            answer_start_idx,
            n_actions,
            num_action_tokens,
            dtype,
        )
        utility = build_causal_mask_and_position_ids(
            attention_mask,
            answer_start_idx,
            n_actions,
            num_action_tokens,
            dtype,
        )
        wrapper = LegendVLA.build_causal_mask_and_position_ids(
            dummy_self,
            attention_mask,
            answer_start_idx,
            n_actions,
            dtype,
        )
        for actual in (utility, wrapper):
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            torch.testing.assert_close(actual[2], expected[2])

    min_value = torch.finfo(dtype).min
    assert utility[0][0, 0, 0, -1].item() == min_value


def test_forward_siglip_and_text_embedding_matches_original_logic_without_depth():
    LegendVLA = _load_legendvla_class()
    model = _DummyLegendVLANoDepthModel()

    input_ids = torch.tensor(
        [
            [11, 99, 99, 77, 88, 12, 0],
            [21, 99, 99, 77, 88, 22, 0],
        ],
        dtype=torch.long,
    )
    pixel_values = torch.tensor(
        [
            [
                [[[1.0, 2.0, 3.0, 4.0]]],
                [[[5.0, 6.0, 7.0, 8.0]]],
            ],
            [
                [[[9.0, 10.0, 11.0, 12.0]]],
                [[[13.0, 14.0, 15.0, 16.0]]],
            ],
        ],
        dtype=torch.float32,
    )
    states = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [
            [[0.5, 1.5]],
            [[2.5, 3.5]],
        ],
        dtype=torch.float32,
    )
    n_states = torch.tensor([1, 1], dtype=torch.long)
    n_actions = torch.tensor([1, 1], dtype=torch.long)
    is_vla_data = torch.tensor([True, False], dtype=torch.bool)

    expected = _reference_forward_siglip_and_text_embedding(
        model=model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        states=states,
        actions=actions,
        n_states=n_states,
        n_actions=n_actions,
        is_vla_data=is_vla_data,
        dtype=torch.float32,
    )
    actual = LegendVLA._forward_siglip_and_text_embedding(
        model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        states=states,
        actions=actions,
        n_states=n_states,
        n_actions=n_actions,
        is_vla_data=is_vla_data,
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0, 3], model.embed_tokens(input_ids)[0, 3])
    torch.testing.assert_close(actual[1, 3], model.embed_tokens(input_ids)[1, 3])
    torch.testing.assert_close(actual[1, 4], model.embed_tokens(input_ids)[1, 4])


def test_forward_siglip_and_text_embedding_requires_n_states_with_states():
    LegendVLA = _load_legendvla_class()
    model = _DummyLegendVLANoDepthModel()

    try:
        LegendVLA._forward_siglip_and_text_embedding(
            model,
            input_ids=torch.tensor([[11, 99, 77, 12, 0]], dtype=torch.long),
            pixel_values=torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]], dtype=torch.float32),
            states=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
            n_states=None,
            dtype=torch.float32,
        )
    except AssertionError as exc:
        assert str(exc) == "states and n_states must be provided together"
    else:
        raise AssertionError("Expected assertion when states is provided without n_states")



def test_forward_siglip_and_text_embedding_requires_n_actions_with_actions():
    LegendVLA = _load_legendvla_class()
    model = _DummyLegendVLANoDepthModel()

    try:
        LegendVLA._forward_siglip_and_text_embedding(
            model,
            input_ids=torch.tensor([[11, 99, 88, 12, 0]], dtype=torch.long),
            pixel_values=torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]], dtype=torch.float32),
            actions=torch.tensor([[[0.5, 1.5]]], dtype=torch.float32),
            n_actions=None,
            dtype=torch.float32,
        )
    except AssertionError as exc:
        assert str(exc) == "actions and n_actions must be provided together"
    else:
        raise AssertionError("Expected assertion when actions is provided without n_actions")


def test_forward_siglip_and_text_embedding_matches_original_logic_with_depth():
    LegendVLA = _load_legendvla_class()
    model = _DummyLegendVLADepthModel()

    input_ids = torch.tensor(
        [
            [31, 99, 99, 77, 88, 32, 0],
            [41, 99, 99, 77, 88, 42, 0],
        ],
        dtype=torch.long,
    )
    pixel_values = torch.tensor(
        [
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ],
            [
                [[[5.0, 6.0]]],
                [[[7.0, 8.0]]],
            ],
        ],
        dtype=torch.float32,
    )
    depth_values = torch.tensor(
        [
            [
                [[[10.0, 20.0]]],
                [[[30.0, 40.0]]],
            ],
            [
                [[[50.0, 60.0]]],
                [[[70.0, 80.0]]],
            ],
        ],
        dtype=torch.float32,
    )
    has_depth_values = torch.tensor([True, False], dtype=torch.bool)
    states = torch.tensor(
        [
            [[1.0, 1.5]],
            [[2.0, 2.5]],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [
            [[0.25, 0.75]],
            [[1.25, 1.75]],
        ],
        dtype=torch.float32,
    )
    n_states = torch.tensor([1, 1], dtype=torch.long)
    n_actions = torch.tensor([1, 1], dtype=torch.long)
    is_vla_data = torch.tensor([True, False], dtype=torch.bool)

    expected = _reference_forward_siglip_and_text_embedding(
        model=model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        depth_values=depth_values,
        has_depth_values=has_depth_values,
        states=states,
        actions=actions,
        n_states=n_states,
        n_actions=n_actions,
        is_vla_data=is_vla_data,
        dtype=torch.float32,
    )
    actual = LegendVLA._forward_siglip_and_text_embedding(
        model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        depth_values=depth_values,
        has_depth_values=has_depth_values,
        states=states,
        actions=actions,
        n_states=n_states,
        n_actions=n_actions,
        is_vla_data=is_vla_data,
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0, 1], actual[1, 1])
    torch.testing.assert_close(actual[1, 3], model.embed_tokens(input_ids)[1, 3])
    torch.testing.assert_close(actual[1, 4], model.embed_tokens(input_ids)[1, 4])


if __name__ == "__main__":
    test_build_causal_mask_and_position_ids_matches_original_logic()
    test_forward_siglip_and_text_embedding_matches_original_logic_without_depth()
    test_forward_siglip_and_text_embedding_requires_n_states_with_states()
    test_forward_siglip_and_text_embedding_requires_n_actions_with_actions()
    test_forward_siglip_and_text_embedding_matches_original_logic_with_depth()
    print("ok")
