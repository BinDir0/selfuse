from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLTextConfig

from src.model.common.diffloss import SimpleMLPAdaLN
from src.model.vlm.qwen3_expert import Qwen3Expert
from src.model.vlm.prefix_cache import PrefixKVCache


# ---------------------------------------------------------------------------
# Helper: build a tiny action expert for testing
# ---------------------------------------------------------------------------

def _make_expert(monkeypatch, num_layers=2, attn_implementation="flex_attention", use_kv_projection=False):
    base_text_config = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
    )
    monkeypatch.setattr(
        Qwen3VLConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(text_config=base_text_config)),
    )
    return Qwen3Expert(
        model_name_or_path="dummy",
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_layers=num_layers,
        use_adaln=True,
        cond_hidden_size=32,
        attn_implementation=attn_implementation,
        use_kv_projection=use_kv_projection,
    )


def _move_expert_inputs_to_cuda(expert, inputs):
    expert = expert.cuda()
    cuda_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, PrefixKVCache):
            cuda_inputs[key] = PrefixKVCache(
                keys=value.keys.cuda(),
                values=value.values.cuda(),
                mask=value.mask.cuda(),
                lengths=value.lengths.cuda(),
            )
        elif torch.is_tensor(value):
            cuda_value = value.cuda()
            if value.requires_grad:
                cuda_value = cuda_value.detach().requires_grad_(True)
            cuda_inputs[key] = cuda_value
        else:
            cuda_inputs[key] = value
    return expert, cuda_inputs


requires_cuda_action_expert = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Qwen3Expert BlockMask tests require CUDA flex_attention.",
)


def _make_expert_inputs(expert, batch_size=2, prefix_len=3, action_len=4):
    num_layers = expert.num_layers
    prefix_cache = PrefixKVCache(
        keys=torch.randn(num_layers, batch_size, 2, prefix_len, 16),
        values=torch.randn(num_layers, batch_size, 2, prefix_len, 16),
        mask=torch.ones(batch_size, prefix_len, dtype=torch.bool),
        lengths=torch.full((batch_size,), prefix_len, dtype=torch.long),
    )
    suffix_embeds = torch.randn(batch_size, action_len, 64, requires_grad=True)
    cond = torch.randn(batch_size, action_len, 32, requires_grad=True)
    suffix_mask = torch.ones(batch_size, action_len, dtype=torch.bool)
    suffix_position_ids = torch.arange(action_len).unsqueeze(0).expand(batch_size, -1)
    return dict(
        suffix_embeds=suffix_embeds,
        prefix_cache=prefix_cache,
        suffix_position_ids=suffix_position_ids,
        cond=cond,
        suffix_mask=suffix_mask,
        num_parallel_chunks=1,
    )


# ---------------------------------------------------------------------------
# Existing tests (updated)
# ---------------------------------------------------------------------------

def test_diffloss_checkpoint_runs_only_in_train_mode():
    net = SimpleMLPAdaLN(
        in_channels=8,
        model_channels=16,
        out_channels=16,
        z_channels=4,
        num_res_blocks=2,
        grad_checkpointing=True,
    )
    x = torch.randn(3, 8)
    t = torch.randint(0, 10, (3,))
    c = torch.randn(3, 4)

    train_calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        train_calls.append(True)
        return fn(*args)

    net.train()
    with patch("src.model.common.diffloss.checkpoint", fake_checkpoint):
        net(x, t, c)
    assert len(train_calls) == 2

    eval_calls = []

    def fake_checkpoint_eval(fn, *args, **kwargs):
        eval_calls.append(True)
        return fn(*args)

    net.eval()
    with patch("src.model.common.diffloss.checkpoint", fake_checkpoint_eval):
        net(x, t, c)
    assert len(eval_calls) == 0


@requires_cuda_action_expert
def test_action_expert_checkpoint_runs_only_in_train_mode(monkeypatch):
    expert = _make_expert(monkeypatch)
    expert.enable_gradient_checkpointing()
    inputs = _make_expert_inputs(expert)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)

    train_calls = []

    def fake_gc(fn, *args, **kwargs):
        train_calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = fake_gc
    expert.train()
    out = expert(**inputs)
    out.sum().backward()
    assert len(train_calls) == expert.num_layers
    assert inputs["suffix_embeds"].grad is not None
    assert inputs["cond"].grad is not None

    eval_calls = []

    def fake_gc_eval(fn, *args, **kwargs):
        eval_calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = fake_gc_eval
    expert.eval()
    detached = {**inputs, "suffix_embeds": inputs["suffix_embeds"].detach(),
                "cond": inputs["cond"].detach()}
    expert(**detached)
    assert len(eval_calls) == 0


# ---------------------------------------------------------------------------
# New: every_n selective checkpointing
# ---------------------------------------------------------------------------

@requires_cuda_action_expert
def test_action_expert_every_n(monkeypatch):
    """With 6 layers and every_n=3, only layers 0 and 3 should be checkpointed."""
    expert = _make_expert(monkeypatch, num_layers=6)
    expert.enable_gradient_checkpointing(every_n=3)
    assert expert.checkpoint_every_n == 3

    inputs = _make_expert_inputs(expert)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)
    ckpt_layer_indices = []

    def tracking_gc(fn, *args, **kwargs):
        ckpt_layer_indices.append(len(ckpt_layer_indices))
        return fn(*args)

    expert._gradient_checkpointing_func = tracking_gc
    expert.train()
    out = expert(**inputs)
    out.sum().backward()

    # layers 0, 3 → 2 checkpoint calls
    assert len(ckpt_layer_indices) == 2


@requires_cuda_action_expert
def test_action_expert_every_n_equals_1_checkpoints_all(monkeypatch):
    """every_n=1 should checkpoint every layer (default)."""
    expert = _make_expert(monkeypatch, num_layers=4)
    expert.enable_gradient_checkpointing(every_n=1)

    inputs = _make_expert_inputs(expert)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)
    calls = []

    def tracking_gc(fn, *args, **kwargs):
        calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = tracking_gc
    expert.train()
    expert(**inputs).sum().backward()
    assert len(calls) == 4


def test_action_expert_disable_resets_every_n(monkeypatch):
    expert = _make_expert(monkeypatch)
    expert.enable_gradient_checkpointing(every_n=3)
    assert expert.checkpoint_every_n == 3
    expert.disable_gradient_checkpointing()
    assert expert.gradient_checkpointing is False
    assert expert.checkpoint_every_n == 1


# ---------------------------------------------------------------------------
# New: preserve_rng_state=False is set
# ---------------------------------------------------------------------------

def test_action_expert_preserve_rng_state_false(monkeypatch):
    """Verify that the default checkpoint func uses preserve_rng_state=False."""
    expert = _make_expert(monkeypatch)
    expert.enable_gradient_checkpointing()
    func = expert._gradient_checkpointing_func
    # partial(checkpoint, use_reentrant=False, preserve_rng_state=False)
    assert func.keywords.get("preserve_rng_state") is False
    assert func.keywords.get("use_reentrant") is False


@requires_cuda_action_expert
def test_kv_projection_identity_init_and_parity(monkeypatch):
    expert_plain = _make_expert(monkeypatch, use_kv_projection=False)
    expert_proj = _make_expert(monkeypatch, use_kv_projection=True)

    plain_state = expert_plain.state_dict()
    proj_state = expert_proj.state_dict()
    shared_state = {key: value for key, value in plain_state.items() if key in proj_state}
    missing, unexpected = expert_proj.load_state_dict(shared_state, strict=False)
    assert not unexpected
    assert set(missing) == {
        "layers.0.prefix_key_proj.weight",
        "layers.0.prefix_value_proj.weight",
        "layers.1.prefix_key_proj.weight",
        "layers.1.prefix_value_proj.weight",
    }

    for layer in expert_proj.layers:
        eye = torch.eye(layer.prefix_key_proj.weight.shape[0], dtype=layer.prefix_key_proj.weight.dtype)
        torch.testing.assert_close(layer.prefix_key_proj.weight, eye)
        torch.testing.assert_close(layer.prefix_value_proj.weight, eye)

    inputs = _make_expert_inputs(expert_plain)
    expert_plain, inputs = _move_expert_inputs_to_cuda(expert_plain, inputs)
    expert_proj = expert_proj.cuda()
    expert_plain.eval()
    expert_proj.eval()
    out_plain = expert_plain(**inputs)
    out_proj = expert_proj(**inputs)
    torch.testing.assert_close(out_plain, out_proj)


@requires_cuda_action_expert
def test_parallel_flow_block_mask_prevents_chunk_leakage(monkeypatch):
    expert = _make_expert(monkeypatch, attn_implementation="flex_attention")
    expert.eval()
    inputs = _make_expert_inputs(expert, batch_size=1, prefix_len=3, action_len=4)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)

    base_suffix_embeds = inputs["suffix_embeds"].clone()
    chunk0 = base_suffix_embeds[:, :2].clone()
    chunk1_a = base_suffix_embeds[:, 2:].clone()
    chunk1_b = chunk1_a + 10.0

    out_a = expert(
        **{
            **inputs,
            "suffix_embeds": torch.cat([chunk0, chunk1_a], dim=1),
            "num_parallel_chunks": 2,
        }
    )
    out_b = expert(
        **{
            **inputs,
            "suffix_embeds": torch.cat([chunk0, chunk1_b], dim=1),
            "num_parallel_chunks": 2,
        }
    )

    torch.testing.assert_close(out_a[:, :2], out_b[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_a[:, 2:], out_b[:, 2:])


@requires_cuda_action_expert
def test_parallel_chunk_forward_matches_separate_single_chunk(monkeypatch):
    expert = _make_expert(monkeypatch, attn_implementation="flex_attention")
    expert.eval()
    inputs = _make_expert_inputs(expert, batch_size=1, prefix_len=3, action_len=6)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)

    num_parallel_chunks = 3
    chunk_size = inputs["suffix_embeds"].shape[1] // num_parallel_chunks
    base_position_ids = torch.arange(chunk_size, device=inputs["suffix_embeds"].device).unsqueeze(0)
    packed_position_ids = base_position_ids.repeat(1, num_parallel_chunks)
    packed_output = expert(
        **{
            **inputs,
            "suffix_position_ids": packed_position_ids,
            "num_parallel_chunks": num_parallel_chunks,
        }
    )

    for chunk_idx in range(num_parallel_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size
        single_output = expert(
            suffix_embeds=inputs["suffix_embeds"][:, chunk_start:chunk_end],
            prefix_cache=inputs["prefix_cache"],
            suffix_position_ids=base_position_ids,
            cond=inputs["cond"][:, chunk_start:chunk_end],
            suffix_mask=inputs["suffix_mask"][:, chunk_start:chunk_end],
            num_parallel_chunks=1,
        )
        torch.testing.assert_close(
            packed_output[:, chunk_start:chunk_end],
            single_output,
            atol=1e-5,
            rtol=1e-5,
        )


@requires_cuda_action_expert
def test_parallel_chunk_forward_matches_separate_single_chunk_with_kv_projection(monkeypatch):
    expert = _make_expert(
        monkeypatch,
        attn_implementation="flex_attention",
        use_kv_projection=True,
    )
    expert.eval()
    inputs = _make_expert_inputs(expert, batch_size=1, prefix_len=3, action_len=6)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)

    num_parallel_chunks = 3
    chunk_size = inputs["suffix_embeds"].shape[1] // num_parallel_chunks
    base_position_ids = torch.arange(chunk_size, device=inputs["suffix_embeds"].device).unsqueeze(0)
    packed_position_ids = base_position_ids.repeat(1, num_parallel_chunks)
    packed_output = expert(
        **{
            **inputs,
            "suffix_position_ids": packed_position_ids,
            "num_parallel_chunks": num_parallel_chunks,
        }
    )

    for chunk_idx in range(num_parallel_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size
        single_output = expert(
            suffix_embeds=inputs["suffix_embeds"][:, chunk_start:chunk_end],
            prefix_cache=inputs["prefix_cache"],
            suffix_position_ids=base_position_ids,
            cond=inputs["cond"][:, chunk_start:chunk_end],
            suffix_mask=inputs["suffix_mask"][:, chunk_start:chunk_end],
            num_parallel_chunks=1,
        )
        torch.testing.assert_close(
            packed_output[:, chunk_start:chunk_end],
            single_output,
            atol=1e-5,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# BlockMask correctness: mask_mod pure logic tests
# ---------------------------------------------------------------------------

def _replicate_mask_mod(full_attention_mask_bool, prefix_len, chunk_size):
    """Replicate the exact mask_mod closure from Qwen3Expert.forward."""
    # Source: src/model/vlm/qwen3_expert.py Qwen3Expert.forward mask_mod
    def mask_mod(b, h, q_idx, kv_idx):
        del h
        valid = full_attention_mask_bool[b, kv_idx]
        is_prefix = kv_idx < prefix_len
        same_chunk = (q_idx // chunk_size) == ((kv_idx - prefix_len) // chunk_size)
        return valid & (is_prefix | same_chunk)
    return mask_mod


def _build_expected_mask(prefix_len, action_len, num_chunks, prefix_valid, action_valid):
    """Build expected bool mask [action_len, prefix_len + action_len] by definition.

    Rule: query q sees kv iff valid[kv] AND (kv is prefix OR kv is in same chunk as q).
    """
    chunk_size = action_len // num_chunks
    full_valid = prefix_valid + action_valid
    mask = []
    for q in range(action_len):
        row = []
        for kv in range(prefix_len + action_len):
            valid = full_valid[kv]
            is_prefix = kv < prefix_len
            same_chunk = (q // chunk_size) == ((kv - prefix_len) // chunk_size)
            row.append(valid and (is_prefix or same_chunk))
        mask.append(row)
    return mask


def _evaluate_mask_mod(mask_mod, batch_idx, action_len, total_kv_len):
    """Evaluate mask_mod for all (q, kv) pairs → [action_len, total_kv_len] bool list."""
    return [
        [bool(mask_mod(batch_idx, 0, q, kv)) for kv in range(total_kv_len)]
        for q in range(action_len)
    ]


def test_mask_mod_no_padding_two_chunks():
    """Every (q, kv) cell matches expected mask: 2 prefix, 4 action, T=2."""
    prefix_len, action_len, num_chunks = 2, 4, 2
    chunk_size = action_len // num_chunks

    full_mask = torch.ones(1, prefix_len + action_len, dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)
    actual = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)

    #         P0    P1    A0:C0  A1:C0  A2:C1  A3:C1
    assert actual[0] == [True, True, True,  True,  False, False]  # q0, chunk 0
    assert actual[1] == [True, True, True,  True,  False, False]  # q1, chunk 0
    assert actual[2] == [True, True, False, False, True,  True]   # q2, chunk 1
    assert actual[3] == [True, True, False, False, True,  True]   # q3, chunk 1

    expected = _build_expected_mask(
        prefix_len, action_len, num_chunks,
        [True] * prefix_len, [True] * action_len,
    )
    assert actual == expected


def test_mask_mod_prefix_padding():
    """Padded prefix position (idx=2) is masked for all queries."""
    prefix_len, action_len, num_chunks = 3, 4, 2
    chunk_size = action_len // num_chunks
    prefix_valid = [True, True, False]
    action_valid = [True, True, True, True]

    full_mask = torch.tensor([prefix_valid + action_valid], dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)
    actual = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)

    for q in range(action_len):
        assert not actual[q][2], f"Padded prefix kv=2 should be False for q={q}"
    expected = _build_expected_mask(prefix_len, action_len, num_chunks, prefix_valid, action_valid)
    assert actual == expected


def test_mask_mod_action_padding():
    """Padded action position masked out even within same chunk."""
    prefix_len, action_len, num_chunks = 2, 4, 2
    chunk_size = action_len // num_chunks
    prefix_valid = [True, True]
    action_valid = [True, True, True, False]

    full_mask = torch.tensor([prefix_valid + action_valid], dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)
    actual = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)

    # kv=5 (action idx 3, chunk 1) False even for same-chunk queries
    assert not actual[2][5], "Padded action kv=5 should be False for q=2 (same chunk)"
    assert not actual[3][5], "Padded action kv=5 should be False for q=3 (same chunk)"
    expected = _build_expected_mask(prefix_len, action_len, num_chunks, prefix_valid, action_valid)
    assert actual == expected


def test_mask_mod_combined_padding():
    """Both prefix and action padding applied simultaneously."""
    prefix_len, action_len, num_chunks = 3, 4, 2
    chunk_size = action_len // num_chunks
    prefix_valid = [True, False, True]
    action_valid = [True, True, False, True]

    full_mask = torch.tensor([prefix_valid + action_valid], dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)
    actual = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)

    for q in range(action_len):
        assert not actual[q][1], f"Padded prefix kv=1 should be False for q={q}"
    # kv=5 (action idx 2, chunk 1) False for same-chunk queries
    assert not actual[2][5]
    assert not actual[3][5]
    expected = _build_expected_mask(prefix_len, action_len, num_chunks, prefix_valid, action_valid)
    assert actual == expected


def test_mask_mod_three_chunks():
    """Correctness with T=3: 3 prefix, 6 action, 3 chunks of size 2."""
    prefix_len, action_len, num_chunks = 3, 6, 3
    chunk_size = action_len // num_chunks

    full_mask = torch.ones(1, prefix_len + action_len, dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)
    actual = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)

    # KV layout: P0 P1 P2 | A0:C0 A1:C0 | A2:C1 A3:C1 | A4:C2 A5:C2
    assert actual[0] == [True, True, True, True, True, False, False, False, False]
    assert actual[1] == [True, True, True, True, True, False, False, False, False]
    assert actual[2] == [True, True, True, False, False, True, True, False, False]
    assert actual[3] == [True, True, True, False, False, True, True, False, False]
    assert actual[4] == [True, True, True, False, False, False, False, True, True]
    assert actual[5] == [True, True, True, False, False, False, False, True, True]

    expected = _build_expected_mask(prefix_len, action_len, num_chunks, [True] * 3, [True] * 6)
    assert actual == expected


def test_mask_mod_single_chunk_is_bidirectional():
    """T=1: all action tokens see all valid positions (fully bidirectional)."""
    prefix_len, action_len = 2, 4
    chunk_size = action_len  # single chunk

    full_mask = torch.ones(1, prefix_len + action_len, dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)
    actual = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)

    for q in range(action_len):
        assert actual[q] == [True] * (prefix_len + action_len), (
            f"q={q} should see all valid positions in single-chunk mode"
        )


def test_mask_mod_batch_independence():
    """Different batch elements have independent padding patterns."""
    prefix_len, action_len, num_chunks = 2, 4, 2
    chunk_size = action_len // num_chunks

    full_mask = torch.tensor([
        [True, True, True, True, True, True],   # batch 0: all valid
        [True, False, True, True, True, False],  # batch 1: prefix[1] + action[3] padded
    ], dtype=torch.bool)
    mask_mod = _replicate_mask_mod(full_mask, prefix_len, chunk_size)

    actual_b0 = _evaluate_mask_mod(mask_mod, 0, action_len, prefix_len + action_len)
    actual_b1 = _evaluate_mask_mod(mask_mod, 1, action_len, prefix_len + action_len)

    expected_b0 = _build_expected_mask(
        prefix_len, action_len, num_chunks,
        [True, True], [True, True, True, True],
    )
    expected_b1 = _build_expected_mask(
        prefix_len, action_len, num_chunks,
        [True, False], [True, True, True, False],
    )

    assert actual_b0 == expected_b0
    assert actual_b1 == expected_b1
    assert actual_b0 != actual_b1


# ---------------------------------------------------------------------------
# BlockMask correctness: forward-level isolation tests
# ---------------------------------------------------------------------------

@requires_cuda_action_expert
def test_forward_same_chunk_influence(monkeypatch):
    """Perturbing position 0 (chunk 0) changes chunk 0 output but not chunk 1."""
    expert = _make_expert(monkeypatch, attn_implementation="flex_attention")
    expert.eval()
    inputs = _make_expert_inputs(expert, batch_size=1, prefix_len=3, action_len=4)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)

    base = inputs["suffix_embeds"].clone()
    perturbed = base.clone()
    perturbed[:, 0] += 10.0

    out_base = expert(**{**inputs, "num_parallel_chunks": 2})
    out_pert = expert(**{**inputs, "suffix_embeds": perturbed, "num_parallel_chunks": 2})

    assert not torch.allclose(out_base[:, :2], out_pert[:, :2]), (
        "Chunk 0 should change when a chunk-0 position is perturbed"
    )
    torch.testing.assert_close(out_base[:, 2:], out_pert[:, 2:], atol=1e-5, rtol=1e-5)


@requires_cuda_action_expert
def test_forward_padding_queries_produce_finite_output(monkeypatch):
    """Queries at padded action positions produce finite output (no NaN).

    Padded Q positions still attend to the prefix, so softmax never gets an
    all-masked row. The final suffix_mask multiply zeros them out, but the
    intermediate hidden states must remain finite.
    """
    expert = _make_expert(monkeypatch, attn_implementation="flex_attention")
    expert.eval()
    inputs = _make_expert_inputs(expert, batch_size=1, prefix_len=3, action_len=4)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)
    inputs["suffix_mask"][:, 3] = False

    out = expert(**{**inputs, "num_parallel_chunks": 2})
    assert torch.isfinite(out).all(), "Output contains NaN or Inf with padded query"
    # Padded position zeroed by final suffix_mask multiply
    torch.testing.assert_close(
        out[:, 3], torch.zeros_like(out[:, 3]),
        atol=0.0, rtol=0.0,
    )


@requires_cuda_action_expert
def test_forward_three_chunk_isolation(monkeypatch):
    """With T=3, perturbing chunk 1 leaves chunks 0 and 2 unchanged."""
    expert = _make_expert(monkeypatch, attn_implementation="flex_attention")
    expert.eval()
    inputs = _make_expert_inputs(expert, batch_size=1, prefix_len=3, action_len=6)
    expert, inputs = _move_expert_inputs_to_cuda(expert, inputs)

    base = inputs["suffix_embeds"].clone()
    perturbed = base.clone()
    perturbed[:, 2:4] += 10.0  # chunk 1

    out_base = expert(**{**inputs, "num_parallel_chunks": 3})
    out_pert = expert(**{**inputs, "suffix_embeds": perturbed, "num_parallel_chunks": 3})

    torch.testing.assert_close(out_base[:, :2], out_pert[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_base[:, 2:4], out_pert[:, 2:4])
    torch.testing.assert_close(out_base[:, 4:], out_pert[:, 4:], atol=1e-5, rtol=1e-5)
