from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLTextConfig

from src.model.common.diffloss import SimpleMLPAdaLN
from src.model.action.qwen3_action_expert import Qwen3ActionExpert
from src.model.vlm.prefix_cache import PrefixKVCache


# ---------------------------------------------------------------------------
# Helper: build a tiny action expert for testing
# ---------------------------------------------------------------------------

def _make_expert(monkeypatch, num_layers=2):
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
    return Qwen3ActionExpert(
        model_name_or_path="dummy",
        time_hidden_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        attn_implementation="eager",
    )


def _make_expert_inputs(expert, batch_size=2, prefix_len=3, action_len=4):
    num_layers = expert.num_layers
    prefix_cache = PrefixKVCache(
        keys=torch.randn(num_layers, batch_size, 2, prefix_len, 16),
        values=torch.randn(num_layers, batch_size, 2, prefix_len, 16),
        mask=torch.ones(batch_size, prefix_len, dtype=torch.bool),
        lengths=torch.full((batch_size,), prefix_len, dtype=torch.long),
    )
    action_embeds = torch.randn(batch_size, action_len, 64, requires_grad=True)
    time_cond = torch.randn(batch_size, action_len, 32, requires_grad=True)
    action_mask = torch.ones(batch_size, action_len, dtype=torch.bool)
    action_position_ids = torch.arange(action_len).unsqueeze(0).expand(batch_size, -1)
    return dict(
        action_embeds=action_embeds,
        prefix_cache=prefix_cache,
        action_position_ids=action_position_ids,
        time_cond=time_cond,
        action_mask=action_mask,
        mode="flow",
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


def test_action_expert_checkpoint_runs_only_in_train_mode(monkeypatch):
    expert = _make_expert(monkeypatch)
    expert.enable_gradient_checkpointing()
    inputs = _make_expert_inputs(expert)

    train_calls = []

    def fake_gc(fn, *args, **kwargs):
        train_calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = fake_gc
    expert.train()
    out = expert(**inputs)
    out.sum().backward()
    assert len(train_calls) == expert.num_layers
    assert inputs["action_embeds"].grad is not None
    assert inputs["time_cond"].grad is not None

    eval_calls = []

    def fake_gc_eval(fn, *args, **kwargs):
        eval_calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = fake_gc_eval
    expert.eval()
    detached = {**inputs, "action_embeds": inputs["action_embeds"].detach(),
                "time_cond": inputs["time_cond"].detach()}
    expert(**detached)
    assert len(eval_calls) == 0


# ---------------------------------------------------------------------------
# New: every_n selective checkpointing
# ---------------------------------------------------------------------------

def test_action_expert_every_n(monkeypatch):
    """With 6 layers and every_n=3, only layers 0 and 3 should be checkpointed."""
    expert = _make_expert(monkeypatch, num_layers=6)
    expert.enable_gradient_checkpointing(every_n=3)
    assert expert.checkpoint_every_n == 3

    inputs = _make_expert_inputs(expert)
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


def test_action_expert_every_n_equals_1_checkpoints_all(monkeypatch):
    """every_n=1 should checkpoint every layer (default)."""
    expert = _make_expert(monkeypatch, num_layers=4)
    expert.enable_gradient_checkpointing(every_n=1)

    inputs = _make_expert_inputs(expert)
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
