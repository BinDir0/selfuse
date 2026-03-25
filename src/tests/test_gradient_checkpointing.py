from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3VLConfig, Qwen3VLTextConfig

from src.model.action.diffloss import SimpleMLPAdaLN
from src.model.action_expert.qwen3_action_expert import Qwen3ActionExpert
from src.model.vlm.prefix_cache import PrefixKVCache


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
    with patch("src.model.action.diffloss.checkpoint", fake_checkpoint):
        net(x, t, c)
    assert len(train_calls) == 2

    eval_calls = []

    def fake_checkpoint_eval(fn, *args, **kwargs):
        eval_calls.append(True)
        return fn(*args)

    net.eval()
    with patch("src.model.action.diffloss.checkpoint", fake_checkpoint_eval):
        net(x, t, c)
    assert len(eval_calls) == 0


def test_action_expert_checkpoint_runs_only_in_train_mode(monkeypatch):
    base_text_config = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
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

    expert = Qwen3ActionExpert(
        model_name_or_path="dummy",
        time_hidden_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        attn_implementation="eager",
    )
    expert.enable_gradient_checkpointing()

    batch_size = 2
    prefix_len = 3
    action_len = 4
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

    train_calls = []

    def fake_gc(fn, *args, **kwargs):
        train_calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = fake_gc
    expert.train()
    out = expert(
        action_embeds=action_embeds,
        prefix_cache=prefix_cache,
        action_position_ids=action_position_ids,
        time_cond=time_cond,
        action_mask=action_mask,
        mode="flow",
    )
    out.sum().backward()
    assert len(train_calls) == expert.num_layers
    assert action_embeds.grad is not None
    assert time_cond.grad is not None

    eval_calls = []

    def fake_gc_eval(fn, *args, **kwargs):
        eval_calls.append(True)
        return fn(*args)

    expert._gradient_checkpointing_func = fake_gc_eval
    expert.eval()
    expert(
        action_embeds=action_embeds.detach(),
        prefix_cache=prefix_cache,
        action_position_ids=action_position_ids,
        time_cond=time_cond.detach(),
        action_mask=action_mask,
        mode="flow",
    )
    assert len(eval_calls) == 0
