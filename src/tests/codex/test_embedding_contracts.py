from __future__ import annotations

import torch

from src.tests.test_unified_vla_scaffold import make_model, make_vla_batch
from src.utils.embedding_analysis import analyze_embedding_distribution


def test_slot_embeddings_are_finite_and_non_degenerate():
    model = make_model(diffloss=None)
    batch = make_vla_batch(batch_size=1)

    slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)

    for key in ["state", "action"]:
        value = slot_embeds[key]
        assert value is not None
        assert torch.isfinite(value).all()
        assert torch.linalg.norm(value, dim=-1).gt(0).any()
        assert value.var().item() > 0


def test_backbone_stream_hidden_states_and_prefix_cache_are_finite():
    model = make_model(diffloss=None)
    batch = make_vla_batch(batch_size=1)

    slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
    backbone_output = model.forward_backbone_stream(batch, slot_embeds)
    prefix_lengths = model.build_prefix_lengths(batch)

    assert torch.isfinite(backbone_output.last_hidden_states).all()
    assert backbone_output.prefix_cache is not None
    assert torch.equal(backbone_output.prefix_cache.mask.sum(dim=1), prefix_lengths)
    for layer in backbone_output.prefix_cache.layers:
        assert torch.isfinite(layer.key).all()
        assert torch.isfinite(layer.value).all()


def test_hidden_state_distribution_summary_has_non_zero_variance():
    model = make_model(diffloss=None)
    batch = make_vla_batch(batch_size=1)

    slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
    backbone_output = model.forward_backbone_stream(batch, slot_embeds)
    hidden_states = backbone_output.last_hidden_states.reshape(-1, backbone_output.last_hidden_states.shape[-1])

    summary = analyze_embedding_distribution(hidden_states, plot=False)
    assert summary["global_stats"]["std"] > 0
    assert summary["per_vector_stats"]["norms"]["mean"] > 0

