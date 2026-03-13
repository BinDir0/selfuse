"""Regression tests for MEM-style SigLIP video encoding."""

import sys
import traceback
from types import SimpleNamespace

import torch

from src.model.vlm.paligemma.siglip import SiglipVisionTransformer


def _make_config(use_mem: bool = True):
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        attention_dropout=0.0,
        intermediate_size=64,
        layer_norm_eps=1e-6,
        image_size=16,
        patch_size=4,
        num_channels=3,
        num_hidden_layers=6,
        use_mem=use_mem,
        temporal_max_frames=18,
        temporal_interval=4,
        lora={},
    )


def _make_model(use_mem: bool = True) -> SiglipVisionTransformer:
    model = SiglipVisionTransformer(_make_config(use_mem=use_mem))
    model.eval()
    return model


def test_use_mem_single_frame_matches_spatial_path():
    model = _make_model(use_mem=True)
    torch.manual_seed(0)
    pixel_values = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        expected = model.encode_spatial(pixel_values)
        actual = model(pixel_values[:, None, ...])

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_use_mem_multiframe_keeps_single_frame_token_shape():
    model = _make_model(use_mem=True)
    torch.manual_seed(1)
    pixel_values = torch.randn(2, 6, 3, 16, 16)

    with torch.no_grad():
        encoded = model(pixel_values)

    expected_num_patches = (16 // 4) ** 2
    assert encoded.shape == (2, expected_num_patches, 32)


def test_no_mem_multiframe_returns_all_frame_tokens():
    model = _make_model(use_mem=False)
    torch.manual_seed(2)
    pixel_values = torch.randn(2, 3, 3, 16, 16)

    with torch.no_grad():
        encoded = model(pixel_values)

    expected_num_patches = (16 // 4) ** 2
    assert encoded.shape == (2, 3 * expected_num_patches, 32)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
