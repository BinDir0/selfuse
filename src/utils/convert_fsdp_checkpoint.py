#!/usr/bin/env python3
"""Convert an FSDP2 sharded checkpoint into a single .pt file.

Typical usage:

    python scripts/convert_fsdp_checkpoint.py \
        --checkpoint /efs-exp/.../step_checkpoints/update_step_130000
"""
import argparse
import pathlib

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    parser = argparse.ArgumentParser(
        description="Convert an FSDP2 sharded checkpoint into a single .pt file."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint root (e.g. update_step_130000) or directly to pytorch_model_fsdp_0.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path. Defaults to <checkpoint_root>/converted_model.pt.",
    )
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")

    fsdp_dir = ckpt / "pytorch_model_fsdp_0" if ckpt.name != "pytorch_model_fsdp_0" else ckpt
    if not fsdp_dir.exists():
        raise FileNotFoundError(f"FSDP shard directory not found: {fsdp_dir}")

    output = pathlib.Path(args.output).expanduser().resolve() if args.output else fsdp_dir.parent / "converted_model.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {fsdp_dir} ...")
    dcp_to_torch_save(str(fsdp_dir), str(output))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
