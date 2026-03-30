#!/usr/bin/env python3
"""将单个 Zarr 数据集导出为 WebDataset 兼容的 tar（单 sample，字段为 .npy）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bundle_io import export_zarr_bundle_to_tar
from episode_source import load_episode_bundle


def main() -> None:
    p = argparse.ArgumentParser(
        description="读取 teleop Zarr，写出可被 webdataset.WebDataset 读取的 tar。"
    )
    p.add_argument("--zarr", required=True, help="输入 zarr 根路径")
    p.add_argument("--out", required=True, type=Path, help="输出 .tar 路径")
    p.add_argument(
        "--sample-key",
        default="0000000000",
        help="tar 内 sample 前缀（默认与 bundle_io.SAMPLE_KEY_DEFAULT 一致）",
    )
    args = p.parse_args()
    data, _ = load_episode_bundle(args.zarr, data_format="zarr")
    export_zarr_bundle_to_tar(data, args.out, sample_key=args.sample_key)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
