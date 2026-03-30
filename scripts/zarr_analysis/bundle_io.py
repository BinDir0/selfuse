"""
将 Zarr 分析用的「整包 numpy 字典」与 WebDataset 习惯的 tar 互转。

约定：每个数据集一个 tar，内含单条 sample，成员名为 ``{sample_key}.{field}.npy``，
与 webdataset 常见命名一致，可直接 ``WebDataset("dir/shard.tar")`` 迭代（取第一条即可）。
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

SAMPLE_KEY_DEFAULT = "0000000000"


def export_zarr_bundle_to_tar(
    data: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    sample_key: str = SAMPLE_KEY_DEFAULT,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as tar:
        for name, arr in data.items():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=True)
            raw = buf.getvalue()
            arcname = f"{sample_key}.{name}.npy"
            info = tarfile.TarInfo(name=arcname)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


def load_bundle_from_tar(tar_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    tar_path = Path(tar_path)
    out: Dict[str, np.ndarray] = {}
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.endswith(".npy"):
                continue
            parts = m.name.split(".")
            if len(parts) < 3:
                continue
            field = parts[-2]
            f = tar.extractfile(m)
            if f is None:
                continue
            out[field] = np.load(f, allow_pickle=True)
    return out


def load_bundle_from_webdataset(uri: str) -> Dict[str, np.ndarray]:
    import webdataset as wds

    ds = wds.WebDataset(uri)
    for sample in ds:
        return _webdataset_sample_to_bundle(sample)
    raise ValueError(f"empty webdataset: {uri}")


def _webdataset_sample_to_bundle(sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in sample.items():
        if k.startswith("__"):
            continue
        if k.endswith(".npy"):
            name = k[: -len(".npy")]
            if isinstance(v, np.ndarray):
                out[name] = v
            elif isinstance(v, (bytes, memoryview)):
                out[name] = np.load(io.BytesIO(bytes(v)), allow_pickle=True)
        elif isinstance(v, np.ndarray):
            out[k] = v
    return out
