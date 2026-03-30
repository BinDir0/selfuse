"""

用法::

    from episode_source import load_episode_bundle

    d = load_episode_bundle("/path/to/store", data_format="auto")
    d = load_episode_bundle("/path/to/exported.tar", data_format="auto")
    d = load_episode_bundle("pipe:curl -Ls ...", data_format="webdataset")
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

from bundle_io import load_bundle_from_tar, load_bundle_from_webdataset
from zarr_arrays import load_episode_data

DataFormat = Literal["auto", "zarr", "webdataset"]

_ALIASES: Dict[str, DataFormat] = {
    "auto": "auto",
    "zarr": "zarr",
    "webdataset": "webdataset",
    "wds": "webdataset",
    "wd": "webdataset",
    "wds_tar": "webdataset",
    "wds_url": "webdataset",
}


def normalize_data_format(fmt: str) -> DataFormat:
    key = fmt.strip().lower().replace("-", "_")
    out = _ALIASES.get(key)
    if out is None:
        raise ValueError(
            f"未知 data_format: {fmt!r}，请使用 auto | zarr | webdataset（或别名 wds、wds-tar、wds-url）"
        )
    return out


def _local_path_is_tar(p: Path) -> bool:
    try:
        return p.is_file() and tarfile.is_tarfile(p)
    except OSError:
        return False


def _looks_like_zarr_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for name in (".zgroup", ".zmetadata", "zarr.json"):
        if (p / name).exists():
            return True
    return False


def _looks_like_webdataset_uri(s: str) -> bool:
    low = s.strip().lower()
    if low.startswith(("pipe:", "http://", "https://", "s3://", "gs://")):
        return True
    if "{" in s or "*" in s:
        return True
    if low.startswith("file:") and ("*" in s or "{" in s):
        return True
    return False


def load_episode_bundle(
    path_or_uri: str,
    *,
    data_format: str = "auto",
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    读取与 ``zarr_arrays.load_episode_data`` 相同键集的 dict。

    :returns: ``(data_dict, resolved_label)``，``resolved_label`` 为 ``zarr`` / ``webdataset(tar)`` / ``webdataset(uri)``。
    """
    fmt = normalize_data_format(data_format)
    raw = path_or_uri.strip()
    p = Path(raw)

    if fmt == "zarr":
        return load_episode_data(raw), "zarr"

    if fmt == "webdataset":
        if _local_path_is_tar(p):
            return load_bundle_from_tar(p), "webdataset(tar)"
        return load_bundle_from_webdataset(raw), "webdataset(uri)"

    # ----- auto -----
    if _looks_like_webdataset_uri(raw):
        if _local_path_is_tar(p):
            return load_bundle_from_tar(p), "webdataset(tar)"
        return load_bundle_from_webdataset(raw), "webdataset(uri)"

    if _local_path_is_tar(p):
        return load_bundle_from_tar(p), "webdataset(tar)"

    if _looks_like_zarr_dir(p):
        return load_episode_data(raw), "zarr"

    if p.exists() and p.is_dir():
        try:
            return load_episode_data(raw), "zarr"
        except Exception as e:
            raise ValueError(
                f"自动模式：{raw} 是目录但无法作为本脚本期望的 Zarr 布局打开: {e}"
            ) from e

    if p.exists() and p.is_file():
        raise ValueError(
            f"自动模式：本地文件 {raw} 不是可识别的 tar；请改用 --data-format webdataset 指定 URI，"
            "或确认是否为 Zarr（Zarr 通常为目录）。"
        )

    return load_bundle_from_webdataset(raw), "webdataset(uri)"
