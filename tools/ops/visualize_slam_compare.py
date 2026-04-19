#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and compare HaWoR SLAM npz outputs")
    parser.add_argument("--old", required=True, help="Old seq folder or hawor_slam_w_scale_*.npz")
    parser.add_argument("--new", required=True, help="New seq folder or hawor_slam_w_scale_*.npz")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures and summary json")
    parser.add_argument("--title", default=None, help="Optional figure title")
    return parser.parse_args()


def resolve_slam_npz(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted((path / "SLAM").glob("hawor_slam_w_scale_*.npz"))
        if not candidates:
            candidates = sorted(path.glob("hawor_slam_w_scale_*.npz"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not resolve hawor_slam_w_scale npz from {path}")


def load_slam(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    tstamp = np.asarray(data["tstamp"], dtype=np.int64).reshape(-1)
    traj = np.asarray(data["traj"], dtype=np.float32)
    scale = float(data["scale"])
    xyz = traj[:, :3].astype(np.float64) * scale
    return {
        "path": str(npz_path),
        "name": npz_path.parent.parent.name if npz_path.parent.name == "SLAM" else npz_path.stem,
        "tstamp": tstamp,
        "traj": traj,
        "scale": scale,
        "xyz": xyz,
    }


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def summarize(run: dict) -> dict:
    xyz = run["xyz"]
    tstamp = run["tstamp"]
    return {
        "path": run["path"],
        "num_keyframes": int(len(tstamp)),
        "first_tstamp": int(tstamp[0]) if len(tstamp) else None,
        "last_tstamp": int(tstamp[-1]) if len(tstamp) else None,
        "scale": float(run["scale"]),
        "path_length_m": path_length(xyz),
        "translation_min": xyz.min(axis=0).tolist() if len(xyz) else None,
        "translation_max": xyz.max(axis=0).tolist() if len(xyz) else None,
    }


def make_aligned_xyz(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    return xyz - xyz[0:1]


def plot_compare(old_run: dict, new_run: dict, out_dir: Path, title: str | None):
    old_xyz = old_run["xyz"]
    new_xyz = new_run["xyz"]
    old_aligned = make_aligned_xyz(old_xyz)
    new_aligned = make_aligned_xyz(new_xyz)

    fig = plt.figure(figsize=(15, 10))
    if title:
        fig.suptitle(title)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(old_xyz[:, 0], old_xyz[:, 2], "-o", ms=3, label=f"old ({len(old_xyz)} kf)")
    ax1.plot(new_xyz[:, 0], new_xyz[:, 2], "-o", ms=3, label=f"new ({len(new_xyz)} kf)")
    ax1.set_title("Top View (raw world XZ)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(old_aligned[:, 0], old_aligned[:, 2], "-o", ms=3, label="old aligned")
    ax2.plot(new_aligned[:, 0], new_aligned[:, 2], "-o", ms=3, label="new aligned")
    ax2.set_title("Top View (start-aligned XZ)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot(old_aligned[:, 0], old_aligned[:, 1], old_aligned[:, 2], "-o", ms=3, label="old aligned")
    ax3.plot(new_aligned[:, 0], new_aligned[:, 1], new_aligned[:, 2], "-o", ms=3, label="new aligned")
    ax3.set_title("3D Trajectory (start-aligned)")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(np.arange(len(old_run["tstamp"])), old_run["tstamp"], "-o", ms=3, label="old")
    ax4.plot(np.arange(len(new_run["tstamp"])), new_run["tstamp"], "-o", ms=3, label="new")
    ax4.set_title("Keyframe timestamps")
    ax4.set_xlabel("keyframe row")
    ax4.set_ylabel("video frame index")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "slam_compare.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_npz = resolve_slam_npz(args.old)
    new_npz = resolve_slam_npz(args.new)
    old_run = load_slam(old_npz)
    new_run = load_slam(new_npz)

    plot_compare(old_run, new_run, out_dir, args.title)

    summary = {
        "old": summarize(old_run),
        "new": summarize(new_run),
        "diff": {
            "keyframe_count_delta": int(len(new_run["tstamp"]) - len(old_run["tstamp"])),
            "scale_delta": float(new_run["scale"] - old_run["scale"]),
            "path_length_delta_m": float(path_length(new_run["xyz"]) - path_length(old_run["xyz"])),
        },
    }
    (out_dir / "slam_compare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {out_dir / 'slam_compare.png'}")
    print(f"Saved {out_dir / 'slam_compare_summary.json'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
