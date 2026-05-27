"""Stage 2: run BOTH inference systems on each prepared video.

system = "fork": THIS repo's dataset-pipeline infer stage — the code under test.
    scripts/run_dataset_pipeline.py --config <cfg> --stages prepare,infer
    (cfg = {video: <prepared mp4>, output_root: <seq>/runs/fork}). The pipeline writes
    world_space_res.pth + SLAM/hawor_slam_w_scale_*.npz under output_root/stage_outputs/.

system = "orig": the whole upstream HaWoR at /root/HaWoR.
    demo.py --video_path <mp4> --vis_mode world  -> seq_folder <video_dir>/<video_stem>.

Both end up with world_space_res.pth + SLAM npz in some folder; we resolve that folder by
globbing and record it in runs/<system>/pred_path.txt for the evaluator.

Run on the PRODUCTION machine (GPU + conda envs). Assumes ~30 fps inputs.

    python -m scripts.eval_compare.run_inference --config scripts/eval_compare/config.yaml
    python -m scripts.eval_compare.run_inference --config ... --systems fork
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess

from scripts.eval_compare.common import load_config


def _resolve_pred_folder(search_root: str) -> str | None:
    """Folder containing world_space_res.pth AND a SLAM npz, searched under search_root."""
    for wr in sorted(glob.glob(os.path.join(search_root, "**", "world_space_res.pth"), recursive=True)):
        folder = os.path.dirname(wr)
        if glob.glob(os.path.join(folder, "SLAM", "hawor_slam_w_scale_*.npz")):
            return folder
    return None


def _bash(repo: str, py_cmd: str, env: str, log: str, env_vars: dict | None = None) -> int:
    # cd happens in the outer shell; env vars + conda run wrap the python invocation.
    # env may be a NAME (conda run -n) or a PREFIX PATH (conda run -p, e.g. on a big disk).
    if env:
        flag = "-p" if ("/" in env or os.path.isabs(env)) else "-n"
        conda = f"conda run --no-capture-output {flag} {env} "
    else:
        conda = ""
    exports = "".join(f"{k}={shlex.quote(str(v))} " for k, v in (env_vars or {}).items())
    mkdirs = "".join(
        f"mkdir -p {shlex.quote(str(v))} && " for k, v in (env_vars or {}).items() if "TMP" in k or "DIR" in k
    )
    full = f"cd {repo} && {mkdirs}{exports}{conda}{py_cmd}"
    print(f"  $ {full}")
    with open(log, "w") as f:
        return subprocess.run(["bash", "-lc", full], stdout=f, stderr=subprocess.STDOUT).returncode


def run_fork(seq_dir: str, repo: str, env: str, force: bool, env_vars: dict | None = None) -> str | None:
    run_dir = os.path.join(seq_dir, "runs", "fork")
    os.makedirs(run_dir, exist_ok=True)
    video = os.path.abspath(os.path.join(seq_dir, "video.mp4"))
    if not os.path.exists(video):
        print("  skip fork: no prepared video"); return None

    existing = _resolve_pred_folder(run_dir)
    if existing and not force:
        print("  fork: outputs present, skip"); _write_marker(run_dir, existing); return existing

    cfg_path = os.path.join(run_dir, "pipeline_config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"video: {video}\noutput_root: {os.path.abspath(run_dir)}\n")
    py = f"python scripts/run_dataset_pipeline.py --config {os.path.abspath(cfg_path)} --stages prepare,infer"
    rc = _bash(repo, py, env, os.path.join(run_dir, "run.log"), env_vars=env_vars)
    folder = _resolve_pred_folder(run_dir)
    if rc != 0 or folder is None:
        print(f"  fork: FAILED (rc={rc}); see {run_dir}/run.log"); return None
    _write_marker(run_dir, folder)
    return folder


def run_orig(seq_dir: str, repo: str, env: str, force: bool, env_vars: dict | None = None) -> str | None:
    run_dir = os.path.join(seq_dir, "runs", "orig")
    os.makedirs(run_dir, exist_ok=True)
    src = os.path.join(seq_dir, "video.mp4")
    if not os.path.exists(src):
        print("  skip orig: no prepared video"); return None
    link = os.path.join(run_dir, "video.mp4")
    if not os.path.exists(link):
        os.symlink(os.path.relpath(src, run_dir), link)

    existing = _resolve_pred_folder(run_dir)
    if existing and not force:
        print("  orig: outputs present, skip"); _write_marker(run_dir, existing); return existing

    py = f"python demo.py --video_path {os.path.abspath(link)} --vis_mode world"
    rc = _bash(repo, py, env, os.path.join(run_dir, "run.log"), env_vars=env_vars)
    folder = _resolve_pred_folder(run_dir)
    if rc != 0 or folder is None:
        print(f"  orig: FAILED (rc={rc}); see {run_dir}/run.log"); return None
    _write_marker(run_dir, folder)
    return folder


def _write_marker(run_dir: str, pred_folder: str) -> None:
    with open(os.path.join(run_dir, "pred_path.txt"), "w") as f:
        f.write(os.path.abspath(pred_folder) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--systems", default="fork,orig")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N prepared seqs per dataset (debug)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = cfg["work_dir"]
    envs = cfg["envs"]
    fork_env_name = envs.get("fork", envs.get("hawor", ""))
    orig_env_name = envs.get("orig", envs.get("hawor", ""))
    systems = args.systems.split(",")
    datasets = [args.dataset] if args.dataset else list(cfg["datasets"])

    for name in datasets:
        seq_dirs = [d for d in sorted(glob.glob(os.path.join(work_dir, name, "*"))) if os.path.isdir(d)]
        if args.limit:
            seq_dirs = seq_dirs[: args.limit]
        for seq_dir in seq_dirs:
            print(f"[{name}] {os.path.basename(seq_dir)}")
            if "fork" in systems:
                run_fork(seq_dir, cfg["repos"]["fork"], fork_env_name, args.force, cfg.get("fork_env"))
            if "orig" in systems:
                run_orig(seq_dir, cfg["repos"]["upstream"], orig_env_name, args.force, cfg.get("orig_env"))


if __name__ == "__main__":
    main()
