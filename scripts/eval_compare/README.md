# HaWoR fork ↔ upstream comparison (ATE / RPE / PA-MPJPE / WA-MPJPE)

Runs **this fork's dataset-pipeline infer stage** (`run_dataset_pipeline.py --stages
prepare,infer`) and the **whole upstream `/root/HaWoR`** (`demo.py`) on the *same*
egocentric videos drawn from four GT datasets, then scores both against ground truth.
Both write `world_space_res.pth` + a SLAM `.npz`; the runner records each system's output
folder in `runs/<system>/pred_path.txt`, which the evaluator resolves automatically.

- **ATE / ATE-S** — camera trajectory (m). ATE = Sim(3) scale-aligned; ATE-S uses HaWoR's
  own SLAM metric scale (rigid align), so it tests absolute scale.
- **RPE** — relative pose error over a 1-frame delta (translation m, rotation deg).
- **PA-MPJPE** — per-frame Procrustes hand error (mm), reference-frame independent.
- **WA-MPJPE** — single similarity transform per 100-frame segment, world frame (mm).

Definitions follow the HaWoR paper (arXiv:2501.02973). Both systems write identical
artifacts (`world_space_res.pth` + `SLAM/hawor_slam_w_scale_*.npz`), so one evaluator
scores both.

> **Run everything on the PRODUCTION machine.** The office machine has no GPU and no
> dataset files. Only `metrics.py` (pure NumPy) runs anywhere.

## 0. Configure

Edit `config.yaml`: set each dataset's production `data_root`, the conda env names, and
`subset_per_dataset` (default 10). For multi-view datasets we use the **egocentric** view
only (`cam4` for H2O, `ego_cam` for OakInk2, the ego video for TACO).

## 1. Prepare inputs (select subset, extract ego video + aligned GT)

```bash
# smoke-test one sequence first
python -m scripts.eval_compare.prepare_inputs --config scripts/eval_compare/config.yaml --dataset egoverse --limit 1
# then all four
python -m scripts.eval_compare.prepare_inputs --config scripts/eval_compare/config.yaml
```
Produces `<work_dir>/<dataset>/<seq>/{video.mp4, gt.npz}`.

## 2. (one-time) verify joint ordering for raw-joint datasets

H2O and EgoVerse provide raw 3D joints; confirm their order maps to OpenPose. After running
inference on one sequence (step 3), or using any prediction folder:

```bash
python -m scripts.eval_compare.verify_joints --gt <work>/h2o/<seq>/gt.npz \
    --pred <work>/h2o/<seq>/runs/orig/video --hand R
```
If the best permutation isn't `identity`, set `JOINT_PERM` in `gt_adapters/h2o.py` /
`egoverse.py` accordingly and re-run `prepare_inputs` for that dataset. (OakInk2/TACO need
no check — their GT is MANO-FK'd with the repo's own `run_mano`.)

## 3. Run both inference systems

```bash
python -m scripts.eval_compare.run_inference --config scripts/eval_compare/config.yaml
# subset while debugging:
python -m scripts.eval_compare.run_inference --config ... --dataset egoverse --systems orig
```
fork outputs land under `<seq>/runs/fork/stage_outputs/...`; orig under `<seq>/runs/orig/video/`.
Each run dir gets a `pred_path.txt` pointing at the folder with `world_space_res.pth`.

## 4. Evaluate + compare

```bash
# single sequence sanity (use the resolved fork pred folder)
python -m scripts.eval_compare.evaluate --gt <work>/h2o/<seq>/gt.npz \
    --pred "$(cat <work>/h2o/<seq>/runs/fork/pred_path.txt)"

# full table
python -m scripts.eval_compare.compare --config scripts/eval_compare/config.yaml
```
Outputs `<work_dir>/results/comparison.{csv,md}` — per-sequence rows, per-dataset means, and a
fork−orig delta row (≈0 means the fork's infer stage matches the reference).

## Sanity checks
- `python scripts/eval_compare/metrics.py` → self-test passes (runs anywhere).
- Per segment the ordering **PA-MPJPE ≤ WA-MPJPE ≤ W-MPJPE** should hold.
- Cross-check one sequence's ATE/RPE against the vendored
  `thirdparty/DROID-SLAM/thirdparty/tartanair_tools/evaluation/evaluate_ate_scale.py` and `evaluate_rpe.py`.

## Assumptions to confirm on production (load-bearing)
- **30 fps**: `demo.py` re-extracts frames at fps=30; prepared videos are encoded at the
  dataset fps. All four datasets are ~30 fps; if one isn't, re-encode the prepared video to 30.
- **TACO translation unit**: `trans_unit` in config (1.0 = metres). If TACO hand_trans is in
  mm, set `0.001`.
- **OakInk2 ego camera key**: `ego_cam` in config (default `egocentric`).
- **Joint order** for H2O/EgoVerse: confirm via `verify_joints.py` (step 2).
