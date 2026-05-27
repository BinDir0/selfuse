"""Cheapest decisive test for the joint-ordering assumption of raw-joint datasets
(H2O, EgoVerse). Predictions are fixed in 21-joint OpenPose order; if a dataset's
GT joints use a different order, PA-MPJPE is inflated by a constant relabelling.

Given one prepared gt.npz and its prediction seq_folder, this tries a handful of
candidate GT->OpenPose permutations and reports the PA-MPJPE for each. The smallest
error reveals the correct ordering; bake it into the adapter's ``JOINT_PERM``.

Run on the PRODUCTION machine on a single well-tracked sequence:
    python -m scripts.eval_compare.verify_joints --gt <gt.npz> --pred <seq_folder> --hand R
"""

from __future__ import annotations

import argparse

import numpy as np

from scripts.eval_compare import metrics as M
from scripts.eval_compare.gt_adapters.base import GTSequence
from scripts.eval_compare.load_pred import load_prediction

# Candidate permutations applied to GT joints (pred stays OpenPose).
# Blocks: wrist=0, then 5 fingers of 4 joints. OpenPose order = thumb,index,middle,ring,pinky.
WRIST = [0]
FINGERS_OP = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
              "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}


def _perm(order, reverse_within=False):
    out = list(WRIST)
    for fname in order:
        block = FINGERS_OP[fname]
        out += block[::-1] if reverse_within else block
    return out


def candidate_perms() -> dict[str, list[int]]:
    op = ["thumb", "index", "middle", "ring", "pinky"]
    rev = op[::-1]
    return {
        "identity": list(range(21)),
        "openpose": _perm(op),
        "openpose_tip_first": _perm(op, reverse_within=True),
        "reversed_fingers": _perm(rev),
        "reversed_fingers_tip_first": _perm(rev, reverse_within=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--hand", choices=["L", "R"], default="R")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    h = 0 if args.hand == "L" else 1
    gt = GTSequence.load_npz(args.gt)
    pred = load_prediction(args.pred, use_cuda=not args.cpu)
    n = min(gt.joints_world.shape[1], pred.joints_world.shape[1])
    gj, pj = gt.joints_world[h, :n], pred.joints_world[h, :n]
    valid = gt.valid[h, :n].astype(bool) & pred.valid[h, :n].astype(bool)
    print(f"hand={args.hand} valid frames={int(valid.sum())}")

    results = []
    for name, perm in candidate_perms().items():
        pa = M.pa_mpjpe(pj, gj[:, perm, :], valid=valid)
        results.append((pa, name, perm))
    results.sort()
    print("\nPA-MPJPE by candidate GT permutation (lower = correct ordering):")
    for pa, name, perm in results:
        print(f"  {pa:8.2f} mm  {name}")
    print(f"\nbest: {results[0][1]} -> set JOINT_PERM = {results[0][2]}")


if __name__ == "__main__":
    main()
