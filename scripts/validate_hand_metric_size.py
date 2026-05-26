#!/usr/bin/env python3
"""Is HaWoR's recovered hand the right ABSOLUTE metric size?

Phase-3 of the metric-alignment plan anchors depth + camera to the HaWoR hand, trusting the
hand's metric size as the absolute reference. That trust is only valid if HaWoR's hand is
actually adult-hand-sized. A subagent flagged the implied size as too large (using a
non-standard "sum of 15 bones"); this script settles it with STANDARD anthropometric measures.

Method (no pipeline inference; just a MANO shape forward = a few matmuls):
  - read per-frame MANO betas from a result dir's cam_space/<hand>/*.json (init_betas),
  - take the per-clip robust median shape per hand (the stable size; see hand-beta wobble finding),
  - forward MANO in a STRAIGHT rest pose (flat_hand_mean + identity pose) to get rest-pose joints,
  - measure standard hand dimensions from the OpenPose-ordered joints and compare to adult norms.

Joint order (lib/models/mano_wrapper.py mano_to_openpose): 0=wrist, 5=index-MCP, 9=middle-MCP,
12=middle-tip, 17=pinky-MCP. So:
  hand length  = |wrist - middle tip| (joint 0->12)   palm length  = |wrist - middle MCP| (0->9)
  hand breadth = |index MCP - pinky MCP| (5->17)       middle finger= |middle MCP - tip| (9->12)

Run from repo root (production env with torch+smplx+the MANO model):
  python scripts/validate_hand_metric_size.py --seq_folder /path/stage_outputs/<clip>
You can also pass several --seq_folder to pool clips, or --use_cuda.
"""
import argparse
import glob
import json
import os

import numpy as np

# adult anthropometric plausible ranges (mm), straight hand. Sources: ANSUR II / Tilley
# "The Measure of Man and Woman" (female-small .. male-large spans).
NORMS = {
    "hand_length_wrist_to_midtip": (165.0, 210.0),   # wrist crease -> middle fingertip
    "palm_length_wrist_to_midMCP": (90.0, 120.0),    # wrist -> middle-finger knuckle
    "hand_breadth_idx_to_pinkyMCP": (72.0, 98.0),    # across the metacarpal knuckles
    "middle_finger_MCP_to_tip": (68.0, 90.0),
}


def load_betas(seq):
    """Return {hand_dir: (T,10) betas} from cam_space init_betas."""
    out = {}
    for hd in sorted(glob.glob(os.path.join(seq, "cam_space", "*"))):
        if not os.path.isdir(hd):
            continue
        bl = []
        for jf in sorted(glob.glob(os.path.join(hd, "*.json"))):
            try:
                b = np.asarray(json.load(open(jf))["init_betas"], np.float64)  # (1,T,10)
                bl.append(b.reshape(-1, b.shape[-1]))
            except Exception:
                continue
        if bl:
            out[os.path.basename(hd)] = np.concatenate(bl, 0)
    return out


def measures_from_joints(J):
    """J: (>=21,3) OpenPose-ordered MANO joints, meters. Return standard dims in mm."""
    d = lambda a, b: float(np.linalg.norm(J[a] - J[b]) * 1000.0)
    return {
        "hand_length_wrist_to_midtip": d(0, 12),
        "palm_length_wrist_to_midMCP": d(0, 9),
        "hand_breadth_idx_to_pinkyMCP": d(5, 17),
        "middle_finger_MCP_to_tip": d(9, 12),
    }


def forward_rest(mano, betas, use_cuda):
    """MANO straight-rest-pose joints for a (10,) betas vector. Returns (J,3) numpy (meters)."""
    import torch
    dev = "cuda" if use_cuda else "cpu"
    b = torch.tensor(betas, dtype=torch.float32, device=dev).reshape(1, -1)
    eye = torch.eye(3, device=dev)
    go = eye.reshape(1, 1, 3, 3)
    hp = eye.reshape(1, 1, 3, 3).repeat(1, 15, 1, 1)
    with torch.inference_mode():
        out = mano(global_orient=go, hand_pose=hp, betas=b, pose2rot=False)
    return out.joints[0].detach().cpu().numpy()


def build_mano(use_cuda):
    """Build the repo's MANO with a STRAIGHT rest pose (flat_hand_mean) for size measurement."""
    from lib.models.mano_wrapper import MANO
    cfg = {
        "data_dir": "_DATA/data/",
        "model_path": "_DATA/data/mano",
        "gender": "neutral",
        "num_hand_joints": 15,
        "create_body_pose": False,
        "flat_hand_mean": True,   # identity pose => straight fingers (true anthropometric length)
    }
    mano = MANO(**cfg)
    if use_cuda:
        mano = mano.cuda()
    mano.eval()
    return mano


def report(tag, measures, ref=None):
    print(f"\n  [{tag}]")
    print(f"    {'measure':<32}{'HaWoR(mm)':>11}{'adult range':>16}{'flag':>8}" +
          ("    vs β=0" if ref else ""))
    for k, (lo, hi) in NORMS.items():
        v = measures[k]
        flag = "OK" if lo <= v <= hi else ("HIGH" if v > hi else "LOW")
        extra = f"    {v/ref[k]:.2f}x" if ref else ""
        print(f"    {k:<32}{v:>11.1f}{f'{lo:.0f}-{hi:.0f}':>16}{flag:>8}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", action="append", required=True, help="result stage_outputs/<clip> (repeatable)")
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()

    mano = build_mano(args.use_cuda)

    # reference: neutral MANO (betas=0) — what size does the template itself imply?
    J0 = forward_rest(mano, np.zeros(10), args.use_cuda)
    m0 = measures_from_joints(J0)
    report("neutral template (betas=0)", m0)

    # pool betas across all given clips/hands
    all_betas = []
    for seq in args.seq_folder:
        per = load_betas(seq)
        if not per:
            print(f"  [warn] no cam_space betas in {seq}")
            continue
        print(f"\nseq={seq}")
        for hand, B in per.items():
            med = np.median(B, 0)
            J = forward_rest(mano, med, args.use_cuda)
            m = measures_from_joints(J)
            # per-frame size spread (use hand_length) to show the wobble we are medianing through
            lens = np.array([measures_from_joints(forward_rest(mano, B[i], args.use_cuda))["hand_length_wrist_to_midtip"]
                             for i in np.linspace(0, len(B) - 1, min(60, len(B))).astype(int)])
            report(f"hand '{hand}'  median shape  (frames={len(B)}, β0_med={med[0]:+.2f}, "
                   f"per-frame hand_len std={lens.std():.1f}mm = {100*lens.std()/lens.mean():.1f}%)", m, ref=m0)
            all_betas.append(B)

    if all_betas:
        med_all = np.median(np.concatenate(all_betas, 0), 0)
        m = measures_from_joints(forward_rest(mano, med_all, args.use_cuda))
        report("POOLED median shape (all clips/hands)", m, ref=m0)
        # verdict
        hl = m["hand_length_wrist_to_midtip"]; lo, hi = NORMS["hand_length_wrist_to_midtip"]
        mid = 0.5 * (lo + hi)
        print("\n=== VERDICT ===")
        print(f"  pooled hand length = {hl:.1f}mm  (adult plausible {lo:.0f}-{hi:.0f}, midpoint {mid:.0f})")
        if hl > hi:
            print(f"  => HaWoR hand is OVERSIZED by ~{100*(hl/mid-1):.0f}% vs typical => its ABSOLUTE metric")
            print("     scale is biased LARGE; anchoring depth+camera to it inherits that bias.")
            print("     Trust-the-hand-as-absolute-anchor is NOT safe as-is (needs a size correction / external scale).")
        elif hl < lo:
            print(f"  => HaWoR hand is UNDERSIZED by ~{100*(1-hl/mid):.0f}% => same caveat, biased small.")
        else:
            print("  => within adult range => hand is a plausible absolute metric anchor (size-wise).")
        print("  NOTE: neutral-template row shows MANO's own baseline size; compare to see if the bias is")
        print("        from the betas (HaWoR regression) or already in the template/units.")


if __name__ == "__main__":
    main()
