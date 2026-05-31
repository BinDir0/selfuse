"""End-to-end (CPU) integration test for the Phase 4 retention + consolidation chain.

Simulates a finished clip on disk exactly as the real stages leave it, then drives
the REAL product code (no fakes for the chain under test):

    consolidate_result  ->  cleanup_seq_folder(level=none)  ->  build-path readback

This exercises, without a GPU:
  * SLAM depth (uint16 mm) -> result.npz consolidation
  * retention cleanup leaving only final artifacts (result.npz + slam scale)
  * the build-path depth loader (load_export_depths) reading depth from result.npz
    AFTER the separate depth npz was cleaned away
  * pose readback via result_io.load_pose_arrays after the legacy .pth was removed
  * resume short-circuit (is_stage_complete) keyed on the final artifact

stage_api transitively imports torch via frame_sources; stub it only if genuinely
absent so this runs here and uses the real module on production.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def _module_absent(name):
    # Check sys.modules first: a prior test may have injected a spec-less fake
    # module, and calling find_spec on that raises.
    return name not in sys.modules and importlib.util.find_spec(name) is None


# Minimal stubs only when the real deps are missing (offline CI box).
if _module_absent("tqdm"):
    _t = types.ModuleType("tqdm")
    _t.tqdm = lambda it=None, **k: it if it is not None else []
    sys.modules["tqdm"] = _t
if _module_absent("torch"):
    _torch = types.ModuleType("torch")
    sys.modules["torch"] = _torch
if _module_absent("cv2"):
    sys.modules["cv2"] = types.ModuleType("cv2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402

from lib.pipeline import result_io  # noqa: E402
from lib.pipeline.cleanup import cleanup_seq_folder  # noqa: E402
from lib.pipeline.depth_artifacts import load_export_depths  # noqa: E402


T = 5  # frames
H, W = 6, 8


def _make_finished_clip(root: Path) -> Path:
    """Create a seq_folder exactly as detect/motion/slam/infiller would leave it."""
    seq = root / "clip0"
    slam = seq / "SLAM"
    tracks = seq / "tracks_0_5"
    cam = seq / "cam_space"
    frames = seq / "extracted_images"
    for d in (slam, tracks, cam, frames):
        d.mkdir(parents=True)

    # infiller output (legacy .pth: list of 5 arrays, 2 hands)
    poses = [
        np.random.rand(2, T, 3).astype(np.float32),       # pred_trans
        np.random.rand(2, T, 3).astype(np.float32),       # pred_rot
        np.random.rand(2, T, 45).astype(np.float32),      # pred_hand_pose
        np.random.rand(2, T, 10).astype(np.float32),      # pred_betas
        np.ones((2, T), dtype=bool),                      # pred_valid
    ]
    joblib.dump(poses, seq / "world_space_res.pth")

    # slam scale result (final, preserved)
    np.savez(slam / "hawor_slam_w_scale_0_5.npz", traj=np.zeros((T, 7), np.float32), scale=np.float32(1.0))

    # dense depth (uint16 mm), all frames -- slam.py format
    depth_m = (np.random.rand(T, H, W).astype(np.float32) * 2.0)  # 0..2 m
    depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    np.savez_compressed(
        slam / "dense_depth_any4d_0_5.npz",
        frame_indices=np.arange(T, dtype=np.int64),
        depths_uint16=depth_mm,
        height=np.int32(H), width=np.int32(W),
    )

    # heavy + intermediate artifacts that should be cleaned
    np.save(tracks / "model_masks.npy", np.zeros((T, H, W), np.uint8))
    np.save(tracks / "model_tracks.npy", np.zeros((T, 4), np.float32))
    joblib.dump({"chunks": []}, tracks / "frame_chunks_all.npy")
    np.savez_compressed(slam / "dpvo_raw_0_5.npz", x=np.zeros(1000))
    np.savez_compressed(slam / "any4d_depth_dpvo_0_5.npz", x=np.zeros(1000))
    (cam / "0.json").write_text("{}")
    (seq / "cam_space_cache.joblib").write_bytes(b"x" * 100)
    (seq / "est_focal.txt").write_text("600")
    (seq / ".track_range").write_text("0,5")
    for stage in ("detect_track", "motion", "slam", "infiller"):
        (seq / f".{stage}.done").write_text("")
    (frames / "000000.jpg").write_bytes(b"img")
    return seq, depth_mm


class Phase4ChainTests(unittest.TestCase):
    def test_consolidate_cleanup_and_readback(self):
        with tempfile.TemporaryDirectory() as d:
            seq, depth_mm = _make_finished_clip(Path(d))

            # 1) consolidate (real product code) -> result.npz with poses + depth
            out = result_io.consolidate_result(seq, start_idx=0, end_idx=5)
            self.assertIsNotNone(out)
            self.assertTrue(result_io.result_exists(seq))

            # 2) retention cleanup (level none) -- removes everything but finals
            report = cleanup_seq_folder(seq, level="none")
            self.assertGreater(report.freed_bytes, 0)
            self.assertEqual(report.errors, [])

            # final artifacts preserved
            self.assertTrue((seq / "result.npz").exists())
            self.assertTrue((seq / "SLAM" / "hawor_slam_w_scale_0_5.npz").exists())
            # redundant / intermediate removed
            self.assertFalse((seq / "world_space_res.pth").exists())
            self.assertFalse((seq / "SLAM" / "dense_depth_any4d_0_5.npz").exists())
            self.assertFalse((seq / "SLAM" / "dpvo_raw_0_5.npz").exists())
            self.assertFalse((seq / "tracks_0_5").exists())
            self.assertFalse((seq / "cam_space").exists())
            self.assertFalse((seq / "cam_space_cache.joblib").exists())
            self.assertFalse((seq / ".infiller.done").exists())
            self.assertFalse((seq / ".track_range").exists())

            # 3) build-path depth loader reads depth from result.npz (post-cleanup)
            depths = load_export_depths(seq, expected_frame_count=T)
            self.assertEqual(depths.shape, (T, H, W))
            np.testing.assert_allclose(depths, depth_mm.astype(np.float32) * 1e-3, rtol=1e-5)

            # 4) pose readback from result.npz (post-cleanup, legacy .pth gone)
            poses = result_io.load_pose_arrays(seq)
            self.assertEqual(poses[0].shape, (2, T, 3))
            self.assertEqual(poses[2].shape, (2, T, 45))

            # 5) the clip is still detectable as complete
            self.assertTrue(result_io.final_artifact_exists(seq))

    def test_resume_short_circuit_after_cleanup(self):
        from lib.pipeline.stage_api import is_stage_complete

        with tempfile.TemporaryDirectory() as d:
            seq, _ = _make_finished_clip(Path(d))
            result_io.consolidate_result(seq, start_idx=0, end_idx=5)
            cleanup_seq_folder(seq, level="none")
            # markers + tracks gone, but result.npz exists -> every stage "complete"
            for stage in ("detect_track", "motion", "slam", "infiller"):
                self.assertTrue(is_stage_complete(stage, seq, fast_check=True), stage)


if __name__ == "__main__":
    unittest.main()
