"""Tests for the consolidated result.npz I/O and its legacy fallback."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline import result_io
from lib.pipeline.depth_artifacts import load_export_depths


def _fake_poses():
    # (2 hands, T frames, ...) shapes; values don't matter for I/O round-trips.
    t = 3
    return (
        np.random.rand(2, t, 3).astype(np.float32),   # pred_trans
        np.random.rand(2, t, 3).astype(np.float32),   # pred_rot
        np.random.rand(2, t, 15, 3).astype(np.float32),  # pred_hand_pose
        np.random.rand(2, t, 10).astype(np.float32),  # pred_betas
        np.ones((2, t), dtype=bool),                   # pred_valid
    )


def _write_legacy(seq: Path, start=0, end=3):
    import joblib

    seq.mkdir(parents=True, exist_ok=True)
    poses = _fake_poses()
    joblib.dump(list(poses), seq / "world_space_res.pth")
    # depth npz in slam dir, slam.py format
    slam = seq / "SLAM"
    slam.mkdir(parents=True, exist_ok=True)
    depths = (np.random.rand(end - start, 4, 5) * 1000).astype(np.uint16)
    np.savez_compressed(
        slam / f"dense_depth_any4d_{start}_{end}.npz",
        frame_indices=np.arange(start, end, dtype=np.int64),
        depths_uint16=depths,
        height=np.int32(4),
        width=np.int32(5),
    )
    return poses, depths


class SaveLoadTests(unittest.TestCase):
    def test_save_and_load_pose_arrays_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            seq.mkdir()
            poses = _fake_poses()
            result_io.save_result(
                seq,
                pred_trans=poses[0], pred_rot=poses[1], pred_hand_pose=poses[2],
                pred_betas=poses[3], pred_valid=poses[4],
            )
            self.assertTrue(result_io.result_exists(seq))
            loaded = result_io.load_pose_arrays(seq)
            np.testing.assert_allclose(loaded[0], poses[0])
            np.testing.assert_array_equal(loaded[4], poses[4])

    def test_load_pose_arrays_prefers_result_over_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            _write_legacy(seq)
            # consolidate -> result.npz exists
            result_io.consolidate_result(seq, start_idx=0, end_idx=3)
            loaded = result_io.load_pose_arrays(seq / "world_space_res.pth")
            self.assertEqual(loaded[0].shape, (2, 3, 3))

    def test_load_pose_arrays_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            poses, _ = _write_legacy(seq)
            # no result.npz -> falls back to world_space_res.pth
            loaded = result_io.load_pose_arrays(seq / "world_space_res.pth")
            np.testing.assert_allclose(np.asarray(loaded[0]), poses[0])


class ConsolidateTests(unittest.TestCase):
    def test_consolidate_embeds_depth(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            _, depths = _write_legacy(seq, 0, 3)
            out = result_io.consolidate_result(seq, start_idx=0, end_idx=3)
            self.assertIsNotNone(out)
            depth = result_io.load_result_depth(seq)
            self.assertIsNotNone(depth)
            frame_indices, depths_uint16 = depth
            np.testing.assert_array_equal(frame_indices, np.arange(3))
            np.testing.assert_array_equal(depths_uint16, depths)

    def test_consolidate_without_depth_writes_poses_only(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            seq.mkdir()
            import joblib

            joblib.dump(list(_fake_poses()), seq / "world_space_res.pth")
            out = result_io.consolidate_result(seq, start_idx=0, end_idx=3)
            self.assertIsNotNone(out)
            self.assertIsNone(result_io.load_result_depth(seq))

    def test_consolidate_missing_pose_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            seq.mkdir()
            self.assertIsNone(result_io.consolidate_result(seq, start_idx=0, end_idx=3))


class DepthLoaderIntegrationTests(unittest.TestCase):
    def test_load_export_depths_reads_result_npz(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            _, depths = _write_legacy(seq, 0, 3)
            result_io.consolidate_result(seq, start_idx=0, end_idx=3)
            # Remove the legacy dense depth so only result.npz can serve depth.
            (seq / "SLAM" / "dense_depth_any4d_0_3.npz").unlink()
            (seq / ".track_range").write_text("0,3")
            out = load_export_depths(seq, expected_frame_count=3)
            self.assertEqual(out.shape, (3, 4, 5))
            # uint16 mm -> meters
            np.testing.assert_allclose(out, depths.astype(np.float32) * 1e-3, rtol=1e-5)


class FinalArtifactTests(unittest.TestCase):
    def test_final_artifact_exists_result_or_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            seq.mkdir()
            self.assertFalse(result_io.final_artifact_exists(seq))
            (seq / "world_space_res.pth").write_bytes(b"x")
            self.assertTrue(result_io.final_artifact_exists(seq))
            (seq / "world_space_res.pth").unlink()
            result_io.save_result(
                seq, pred_trans=np.zeros((2, 1, 3)), pred_rot=np.zeros((2, 1, 3)),
                pred_hand_pose=np.zeros((2, 1, 15, 3)), pred_betas=np.zeros((2, 1, 10)),
                pred_valid=np.ones((2, 1), dtype=bool),
            )
            self.assertTrue(result_io.final_artifact_exists(seq))


if __name__ == "__main__":
    unittest.main()
