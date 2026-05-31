"""Unit tests for lib.pipeline.cleanup (pure filesystem; runs without torch)."""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline import cleanup


def _build_seq(root: Path) -> Path:
    """Create a realistic post-run seq_folder tree."""
    seq = root / "clip"
    slam = seq / "SLAM"
    slam.mkdir(parents=True)
    tracks = seq / "tracks_0_100"
    tracks.mkdir(parents=True)
    cam = seq / "cam_space"
    cam.mkdir(parents=True)
    frames = seq / "extracted_images"
    frames.mkdir(parents=True)

    # final artifacts
    (seq / "world_space_res.pth").write_bytes(b"final")
    (seq / "est_focal.txt").write_text("600")
    (slam / "hawor_slam_w_scale_0_100.npz").write_bytes(b"slam")
    (slam / "dense_depth_any4d_0_100.npz").write_bytes(b"depth" * 100)

    # heavy redundant slam caches
    (slam / "dpvo_raw_0_100.npz").write_bytes(b"x" * 1000)
    (slam / "any4d_depth_dpvo_0_100.npz").write_bytes(b"x" * 1000)
    (slam / "any4d_stitch_cf_0_100.npz").write_bytes(b"x" * 100)
    (slam / "hand_anchor_k_0_100.npz").write_bytes(b"x" * 100)
    (slam / "dense_depth_any4d_keyframes_0_100.npz").write_bytes(b"x" * 500)

    # intermediates
    (tracks / "model_masks.npy").write_bytes(b"m" * 1000)
    (tracks / "model_tracks.npy").write_bytes(b"t" * 100)
    (tracks / "frame_chunks_all.npy").write_bytes(b"c" * 100)
    (cam / "0.json").write_text("{}")
    (seq / "cam_space_cache.joblib").write_bytes(b"j" * 100)
    (seq / ".track_range").write_text("0,100")
    (seq / ".infiller.done").write_text("")
    (seq / ".slam.done").write_text("")
    (frames / "000000.jpg").write_bytes(b"img")
    return seq


class CleanupLevelTests(unittest.TestCase):
    def test_level_all_keeps_everything(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            r = cleanup.cleanup_seq_folder(seq, level="all")
            self.assertEqual(r.removed, [])
            self.assertTrue((seq / "tracks_0_100").exists())

    def test_level_slam_removes_heavy_keeps_intermediates_and_depth(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            slam = seq / "SLAM"
            cleanup.cleanup_seq_folder(seq, level="slam")
            # heavy gone
            self.assertFalse((slam / "dpvo_raw_0_100.npz").exists())
            self.assertFalse((slam / "any4d_depth_dpvo_0_100.npz").exists())
            self.assertFalse((slam / "any4d_stitch_cf_0_100.npz").exists())
            self.assertFalse((slam / "hand_anchor_k_0_100.npz").exists())
            # depth + final kept
            self.assertTrue((slam / "dense_depth_any4d_0_100.npz").exists())
            self.assertTrue((slam / "hawor_slam_w_scale_0_100.npz").exists())
            self.assertTrue((seq / "world_space_res.pth").exists())
            # intermediates kept at 'slam'
            self.assertTrue((seq / "tracks_0_100").exists())
            self.assertTrue((seq / ".infiller.done").exists())

    def test_level_none_keeps_only_final_and_depth(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            slam = seq / "SLAM"
            cleanup.cleanup_seq_folder(seq, level="none")
            # final artifacts preserved
            self.assertTrue((seq / "world_space_res.pth").exists())
            self.assertTrue((seq / "est_focal.txt").exists())
            self.assertTrue((slam / "hawor_slam_w_scale_0_100.npz").exists())
            # depth preserved (keep_depth default True)
            self.assertTrue((slam / "dense_depth_any4d_0_100.npz").exists())
            # everything else gone
            self.assertFalse((seq / "tracks_0_100").exists())
            self.assertFalse((seq / "cam_space").exists())
            self.assertFalse((seq / "cam_space_cache.joblib").exists())
            self.assertFalse((seq / ".track_range").exists())
            self.assertFalse((seq / ".infiller.done").exists())
            self.assertFalse((seq / ".slam.done").exists())
            # legacy keyframe depth cache removed even though depth kept
            self.assertFalse((slam / "dense_depth_any4d_keyframes_0_100.npz").exists())
            # frames kept unless remove_frames
            self.assertTrue((seq / "extracted_images").exists())

    def test_remove_frames_and_drop_depth(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            slam = seq / "SLAM"
            cleanup.cleanup_seq_folder(seq, level="none", remove_frames=True, keep_depth=False)
            self.assertFalse((seq / "extracted_images").exists())
            self.assertFalse((slam / "dense_depth_any4d_0_100.npz").exists())
            # final still preserved
            self.assertTrue((seq / "world_space_res.pth").exists())

    def test_freed_bytes_positive(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            r = cleanup.cleanup_seq_folder(seq, level="none")
            self.assertGreater(r.freed_bytes, 0)

    def test_dry_run_removes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            r = cleanup.cleanup_seq_folder(seq, level="none", dry_run=True)
            self.assertEqual(r.removed, [])
            self.assertTrue((seq / "tracks_0_100").exists())
            self.assertGreater(r.freed_bytes, 0)  # would-free accounted

    def test_invalid_level_raises(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            with self.assertRaises(ValueError):
                cleanup.cleanup_seq_folder(seq, level="bogus")

    def test_missing_seq_folder_is_noop(self):
        r = cleanup.cleanup_seq_folder("/no/such/seq", level="none")
        self.assertEqual(r.removed, [])


class ConsolidatedResultTests(unittest.TestCase):
    """When result.npz exists, the legacy pose file + separate depth npz are redundant."""

    def _add_result_npz(self, seq: Path):
        import numpy as np
        from lib.pipeline import result_io

        result_io.save_result(
            seq,
            pred_trans=np.zeros((2, 3, 3)), pred_rot=np.zeros((2, 3, 3)),
            pred_hand_pose=np.zeros((2, 3, 45)), pred_betas=np.zeros((2, 3, 10)),
            pred_valid=np.ones((2, 3), dtype=bool),
            depth_frame_indices=np.arange(3), depths_uint16=np.zeros((3, 4, 5), dtype="uint16"),
        )

    def test_none_with_result_removes_legacy_pose_and_depth(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            self._add_result_npz(seq)
            slam = seq / "SLAM"
            cleanup.cleanup_seq_folder(seq, level="none")
            # consolidated -> these are redundant and removed
            self.assertFalse((seq / "world_space_res.pth").exists())
            self.assertFalse((slam / "dense_depth_any4d_0_100.npz").exists())
            # result.npz preserved
            self.assertTrue((seq / "result.npz").exists())
            # slam scale still preserved
            self.assertTrue((slam / "hawor_slam_w_scale_0_100.npz").exists())


class ResumeAfterCleanupTests(unittest.TestCase):
    def test_is_stage_complete_short_circuits_on_final_artifact(self):
        from lib.pipeline.stage_api import is_stage_complete

        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            seq.mkdir()
            # No markers, no tracks -> every stage would normally be incomplete.
            self.assertFalse(is_stage_complete("detect_track", seq, fast_check=True))
            # Drop the consolidated final product; now all stages report complete.
            (seq / "result.npz").write_bytes(b"x")  # presence is enough for the check
            for stage in ("detect_track", "motion", "slam", "infiller"):
                self.assertTrue(is_stage_complete(stage, seq, fast_check=True), stage)


class Stage3CacheTests(unittest.TestCase):
    def test_stage3_cache_removed_when_tmp_root_given(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as tmp:
            seq = _build_seq(Path(d))
            from lib.pipeline.workspace import stage3_frame_cache_dir

            cache_dir = Path(stage3_frame_cache_dir(tmp, str(seq), 0, 100))
            cache_dir.mkdir(parents=True)
            (cache_dir / "000000.png").write_bytes(b"frame")
            cleanup.cleanup_seq_folder(seq, level="none", tmp_root=tmp, start_idx=0, end_idx=100)
            self.assertFalse(cache_dir.exists())


class FinalArtifactTests(unittest.TestCase):
    def test_final_artifact_exists(self):
        with tempfile.TemporaryDirectory() as d:
            seq = _build_seq(Path(d))
            self.assertTrue(cleanup.final_artifact_exists(seq))
            (seq / "world_space_res.pth").unlink()
            self.assertFalse(cleanup.final_artifact_exists(seq))


if __name__ == "__main__":
    unittest.main()
