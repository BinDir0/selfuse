import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from lib.pipeline.exporters.webdataset_features import _load_episode_camera_features
except ModuleNotFoundError:
    _load_episode_camera_features = None

try:
    from lib.eval_utils.custom_utils import interpolate_slam_cameras_at_video_frames
except ModuleNotFoundError:
    interpolate_slam_cameras_at_video_frames = None


class WebdatasetFeatureCameraTests(unittest.TestCase):
    @unittest.skipIf(_load_episode_camera_features is None, "export camera helpers require project dependencies")
    def test_sparse_slam_traj_uses_timestamp_interpolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_folder = Path(tmpdir)
            slam_dir = seq_folder / "SLAM"
            slam_dir.mkdir(parents=True, exist_ok=True)
            traj = np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            np.savez(
                slam_dir / "hawor_slam_w_scale_0_4.npz",
                tstamp=np.asarray([0, 4], dtype=np.int32),
                traj=traj,
                img_focal=np.float32(500.0),
                img_center=np.asarray([320.0, 240.0], dtype=np.float32),
                scale=np.float32(1.0),
            )

            extrinsics, intrinsic = _load_episode_camera_features(
                {"crop_dir": str(seq_folder), "episode_id": "demo_clip"},
                num_frames=5,
            )

            self.assertEqual(extrinsics.shape, (5, 4, 4))
            self.assertEqual(intrinsic.tolist(), [500.0, 500.0, 320.0, 240.0])
            np.testing.assert_allclose(
                extrinsics[:, 0, 3],
                np.asarray([0.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                extrinsics[:, :3, :3],
                np.tile(np.eye(3, dtype=np.float32)[None, ...], (5, 1, 1)),
                atol=1e-6,
            )

    @unittest.skipIf(_load_episode_camera_features is None, "export camera helpers require project dependencies")
    def test_dense_traj_takes_priority_over_sparse_tstamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_folder = Path(tmpdir)
            slam_dir = seq_folder / "SLAM"
            slam_dir.mkdir(parents=True, exist_ok=True)
            traj = np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            np.savez(
                slam_dir / "hawor_slam_w_scale_0_4.npz",
                tstamp=np.asarray([0, 4], dtype=np.int32),
                disps=np.ones((2, 1, 1), dtype=np.float32),
                traj=traj,
                img_focal=np.float32(500.0),
                img_center=np.asarray([320.0, 240.0], dtype=np.float32),
                scale=np.float32(1.0),
            )

            extrinsics, _intrinsic = _load_episode_camera_features(
                {"crop_dir": str(seq_folder), "episode_id": "demo_clip"},
                num_frames=5,
            )

            np.testing.assert_allclose(
                extrinsics[:, 0, 3],
                np.asarray([0.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32),
                atol=1e-5,
            )

    @unittest.skipIf(
        interpolate_slam_cameras_at_video_frames is None,
        "slam interpolation helpers require project dependencies",
    )
    def test_dense_traj_sparse_tstamp_uses_direct_camera_indexing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slam_path = Path(tmpdir) / "hawor_slam_w_scale_0_4.npz"
            traj = np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            np.savez(
                slam_path,
                tstamp=np.asarray([0, 4], dtype=np.int32),
                disps=np.ones((2, 1, 1), dtype=np.float32),
                traj=traj,
                scale=np.float32(1.0),
            )

            r_c2w, t_c2w = interpolate_slam_cameras_at_video_frames(
                str(slam_path),
                np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            )

            self.assertEqual(tuple(r_c2w.shape), (5, 3, 3))
            self.assertEqual(tuple(t_c2w.shape), (5, 3))
            np.testing.assert_allclose(
                t_c2w[:, 0].cpu().numpy(),
                np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
