import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from lib.pipeline.exporters.webdataset_features import _load_episode_camera_features
except ModuleNotFoundError:
    _load_episode_camera_features = None


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


if __name__ == "__main__":
    unittest.main()
