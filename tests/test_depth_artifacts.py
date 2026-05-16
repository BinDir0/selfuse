import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.pipeline.depth_artifacts import load_export_depths


class DepthArtifactTests(unittest.TestCase):
    def test_load_export_depths_uses_slam_dense_any4d_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_folder = Path(tmpdir) / "clip"
            slam_dir = seq_folder / "SLAM"
            slam_dir.mkdir(parents=True)
            (seq_folder / ".track_range").write_text("0,3", encoding="utf-8")

            depths_m = np.asarray(
                [
                    [[1.0, 1.5], [2.0, 2.5]],
                    [[3.0, 3.5], [4.0, 4.5]],
                    [[5.0, 5.5], [6.0, 6.5]],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                slam_dir / "dense_depth_any4d_0_3.npz",
                frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
                depths_uint16=np.round(depths_m * 1000.0).astype(np.uint16),
            )

            loaded = load_export_depths(seq_folder, expected_frame_count=3)

            self.assertEqual(loaded.shape, (3, 2, 2))
            np.testing.assert_allclose(loaded, depths_m, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
