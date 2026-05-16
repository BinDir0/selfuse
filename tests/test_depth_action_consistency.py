import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.pipeline.depth_action_consistency import analyze_depth_action_consistency


def _encode_npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _add_member(tar_writer, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar_writer.addfile(info, io.BytesIO(payload))


def _lowdim_with_projected_hands(z: float) -> np.ndarray:
    lowdim = np.zeros((116,), dtype=np.float32)
    hand_points = np.tile(np.array([0.0, 0.0, z], dtype=np.float32), (12, 1))
    lowdim[0:6] = hand_points[:2].reshape(-1)
    lowdim[18:48] = hand_points[2:].reshape(-1)
    lowdim[96:112] = np.eye(4, dtype=np.float32).reshape(-1)
    lowdim[112:116] = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return lowdim


class DepthActionConsistencyTests(unittest.TestCase):
    def test_depth_action_consistency_reports_error_without_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_path = root / "shard-000000.tar"
            lowdim = _lowdim_with_projected_hands(1.0)
            depth_ok = np.full((2, 2), 1000, dtype=np.uint16)
            depth_bad = np.full((2, 2), 2000, dtype=np.uint16)

            with tarfile.open(shard_path, "w") as tar_writer:
                _add_member(tar_writer, "clip_f000000.image.jpg", b"image")
                _add_member(tar_writer, "clip_f000000.lowdim.npy", _encode_npy(lowdim))
                _add_member(tar_writer, "clip_f000000.depth.npy", _encode_npy(depth_ok))
                _add_member(tar_writer, "clip_f000000.meta.json", b"{}")

                _add_member(tar_writer, "clip_f000001.image.jpg", b"image")
                _add_member(tar_writer, "clip_f000001.lowdim.npy", _encode_npy(lowdim))
                _add_member(tar_writer, "clip_f000001.depth.npy", _encode_npy(depth_bad))
                _add_member(tar_writer, "clip_f000001.meta.json", b"{}")

            report = analyze_depth_action_consistency(dataset_dir=str(root))

            self.assertEqual(report["samples_total"], 2)
            self.assertEqual(report["samples_checked"], 2)
            self.assertEqual(report["points_compared"], 24)
            self.assertEqual(report["decode_failures"], 0)
            self.assertAlmostEqual(report["abs_error_m"]["mean"], 0.5, places=5)
            self.assertEqual(len(report["examples"]), 2)


if __name__ == "__main__":
    unittest.main()
