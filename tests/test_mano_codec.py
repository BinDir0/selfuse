import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight CI may not ship torch
    torch = None

from lib.pipeline.exporters.mano_codec import rot6d_to_rotmat, run_manopth_mano

try:
    from lib.pipeline.exporters.webdataset_geometry import axis_angle_to_rot6d
except ModuleNotFoundError:  # pragma: no cover - lightweight CI may not ship scipy
    axis_angle_to_rot6d = None

try:
    from lib.utils.geometry import rotmat_to_rot6d
except ModuleNotFoundError:  # pragma: no cover - lightweight CI may not ship torch
    rotmat_to_rot6d = None


class _FakeManoLayer:
    def __call__(self, pose_coeffs, betas):
        del pose_coeffs, betas
        verts_mm = torch.tensor(
            [[[500.0, 0.0, 0.0], [0.0, 250.0, 0.0]]],
            dtype=torch.float32,
        )
        joints_mm = torch.tensor(
            [[[0.0, 0.0, 0.0], [125.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        return verts_mm, joints_mm


class ManoCodecTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is required for MANO replay tests")
    def test_run_manopth_mano_converts_mm_outputs_back_to_meter_space(self):
        verts, joints = run_manopth_mano(
            _FakeManoLayer(),
            wrist_world=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            root_rot_axis_angle=np.zeros((1, 3), dtype=np.float32),
            hand_pose_pca=np.zeros((1, 45), dtype=np.float32),
            betas=np.zeros((1, 10), dtype=np.float32),
            device=torch.device("cpu"),
        )

        np.testing.assert_allclose(joints[0, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(verts[0, 0], np.array([1.5, 2.0, 3.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(verts[0, 1], np.array([1.0, 2.25, 3.0], dtype=np.float32), atol=1e-6)

    @unittest.skipIf(torch is None or axis_angle_to_rot6d is None, "torch and scipy are required for rot6d codec tests")
    def test_axis_angle_to_rot6d_round_trips_with_column_major_layout(self):
        axis_angle = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
        rot6d = axis_angle_to_rot6d(axis_angle).detach().cpu().numpy()
        rotmat = rot6d_to_rotmat(rot6d)[0]

        expected_first_two_columns = rotmat[:, :2].T.reshape(-1)
        np.testing.assert_allclose(rot6d[0], expected_first_two_columns, atol=1e-5)

    @unittest.skipIf(torch is None or rotmat_to_rot6d is None, "torch is required for rot6d codec tests")
    def test_rotmat_to_rot6d_uses_column_major_layout(self):
        rotmat = torch.tensor(
            [[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0], [5.0, 6.0, 9.0]]],
            dtype=torch.float32,
        )
        rot6d = rotmat_to_rot6d(rotmat)
        expected = torch.tensor([[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(rot6d, expected))


if __name__ == "__main__":
    unittest.main()
