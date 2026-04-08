import unittest
from importlib.util import find_spec

import numpy as np

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from lib.pipeline.exporters.manifest_build.resample import resample_episode_features


@unittest.skipUnless(HAS_TORCH, "torch is required for MANO/build resample tests")
class ManifestBuildResampleTests(unittest.TestCase):
    def test_resample_episode_features_preserves_keyframes(self):
        wrist_state = torch.tensor(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 6.0, 40.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        hand_state = torch.tensor([[0.0] * 30, [1.0] * 30], dtype=torch.float32)
        pred_rot = torch.zeros((2, 2, 3), dtype=torch.float32)
        pred_hand_pose = torch.zeros((2, 2, 45), dtype=torch.float32)
        pred_betas = torch.zeros((2, 2, 10), dtype=torch.float32)
        extrinsics = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
        extrinsics[1, :3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        presence = np.array([0, 1], dtype=np.int64)

        (
            wrist_resampled,
            hand_resampled,
            _pred_rot_resampled,
            _pred_hand_pose_resampled,
            _pred_betas_resampled,
            extrinsics_resampled,
            presence_resampled,
        ) = resample_episode_features(
            wrist_state,
            hand_state,
            pred_rot,
            pred_hand_pose,
            pred_betas,
            extrinsics,
            presence,
            7,
            source_fps=5.0,
            target_fps=30.0,
            interpolate_labels=True,
        )

        np.testing.assert_allclose(wrist_resampled[0, :6].numpy(), wrist_state[0, :6].numpy())
        np.testing.assert_allclose(wrist_resampled[-1, :6].numpy(), wrist_state[1, :6].numpy())
        np.testing.assert_allclose(hand_resampled[0].numpy(), hand_state[0].numpy())
        np.testing.assert_allclose(hand_resampled[-1].numpy(), hand_state[1].numpy())
        np.testing.assert_allclose(extrinsics_resampled[0, :3, 3], extrinsics[0, :3, 3])
        np.testing.assert_allclose(extrinsics_resampled[-1, :3, 3], extrinsics[1, :3, 3])
        self.assertEqual(int(presence_resampled[0]), 0)
        self.assertEqual(int(presence_resampled[-1]), 1)


if __name__ == "__main__":
    unittest.main()
