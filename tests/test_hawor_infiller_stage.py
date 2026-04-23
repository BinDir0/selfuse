import unittest
from unittest import mock

import numpy as np
import torch

from lib.pipeline.stages import hawor_infiller_stage as mod


class HaworInfillerStageTests(unittest.TestCase):
    def test_run_infiller_pass_skips_hand_with_no_observations(self):
        state = mod.InfillerState(
            pred_trans=torch.zeros((2, 5, 3), dtype=torch.float32),
            pred_rot=torch.zeros((2, 5, 3), dtype=torch.float32),
            pred_hand_pose=torch.zeros((2, 5, 45), dtype=torch.float32),
            pred_betas=torch.zeros((2, 5, 10), dtype=torch.float32),
            pred_valid=torch.tensor(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
            num_frames=5,
            max_slam_frames=5,
            cam_space_cache={},
            r_c2w_sla_all=torch.zeros((5, 3, 3), dtype=torch.float32),
            t_c2w_sla_all=torch.zeros((5, 3), dtype=torch.float32),
            slam_path="dummy",
            use_dpvo_infiller=False,
        )

        prepare_calls = []

        def _fake_parse_chunks_hand_frame(frame):
            arr = np.asarray(frame, dtype=np.int64)
            if arr.size == 0:
                return []
            return [arr]

        def _fake_prepare(*args, **kwargs):
            frame_ck = np.asarray(args[0], dtype=np.int64)
            prepare_calls.append(frame_ck.tolist())
            return {
                "filling_net_start": int(frame_ck[0]),
                "filling_net_end": int(frame_ck[-1]) + 1,
                "seq_valid": np.zeros((2, len(frame_ck)), dtype=bool),
                "seq_valid_padding": np.zeros((2, 120), dtype=bool),
                "filling_seq": {
                    "trans": np.zeros((2, len(frame_ck), 3), dtype=np.float32),
                    "rot": np.zeros((2, len(frame_ck), 3), dtype=np.float32),
                    "hand_pose": np.zeros((2, len(frame_ck), 45), dtype=np.float32),
                    "betas": np.zeros((2, len(frame_ck), 10), dtype=np.float32),
                },
                "filling_input": np.zeros((120, 2, 61), dtype=np.float32),
                "transform_w_canon": np.eye(4, dtype=np.float32),
                "t_original": int(len(frame_ck)),
            }

        with mock.patch.object(mod, "parse_chunks_hand_frame", side_effect=_fake_parse_chunks_hand_frame), \
            mock.patch.object(mod, "_prepare_infiller_window", side_effect=_fake_prepare), \
            mock.patch.object(
                mod,
                "_flush_infiller_windows",
                return_value={"batch_size": 1, "forward_time": 0.0, "postprocess_time": 0.0},
            ) as flush_mock:
            total_windows, timing = mod._run_infiller_pass(
                state,
                filling_model=object(),
                src_mask=None,
                device=torch.device("cpu"),
                horizon=120,
                window_batch_size=64,
            )

        self.assertEqual(total_windows, 1)
        self.assertEqual(len(prepare_calls), 1)
        self.assertEqual(prepare_calls[0], [1, 2, 3, 4])
        flush_mock.assert_called_once()
        np.testing.assert_array_equal(state.pred_valid[1], np.zeros((5,), dtype=bool))
        self.assertIn("prepare_windows", timing)


if __name__ == "__main__":
    unittest.main()
