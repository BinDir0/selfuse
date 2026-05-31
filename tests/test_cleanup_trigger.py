"""Wiring tests: prove the retention cleanup actually fires after the infiller
stage (and only when keep_intermediates != 'all'), without running any stage.

This closes the gap between "cleanup logic is correct" (test_cleanup) and
"cleanup is actually invoked by the pipeline".

stage_api transitively imports torch via frame_sources; stub torch/tqdm/cv2 only
when genuinely absent so this runs on the offline box and uses real deps on prod.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


def _module_absent(name):
    # Check sys.modules first: a prior test may have injected a spec-less fake
    # module, and calling find_spec on that raises.
    return name not in sys.modules and importlib.util.find_spec(name) is None


if _module_absent("tqdm"):
    _t = types.ModuleType("tqdm"); _t.tqdm = lambda it=None, **k: it if it is not None else []
    sys.modules["tqdm"] = _t
if _module_absent("torch"):
    sys.modules["torch"] = types.ModuleType("torch")
if _module_absent("cv2"):
    sys.modules["cv2"] = types.ModuleType("cv2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline import result_io  # noqa: E402
from lib.pipeline import stage_api  # noqa: E402
from lib.pipeline import cleanup as cleanup_mod  # noqa: E402

T = 4


def _write_valid_result(seq: Path):
    seq.mkdir(parents=True, exist_ok=True)
    result_io.save_result(
        seq,
        pred_trans=np.zeros((2, T, 3), np.float32),
        pred_rot=np.zeros((2, T, 3), np.float32),
        pred_hand_pose=np.zeros((2, T, 45), np.float32),
        pred_betas=np.zeros((2, T, 10), np.float32),
        pred_valid=np.ones((2, T), dtype=bool),
    )


def _fake_task(seq: Path):
    return types.SimpleNamespace(
        seq_folder=Path(seq),
        video_path="dummy.mp4",
        build_frame_source=lambda: None,
    )


class CleanupTriggerTests(unittest.TestCase):
    def _run_infiller(self, keep_intermediates):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            _write_valid_result(seq)  # valid final artifact so validation passes
            task = _fake_task(seq)
            config = stage_api.StageExecutionConfig(keep_intermediates=keep_intermediates)

            with mock.patch.object(
                stage_api, "_run_non_detect_stage", return_value=(0, T, {"timing": {}})
            ), mock.patch.object(
                cleanup_mod, "cleanup_seq_folder", wraps=cleanup_mod.cleanup_seq_folder
            ) as clean_spy:
                result = stage_api.run_pipeline_stage(
                    "infiller", task, config, runtime=types.SimpleNamespace(),
                    resume=False, force=True,
                )
            return result, clean_spy

    def test_cleanup_fires_for_none(self):
        result, clean_spy = self._run_infiller("none")
        self.assertEqual(result["status"], "success")
        self.assertTrue(clean_spy.called, "cleanup should fire after infiller when keep_intermediates=none")
        _, kwargs = clean_spy.call_args
        self.assertEqual(kwargs.get("level"), "none")
        self.assertEqual(kwargs.get("start_idx"), 0)
        self.assertEqual(kwargs.get("end_idx"), T)

    def test_cleanup_fires_for_slam(self):
        _result, clean_spy = self._run_infiller("slam")
        self.assertTrue(clean_spy.called)
        _, kwargs = clean_spy.call_args
        self.assertEqual(kwargs.get("level"), "slam")

    def test_cleanup_not_called_for_all(self):
        _result, clean_spy = self._run_infiller("all")
        self.assertFalse(clean_spy.called, "cleanup must NOT fire when keep_intermediates=all")

    def test_cleanup_failure_does_not_break_stage(self):
        # Cleanup must never fail the run.
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            _write_valid_result(seq)
            task = _fake_task(seq)
            config = stage_api.StageExecutionConfig(keep_intermediates="none")
            with mock.patch.object(stage_api, "_run_non_detect_stage", return_value=(0, T, {})), \
                mock.patch.object(cleanup_mod, "cleanup_seq_folder", side_effect=RuntimeError("boom")):
                result = stage_api.run_pipeline_stage(
                    "infiller", task, config, runtime=types.SimpleNamespace(),
                    resume=False, force=True,
                )
            self.assertEqual(result["status"], "success")

    def test_non_infiller_stage_does_not_trigger_cleanup(self):
        # A motion stage finishing should not trigger retention cleanup. Validation
        # is patched out -- this test is only about the cleanup-trigger gate.
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "clip"
            seq.mkdir(parents=True)
            task = _fake_task(seq)
            config = stage_api.StageExecutionConfig(keep_intermediates="none")
            with mock.patch.object(stage_api, "_run_non_detect_stage", return_value=(0, T, {})), \
                mock.patch.object(stage_api, "validate_stage_output"), \
                mock.patch.object(cleanup_mod, "cleanup_seq_folder") as clean_spy:
                stage_api.run_pipeline_stage(
                    "motion", task, config, runtime=types.SimpleNamespace(),
                    resume=False, force=True,
                )
            self.assertFalse(clean_spy.called)


if __name__ == "__main__":
    unittest.main()
