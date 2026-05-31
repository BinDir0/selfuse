"""Wiring test: batch_infer.main aborts (exit 2) when preflight fails, before any
GPU work, and proceeds when preflight passes. The actual checks are unit-tested in
test_preflight; this only verifies the main() glue.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


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

import scripts.batch_infer as batch_infer  # noqa: E402
from lib.pipeline import preflight as preflight_mod  # noqa: E402


def _fake_inputs():
    return types.SimpleNamespace(
        video_paths=["v.mp4"], descriptors=None, input_mode="video_list",
        input_path="list.txt", total_items=1, start_idx=0, end_idx=1,
    )


class PreflightWiringTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("HAWOR_SKIP_PREFLIGHT", None)

    def test_main_returns_2_when_preflight_fails(self):
        failing = preflight_mod.PreflightReport()
        failing.add("weights", "missing checkpoint", "download it")
        with mock.patch.object(batch_infer, "load_batch_inputs", return_value=_fake_inputs()), \
            mock.patch.object(preflight_mod, "run_preflight", return_value=failing):
            rc = batch_infer.main(["--video_list", "list.txt", "--gpus", "0", "--stages", "slam"])
        self.assertEqual(rc, 2)

    def test_skip_env_bypasses_preflight(self):
        # With skip set, preflight must not be consulted at all. Inject a fake
        # lib.pipeline.batch so main()'s local import doesn't pull torch-backed code.
        os.environ["HAWOR_SKIP_PREFLIGHT"] = "1"
        fake_batch = types.ModuleType("lib.pipeline.batch")
        fake_batch.BatchRunConfig = types.SimpleNamespace(from_args=lambda *a, **k: object())
        sched_instance = types.SimpleNamespace(run=lambda: True)
        fake_batch.BatchScheduler = lambda *a, **k: sched_instance
        import tempfile

        try:
            with mock.patch.dict(sys.modules, {"lib.pipeline.batch": fake_batch}), \
                mock.patch.object(batch_infer, "load_batch_inputs", return_value=_fake_inputs()), \
                mock.patch.object(preflight_mod, "run_preflight", side_effect=AssertionError("preflight should be skipped")), \
                tempfile.TemporaryDirectory() as run_dir, \
                mock.patch.object(batch_infer, "_resolve_run_dir", return_value=Path(run_dir)):
                rc = batch_infer.main(["--video_list", "list.txt", "--gpus", "0", "--stages", "slam"])
            self.assertEqual(rc, 0)
        finally:
            os.environ.pop("HAWOR_SKIP_PREFLIGHT", None)


if __name__ == "__main__":
    unittest.main()
