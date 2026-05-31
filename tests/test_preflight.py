"""Unit tests for lib.pipeline.preflight.

Runs without torch / GPU: the GPU check degrades to a reported problem when
torch is unavailable, which is exercised here.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline import preflight, workspace


class ReportTests(unittest.TestCase):
    def test_empty_report_is_ok(self):
        r = preflight.PreflightReport()
        self.assertTrue(r.ok)
        self.assertIn("passed", r.render())

    def test_add_makes_not_ok_and_renders_all(self):
        r = preflight.PreflightReport()
        r.add("weights", "missing X", "download it")
        r.add("inputs", "no video")
        self.assertFalse(r.ok)
        rendered = r.render()
        self.assertIn("missing X", rendered)
        self.assertIn("download it", rendered)
        self.assertIn("no video", rendered)


class WeightsTests(unittest.TestCase):
    def test_missing_and_empty_and_present(self):
        with tempfile.TemporaryDirectory() as d:
            present = Path(d) / "good.pt"
            present.write_bytes(b"x")
            empty = Path(d) / "empty.pt"
            empty.touch()
            r = preflight.PreflightReport()
            preflight.check_weights(
                r,
                {
                    "present": present,
                    "empty": empty,
                    "missing": Path(d) / "nope.pt",
                    "skip": None,
                },
            )
            details = r.render()
            self.assertNotIn("present", details)  # present file not flagged
            self.assertIn("empty", details)
            self.assertIn("missing", details)
            # exactly two problems (empty + missing)
            self.assertEqual(len(r.problems), 2)


class ManoTests(unittest.TestCase):
    def test_missing_mano(self):
        with tempfile.TemporaryDirectory() as d:
            r = preflight.PreflightReport()
            preflight.check_mano(r, Path(d))
            self.assertEqual(len(r.problems), 2)

    def test_present_mano(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            right = root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"
            left = root / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl"
            right.parent.mkdir(parents=True)
            left.parent.mkdir(parents=True)
            right.touch()
            left.touch()
            r = preflight.PreflightReport()
            preflight.check_mano(r, root)
            self.assertTrue(r.ok)


class InputsTests(unittest.TestCase):
    def test_missing_input(self):
        r = preflight.PreflightReport()
        preflight.check_inputs(r, ["/definitely/not/here.mp4"])
        self.assertFalse(r.ok)

    def test_sample_limits_checks(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "a.mp4"
            good.touch()
            r = preflight.PreflightReport()
            # second (missing) path is beyond the sample of 1 -> not checked
            preflight.check_inputs(r, [str(good), "/nope.mp4"], sample=1)
            self.assertTrue(r.ok)


class TmpRootTests(unittest.TestCase):
    def _clear_env(self):
        for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR):
            os.environ.pop(k, None)

    def test_unset_reports_problem(self):
        saved = {k: os.environ.get(k) for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR)}
        try:
            self._clear_env()
            r = preflight.PreflightReport()
            preflight.check_tmp_root(r, args=None)
            self.assertFalse(r.ok)
            self.assertEqual(r.problems[0].category, "tmp_root")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_set_with_enough_disk_ok(self):
        saved = {k: os.environ.get(k) for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR)}
        try:
            self._clear_env()
            with tempfile.TemporaryDirectory() as d:
                os.environ[workspace.ENV_BATCH_TMPDIR] = d
                r = preflight.PreflightReport()
                preflight.check_tmp_root(r, args=None, min_free_gb=0.0)
                self.assertTrue(r.ok)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    def test_low_disk_reports_problem(self):
        saved = {k: os.environ.get(k) for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR)}
        try:
            self._clear_env()
            with tempfile.TemporaryDirectory() as d:
                os.environ[workspace.ENV_BATCH_TMPDIR] = d
                r = preflight.PreflightReport()
                preflight.check_tmp_root(r, args=None, min_free_gb=10 ** 9)  # absurd requirement
                self.assertFalse(r.ok)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)


class RuntimeTests(unittest.TestCase):
    def test_valid_interpreter_ok(self):
        r = preflight.PreflightReport()
        preflight.check_runtimes(r, {"hawor_python": sys.executable})
        self.assertTrue(r.ok)

    def test_missing_interpreter(self):
        r = preflight.PreflightReport()
        preflight.check_runtimes(r, {"slam_python": "/no/such/python"})
        self.assertFalse(r.ok)


class GpuParseTests(unittest.TestCase):
    def test_parse_indices(self):
        self.assertEqual(preflight._parse_gpu_indices("0,1,2"), [0, 1, 2])
        self.assertEqual(preflight._parse_gpu_indices("0, -1, cpu"), [0])
        self.assertEqual(preflight._parse_gpu_indices([0, 3]), [0, 3])
        self.assertEqual(preflight._parse_gpu_indices(None), [])


class GpuCheckTests(unittest.TestCase):
    """check_gpu must probe CUDA in a subprocess (never init CUDA in-process, which
    would break forked workers) and report missing/insufficient devices."""

    def test_probe_failure_reports_problem(self):
        from unittest import mock
        r = preflight.PreflightReport()
        with mock.patch.object(preflight, "_query_cuda", return_value=None):
            preflight.check_gpu(r, "0")
        self.assertFalse(r.ok)
        self.assertEqual(r.problems[0].category, "gpu")

    def test_cuda_unavailable_reports_problem(self):
        from unittest import mock
        r = preflight.PreflightReport()
        with mock.patch.object(preflight, "_query_cuda", return_value=(False, 0)):
            preflight.check_gpu(r, "0")
        self.assertFalse(r.ok)

    def test_enough_devices_ok(self):
        from unittest import mock
        r = preflight.PreflightReport()
        with mock.patch.object(preflight, "_query_cuda", return_value=(True, 4)):
            preflight.check_gpu(r, "0,1,3")
        self.assertTrue(r.ok, r.render())

    def test_requested_index_out_of_range(self):
        from unittest import mock
        r = preflight.PreflightReport()
        with mock.patch.object(preflight, "_query_cuda", return_value=(True, 2)):
            preflight.check_gpu(r, "5")
        self.assertFalse(r.ok)
        self.assertIn("index 5", r.render())


class GatingTests(unittest.TestCase):
    def test_no_gpu_or_tmp_stage_skips_those_checks(self):
        # 'build'/'validate' are not GPU/tmp stages: report should be ok even with
        # no GPU available and no tmp root configured.
        saved = {k: os.environ.get(k) for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR)}
        try:
            for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR):
                os.environ.pop(k, None)
            r = preflight.run_preflight(stages=["build", "validate"], gpus="0")
            self.assertTrue(r.ok, r.render())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_slam_stage_requires_tmp(self):
        saved = {k: os.environ.get(k) for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR)}
        try:
            for k in (workspace.ENV_STAGE3_TMP_ROOT, workspace.ENV_BATCH_TMPDIR):
                os.environ.pop(k, None)
            r = preflight.run_preflight(stages=["slam"], gpus=None)
            cats = {p.category for p in r.problems}
            self.assertIn("tmp_root", cats)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class CollectBatchWeightsTests(unittest.TestCase):
    def test_collects_per_stage(self):
        import argparse

        args = argparse.Namespace(
            stages="detect_track,motion,infiller",
            checkpoint="/w/hawor.ckpt",
            infiller_weight="/w/infiller.pt",
        )
        weights = preflight.collect_batch_weights(Path("/proj"), args)
        self.assertIn("detector", weights)
        self.assertIn("hawor checkpoint", weights)
        self.assertIn("infiller weight", weights)

    def test_build_only_no_weights(self):
        import argparse

        args = argparse.Namespace(stages="build", checkpoint="/w/hawor.ckpt", infiller_weight="/w/infiller.pt")
        weights = preflight.collect_batch_weights(Path("/proj"), args)
        self.assertEqual(weights, {})


if __name__ == "__main__":
    unittest.main()
