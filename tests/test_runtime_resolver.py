import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lib.pipeline import runtime_resolver
from lib.pipeline.runtime_resolver import resolve_conda_env_python, resolve_pipeline_runtimes


class _Result:
    def __init__(self, *, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


class RuntimeResolverTests(unittest.TestCase):
    def test_resolve_conda_env_python_from_env_list_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hawor = root / "envs" / "hawor"
            any4d = root / "envs" / "any4d"
            (hawor / "bin").mkdir(parents=True)
            (any4d / "bin").mkdir(parents=True)
            (hawor / "bin" / "python").write_text("", encoding="utf-8")
            (any4d / "bin" / "python").write_text("", encoding="utf-8")
            stdout = json.dumps({"envs": [str(hawor), str(any4d)]})

            def fake_run(cmd, **kwargs):
                self.assertEqual(cmd[:3], ["conda", "env", "list"])
                return _Result(stdout=stdout)

            with mock.patch.object(runtime_resolver, "_conda_like_commands", return_value=["conda"]):
                self.assertEqual(
                    resolve_conda_env_python("any4d", command_runner=fake_run),
                    str(any4d / "bin" / "python"),
                )

    def test_resolve_conda_env_python_fails_fast_when_missing(self):
        def fake_run(cmd, **kwargs):
            return _Result(stdout=json.dumps({"envs": []}))

        with mock.patch.object(runtime_resolver, "_conda_like_commands", return_value=["conda"]):
            with self.assertRaisesRegex(FileNotFoundError, "Required conda env `hawor`"):
                resolve_conda_env_python("hawor", command_runner=fake_run)

    def test_resolve_pipeline_runtimes_prefers_legacy_config_paths(self):
        runtimes = resolve_pipeline_runtimes(
            {
                "hawor_python": "/tmp/hawor/bin/python",
                "slam_python": "/tmp/any4d/bin/python",
            },
            require_hawor=True,
            require_slam=True,
        )
        self.assertEqual(runtimes.hawor_python, "/tmp/hawor/bin/python")
        self.assertEqual(runtimes.slam_python, "/tmp/any4d/bin/python")
        self.assertEqual(runtimes.source, {"hawor": "config", "slam": "config"})


if __name__ == "__main__":
    unittest.main()
