import json
import argparse
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from scripts import filter_and_check_datasets as mod


def _npy_bytes(array):
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


class FilterAndCheckDatasetsProgressTests(unittest.TestCase):
    def test_worker_writes_shard_progress_heartbeats(self):
        lowdim = np.zeros((116,), dtype=np.float32)
        lowdim[96:112] = np.eye(4, dtype=np.float32).reshape(-1)
        lowdim[112:116] = np.asarray([100.0, 100.0, 64.0, 64.0], dtype=np.float32)
        sample = {
            "__key__": "sample-000001",
            "__url__": "/data/shard-000000.tar",
            "meta.json": {"instruction": "pick", "instruction_num": 1, "cameras": ["head"]},
            "lowdim.npy": _npy_bytes(lowdim),
        }

        fake_wds = types.SimpleNamespace(WebDataset=lambda *args, **kwargs: [sample])
        original_wds = sys.modules.get("webdataset")
        sys.modules["webdataset"] = fake_wds
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                progress_path = Path(tmpdir) / "progress.jsonl"
                result = mod.scan_one_wds_shard_worker(
                    {
                        "shard_index": 0,
                        "shard_path": "/data/shard-000000.tar",
                        "kind": "auto",
                        "check_media": False,
                        "check_depth": False,
                        "check_image_quality": False,
                        "target_size": None,
                        "write_good": False,
                        "write_bad": False,
                        "tmp_dir": None,
                        "heartbeat_path": str(progress_path),
                        "heartbeat_interval": 1,
                        "heartbeat_run_id": "test-run",
                    }
                )
                records = [
                    json.loads(line)
                    for line in progress_path.read_text(encoding="utf-8").splitlines()
                ]
        finally:
            if original_wds is None:
                sys.modules.pop("webdataset", None)
            else:
                sys.modules["webdataset"] = original_wds

        self.assertEqual(result["samples"], 1)
        self.assertEqual([record["event"] for record in records], ["start", "opened", "sample", "finish"])
        self.assertEqual(records[2]["last_key"], "sample-000001")
        self.assertEqual(records[2]["samples"], 1)

    def test_parallel_scan_writes_outputs_before_nonblocking_shutdown(self):
        class _FakeFuture:
            def __init__(self, payload):
                self._payload = payload

            def result(self):
                return self._payload

        class _FakeExecutor:
            def __init__(self, *args, **kwargs):
                self.shutdown_calls = []
                self._submitted = 0

            def submit(self, _fn, params):
                shard_index = params["shard_index"]
                self._submitted += 1
                return _FakeFuture(
                    {
                        "shard_index": shard_index,
                        "shard": params["shard_path"],
                        "samples": 1,
                        "passed": 1,
                        "failed": 0,
                        "reasons": {},
                        "reason_counts": {},
                        "good_path": None,
                        "bad_path": None,
                    }
                )

            def shutdown(self, wait, cancel_futures):
                self.shutdown_calls.append((wait, cancel_futures))
                if wait:
                    raise AssertionError("shutdown(wait=True) would block output emission")

        fake_executor = _FakeExecutor()
        append_calls = []

        args = argparse.Namespace(
            kind="auto",
            shards=["/data/shard-000000.tar", "/data/shard-000001.tar"],
            max_samples=0,
            workers=2,
            check_media=False,
            check_depth=False,
            check_image_quality=False,
            target_image_size=None,
            max_shard_fail_rate=0.0,
            report=None,
            good_keys_output="/tmp/good.jsonl",
            bad_keys_output="/tmp/bad.jsonl",
            filtered_shards_output=None,
            shard_progress_output=None,
            shard_progress_interval=1000,
        )

        with (
            mock.patch.object(mod, "load_training_checkers"),
            mock.patch.object(mod, "expand_paths", return_value=args.shards),
            mock.patch.object(mod.concurrent.futures, "ProcessPoolExecutor", return_value=fake_executor),
            mock.patch.object(mod.concurrent.futures, "as_completed", side_effect=lambda futures: list(futures)),
            mock.patch.object(mod, "append_jsonl_files", side_effect=lambda paths, output: append_calls.append(output)),
            mock.patch.object(mod, "build_wds_report", return_value={"ok": True}),
        ):
            result = mod.scan_wds(args)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(append_calls, ["/tmp/good.jsonl", "/tmp/bad.jsonl"])
        self.assertEqual(fake_executor.shutdown_calls, [(False, True)])


if __name__ == "__main__":
    unittest.main()
