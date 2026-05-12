import json
import sys
import tempfile
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
