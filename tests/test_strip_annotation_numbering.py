import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar
from scripts import strip_annotation_numbering


class StripAnnotationNumberingTests(unittest.TestCase):
    def test_dry_run_reports_dataset_without_rewriting_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_path = root / "shard-000000.tar"
            meta = {
                "dataset_name": "buildai_test",
                "clip_id": "clip_a",
                "instruction": ["1. pick object", "2. place object"],
                "instruction_num": 2,
            }
            with tarfile.open(shard_path, "w") as tar_writer:
                write_sample_to_tar(
                    tar_writer,
                    "sample_a",
                    b"jpg",
                    b"lowdim",
                    json.dumps(meta).encode("utf-8"),
                )

            report_path = root / "report.json"
            stdout = io.StringIO()
            with patch.object(
                strip_annotation_numbering.sys,
                "argv",
                [
                    "strip_annotation_numbering.py",
                    "--dry-run",
                    "--report-out",
                    str(report_path),
                    str(root),
                ],
            ), redirect_stdout(stdout):
                strip_annotation_numbering.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["datasets"]["buildai_test"]["entries"], 1)
            self.assertEqual(report["datasets"]["buildai_test"]["clips"], 1)
            self.assertIn("would update shard", stdout.getvalue())

            sample = next(iter_shard_samples(str(shard_path)))
            unchanged_meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            self.assertEqual(unchanged_meta["instruction"], ["1. pick object", "2. place object"])

    def test_rewrite_removes_numbering_from_shard_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_path = root / "shard-000000.tar"
            meta = {
                "dataset_name": "buildai_test",
                "clip_id": "clip_a",
                "instruction": ["1. pick object", "2. place object"],
                "instruction_num": 2,
            }
            with tarfile.open(shard_path, "w") as tar_writer:
                write_sample_to_tar(
                    tar_writer,
                    "sample_a",
                    b"jpg",
                    b"lowdim",
                    json.dumps(meta).encode("utf-8"),
                )

            with patch.object(strip_annotation_numbering.sys, "argv", ["strip_annotation_numbering.py", str(root)]):
                strip_annotation_numbering.main()

            sample = next(iter_shard_samples(str(shard_path)))
            rewritten_meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            self.assertEqual(rewritten_meta["instruction"], ["pick object", "place object"])


if __name__ == "__main__":
    unittest.main()
