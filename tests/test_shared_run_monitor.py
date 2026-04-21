import json
import tempfile
import unittest
from pathlib import Path

from lib.pipeline.shared_run_monitor import summarize_runs


class SharedRunMonitorTests(unittest.TestCase):
    def test_summarize_runs_tolerates_null_run_summary_and_event_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            run_dir = log_root / "run_null"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_summary.json").write_text("null\n", encoding="utf-8")
            (run_dir / "events.jsonl").write_text("null\n", encoding="utf-8")

            rows = summarize_runs(log_root, pattern="run_*", limit=8, stall_seconds=1800)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_tag"], "run_null")
            self.assertEqual(rows[0]["health"], "empty")
            self.assertTrue(any("run_summary" in item for item in rows[0]["status_errors"]))

    def test_summarize_runs_tolerates_invalid_task_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            run_dir = log_root / "run_a"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "config": "/tmp/config.yaml",
                        "expanded_internal_stages": ["slam"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "stages": ["slam"],
                        "tasks": {
                            "good": {"stage_status": {"slam": "completed"}},
                            "bad_task": ["not", "a", "dict"],
                            "bad_stage_status": {"stage_status": ["broken"]},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "events.jsonl").write_text(
                json.dumps({"time": "2026-04-21T12:00:00"}) + "\n",
                encoding="utf-8",
            )

            rows = summarize_runs(log_root, pattern="run_*", limit=8, stall_seconds=1800)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["total"], 2)
            self.assertEqual(rows[0]["completed"], 1)
            self.assertTrue(any("invalid task record" in item for item in rows[0]["status_errors"]))
            self.assertTrue(any("invalid stage_status" in item for item in rows[0]["status_errors"]))


if __name__ == "__main__":
    unittest.main()
