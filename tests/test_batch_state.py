import json
import tempfile
import unittest
from pathlib import Path


class BatchStateRecoveryTests(unittest.TestCase):
    def test_load_status_payload_falls_back_to_backup(self):
        try:
            from lib.pipeline.batch.state import load_status_payload_with_fallback, status_backup_path
        except ModuleNotFoundError as error:
            self.skipTest(f"optional dependency missing: {error}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_path = root / "status.json"
            backup_path = status_backup_path(status_path)
            status_path.write_text('{"tasks": {"clip_a": {"stage_status": {"motion": "completed"}}}', encoding="utf-8")
            backup_payload = {
                "run_dir": str(root),
                "gpus": [0],
                "stages": ["motion"],
                "tasks": {"clip_a": {"stage_status": {"motion": "completed"}}},
            }
            backup_path.write_text(json.dumps(backup_payload), encoding="utf-8")

            payload, meta = load_status_payload_with_fallback(status_path, stages=["motion"], video_paths=["clip_a"])

            self.assertEqual(meta["source"], "backup")
            self.assertEqual(payload["tasks"]["clip_a"]["stage_status"]["motion"], "completed")

    def test_load_status_payload_falls_back_to_events(self):
        try:
            from lib.pipeline.batch.state import load_status_payload_with_fallback
        except ModuleNotFoundError as error:
            self.skipTest(f"optional dependency missing: {error}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_path = root / "status.json"
            events_path = root / "events.jsonl"
            status_path.write_text('{"tasks": {"clip_a": {"stage_status": {"motion": "completed"}}}', encoding="utf-8")
            events_path.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "stage_success", "video": "clip_a", "stage": "detect_track"}),
                        json.dumps({"event": "stage_failure", "video": "clip_a", "stage": "motion"}),
                        '{"event":"stage_success","video":"clip_b"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload, meta = load_status_payload_with_fallback(
                status_path,
                events_path=events_path,
                stages=["detect_track", "motion"],
                video_paths=["clip_a", "clip_b"],
            )

            self.assertEqual(meta["source"], "events")
            self.assertEqual(payload["tasks"]["clip_a"]["stage_status"]["detect_track"], "completed")
            self.assertEqual(payload["tasks"]["clip_a"]["stage_status"]["motion"], "failed")
            self.assertEqual(payload["tasks"]["clip_b"]["stage_status"]["motion"], "pending")
            self.assertEqual(meta["malformed_lines"], 1)


if __name__ == "__main__":
    unittest.main()
