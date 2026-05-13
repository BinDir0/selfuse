import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MeasureLegacyRot6DRepairTests(unittest.TestCase):
    def test_measurement_lab_clean_case_passes(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure_legacy_rot6d_repair.py"),
                "--case",
                "clean",
                "--episodes",
                "2",
                "--frames-per-episode",
                "1",
                "--shards",
                "1",
                "--workers",
                "1",
                "--executor",
                "thread",
            ],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["source_samples"], 2)
        self.assertEqual(payload["cases"], ["clean"])


if __name__ == "__main__":
    unittest.main()
