import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QuarantineUnrefactoredToolsTests(unittest.TestCase):
    def test_unrefactored_tool_manifest_points_to_moved_paths(self):
        manifest_path = PROJECT_ROOT / "deprecated" / "unrefactored_tools" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["recommended_entrypoint"], "scripts/run_dataset_pipeline.py")
        for entry in manifest["paths"]:
            original = PROJECT_ROOT / entry["original"]
            deprecated = PROJECT_ROOT / entry["deprecated"]
            self.assertFalse(original.exists(), entry["original"])
            self.assertTrue(deprecated.exists(), entry["deprecated"])


if __name__ == "__main__":
    unittest.main()
