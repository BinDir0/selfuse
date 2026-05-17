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

    def test_no_active_code_references_moved_original_paths(self):
        """Active (non-deprecated) files must not point at moved script paths.

        Guards against regressions where maintained code or docs reference an
        original ``scripts/...`` path that has since moved into the quarantine.
        A reference is allowed only when it is the new
        ``deprecated/unrefactored_tools/...`` path.
        """
        manifest_path = PROJECT_ROOT / "deprecated" / "unrefactored_tools" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        originals = [entry["original"].rstrip("/") for entry in manifest["paths"]]

        skip_dirs = {
            ".git",
            "deprecated",
            "__pycache__",
            "weights",
            "_DATA",
            ".scratch",
            "thirdparty",
        }
        scan_suffixes = {".py", ".md", ".sh", ".yaml", ".yml"}

        offenders = []
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in scan_suffixes:
                continue
            rel = path.relative_to(PROJECT_ROOT)
            if any(part in skip_dirs for part in rel.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if "deprecated/unrefactored_tools/" in line:
                    continue
                for original in originals:
                    if original in line:
                        offenders.append(f"{rel}:{line_no} -> {original}")

        self.assertEqual(
            offenders,
            [],
            f"Active references to moved/quarantined scripts: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
