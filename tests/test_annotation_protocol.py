import json
import tempfile
import unittest
from pathlib import Path

from lib.pipeline.annotation_protocol import (
    build_annotation_issue_from_candidates,
    load_clip_annotation,
    summarize_annotation_issues,
    write_annotation_issue_report,
)


class AnnotationProtocolTests(unittest.TestCase):
    def test_load_clip_annotation_supports_nested_factory_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clip_id = "f001_w001_v00000_i000"
            ann_path = root / "factory001" / f"{clip_id}.annotation.json"
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            ann_path.write_text(
                json.dumps(
                    {
                        "status": "Valid",
                        "global_analysis": {
                            "level1": "pick up object",
                            "level2": "move object",
                        },
                        "language": "en",
                    }
                ),
                encoding="utf-8",
            )

            annotation, error_code, source_path = load_clip_annotation(root, clip_id)

            self.assertIsNone(error_code)
            self.assertIsNotNone(annotation)
            self.assertEqual(annotation.instruction, ["pick up object", "move object"])
            self.assertEqual(source_path, str(ann_path))

    def test_write_annotation_issue_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "annotation_issues.json"
            issues = [
                {"clip_id": "clip_a", "error_code": "missing_annotation", "resolved_path": "/tmp/a.json"},
                {"clip_id": "clip_b", "error_code": "empty_instruction", "resolved_path": "/tmp/b.json"},
            ]

            written_path = write_annotation_issue_report(
                report_path,
                annotation_root=root / "ann",
                annotation_suffix="_qwen-annotation.json",
                issues=issues,
                context={"mode": "test"},
            )

            payload = json.loads(Path(written_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"], summarize_annotation_issues(issues))
            self.assertEqual(payload["annotation_suffix"], "_qwen-annotation.json")
            self.assertEqual(payload["context"]["mode"], "test")
            self.assertEqual(len(payload["issues"]), 2)

    def test_build_annotation_issue_from_candidates_includes_nested_paths(self):
        issue = build_annotation_issue_from_candidates(
            "/tmp/annotations",
            "f011_w016_v00162_i001",
            "missing_annotation",
            annotation_suffix="_qwen-annotation.json",
        )
        self.assertEqual(issue["error_code"], "missing_annotation")
        self.assertEqual(
            issue["candidate_paths"],
            [
                "/tmp/annotations/f011_w016_v00162_i001_qwen-annotation.json",
                "/tmp/annotations/factory011/f011_w016_v00162_i001_qwen-annotation.json",
                "/tmp/annotations/factory011/factory_011_worker_016_0162_cut001_qwen-annotation.json",
                "/tmp/annotations/factory_011_worker_016_0162_cut001_qwen-annotation.json",
            ],
        )

    def test_load_clip_annotation_supports_buildai_qwen_factory_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clip_id = "f026_w001_v00054_i000"
            ann_path = root / "factory026" / "factory_026_worker_001_0054_cut000_qwen-annotation.json"
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            ann_path.write_text(
                json.dumps(
                    {
                        "status": "Valid",
                        "global_analysis": {
                            "level1": "pick part",
                            "level2": "place part",
                        },
                        "language": "en",
                    }
                ),
                encoding="utf-8",
            )

            annotation, error_code, source_path = load_clip_annotation(
                root,
                clip_id,
                annotation_suffix="_qwen-annotation.json",
            )

            self.assertIsNone(error_code)
            self.assertIsNotNone(annotation)
            self.assertEqual(annotation.instruction, ["pick part", "place part"])
            self.assertEqual(source_path, str(ann_path))


if __name__ == "__main__":
    unittest.main()
