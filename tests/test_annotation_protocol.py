import json
import tempfile
import unittest
from pathlib import Path

from lib.pipeline.annotation_protocol import load_clip_annotation


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


if __name__ == "__main__":
    unittest.main()
