import json
import tempfile
import unittest
from pathlib import Path

from lib.annotation.api_annotation import build_pipeline_annotation_payload, main
from lib.annotation.api_annotation_with_clip import _annotation_payload
from lib.pipeline.annotation_protocol import load_clip_annotation
from lib.pipeline.clip_manifest import ClipManifestRecord, write_clip_manifest
from lib.pipeline.datasets.descriptors import ClipDescriptor


def _record(tmp: Path) -> ClipManifestRecord:
    descriptor = ClipDescriptor.from_image_sequence(
        clip_id="clip_a",
        clip_name="clip_a",
        root_dir=str(tmp),
        seq_folder=str(tmp / "seq" / "clip_a"),
        frame_dir=str(tmp / "frames" / "clip_a"),
        frame_names=["000000.jpg"],
        media_path=str(tmp / "clip_a.mp4"),
    )
    return ClipManifestRecord(
        clip_id="clip_a",
        source_id="unit",
        split="train",
        descriptor=descriptor,
        group_id="unit",
    )


class ApiAnnotationTests(unittest.TestCase):
    def test_build_pipeline_annotation_payload_matches_annotation_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = _record(Path(tmpdir))
            payload = build_pipeline_annotation_payload(
                record=record,
                model_payload={
                    "status": "Valid",
                    "is_good_quality": True,
                    "language_instructions": {
                        "level1": "Pick the part",
                        "level2": "Pick the part from the bin",
                        "level5": "Grip the part. Lift the part.",
                    },
                },
                raw_text="{}",
                model="qwen-test",
            )

            self.assertEqual(payload["status"], "Valid")
            self.assertEqual(payload["instruction_num"], 3)
            self.assertEqual(payload["instruction"][0], "Pick the part")
            self.assertEqual(payload["language"], "Grip the part. Lift the part.")
            self.assertEqual(payload["hierarchy"]["level5"], "Grip the part. Lift the part.")

    def test_dry_run_reads_manifest_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "clip_manifest.jsonl"
            annotation_root = tmp / "annotations"
            write_clip_manifest([_record(tmp)], manifest)

            main(
                [
                    "--prepared_state",
                    str(manifest),
                    "--annotation_root",
                    str(annotation_root),
                    "--dry_run",
                    "--workers",
                    "1",
                ]
            )

            report = json.loads((annotation_root / "_annotation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["total"], 1)
            self.assertEqual(report["summary"]["dry_run"], 1)

    def test_api_clip_payload_matches_standard_annotation_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            annotation_root = tmp / "annotations"
            payload = _annotation_payload(
                clip_id="input_clip000",
                clip_name="input_clip000",
                source_video=tmp / "input.mp4",
                segment={
                    "start": 1.0,
                    "end": 2.5,
                    "is_good_quality": True,
                    "language_instructions": {
                        "level1": "Open the drawer.",
                        "level2": "Pull the drawer open.",
                        "level5": "Grip the handle. Pull the drawer outward.",
                    },
                },
                model="qwen-test",
                raw_text="[]",
            )
            annotation_root.mkdir()
            (annotation_root / "input_clip000.annotation.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            annotation, error_code, _path = load_clip_annotation(annotation_root, "input_clip000")

            self.assertIsNone(error_code)
            self.assertIsNotNone(annotation)
            self.assertEqual(annotation.instruction_num, 3)
            self.assertEqual(annotation.language, "Grip the handle. Pull the drawer outward.")


if __name__ == "__main__":
    unittest.main()
