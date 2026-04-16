import tempfile
import unittest
from pathlib import Path

import numpy as np


class ManifestBuildAnnotationTests(unittest.TestCase):
    def test_prepare_manifest_episodes_collects_missing_annotation_issues(self):
        try:
            import joblib
            from lib.pipeline.clip_manifest import ClipManifestRecord, write_clip_manifest
            from lib.pipeline.datasets.descriptors import ClipDescriptor
            from lib.pipeline.exporters.manifest_build.episodes import prepare_manifest_episodes
        except ModuleNotFoundError as error:
            self.skipTest(f"optional dependency missing: {error}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_folder = root / "outputs" / "f001_w001_v00000_i000"
            frame_dir = seq_folder / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump((np.zeros((2, 3, 3), dtype=np.float32), None, None, None, None), seq_folder / "world_space_res.pth")

            descriptor = ClipDescriptor.from_image_sequence(
                clip_id="f001_w001_v00000_i000",
                clip_name="f001_w001_v00000_i000",
                root_dir=str(root),
                seq_folder=str(seq_folder),
                frame_dir=str(frame_dir),
                frame_names=["000000.jpg", "000001.jpg", "000002.jpg"],
            )
            manifest_path = root / "manifest.jsonl"
            write_clip_manifest(
                [
                    ClipManifestRecord(
                        clip_id=descriptor.clip_id,
                        source_id="buildai",
                        split="train",
                        descriptor=descriptor,
                        group_id="factory001",
                    )
                ],
                manifest_path,
            )

            episodes, stats, annotation_issues = prepare_manifest_episodes(
                str(manifest_path),
                annotation_root=str(root / "missing_annotations"),
                annotation_suffix="_qwen-annotation.json",
                require_annotation=False,
                max_episodes=None,
                preprocess_workers=1,
                source_fps=5.0,
                target_fps=30.0,
                interpolate_labels=True,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(stats["kept"], 1)
            self.assertEqual(stats["annotation_issue_count"], 1)
            self.assertEqual(stats["annotation_issue_summary"]["missing_annotation"], 1)
            self.assertEqual(len(annotation_issues), 1)
            self.assertEqual(annotation_issues[0]["clip_id"], descriptor.clip_id)
            self.assertEqual(annotation_issues[0]["error_code"], "missing_annotation")


if __name__ == "__main__":
    unittest.main()
