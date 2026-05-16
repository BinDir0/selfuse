import unittest
import tempfile
from pathlib import Path
from unittest import mock

from lib.pipeline.pipeline_config import normalize_pipeline_config


class PipelineConfigTests(unittest.TestCase):
    def test_normalize_pipeline_config_accepts_official_nested_shape(self):
        config = normalize_pipeline_config(
            {
                "dataset": {"adapter": "buildai", "source_id": "demo", "split": "train"},
                "paths": {"final_dataset_root": "/tmp/out"},
                "runtimes": {"hawor_python": "/tmp/python"},
                "infer": {
                    "common": {"gpus": "0"},
                    "detect_motion": {},
                    "slam": {},
                    "infiller": {},
                },
            }
        )

        self.assertEqual(config["dataset"]["adapter"], "buildai")
        self.assertIn("infer", config)
        self.assertNotIn("batch_infer", config)
        self.assertEqual(config["_meta"]["schema"], "nested")
        self.assertTrue(config["_meta"]["migration_warnings"])

    def test_normalize_pipeline_config_rejects_compact_legacy_shape(self):
        with self.assertRaisesRegex(ValueError, "Compact/legacy"):
            normalize_pipeline_config(
                {
                    "adapter": "buildai",
                    "source_id": "demo",
                    "shard_root": "/tmp/shards",
                    "infer": {"common": {"gpus": "0"}},
                }
            )

    def test_normalize_pipeline_config_accepts_minimal_single_video_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video = tmp / "demo clip.mp4"
            video.write_bytes(b"fake")

            config = normalize_pipeline_config({"video": str(video)})

            output_root = tmp / "demo clip.hawor_pipeline"
            self.assertEqual(config["_meta"]["schema"], "single_video")
            self.assertEqual(config["dataset"]["adapter"], "single_video")
            self.assertEqual(config["adapter_config"]["video"], str(video.resolve()))
            self.assertEqual(config["paths"]["output_root"], str(output_root.resolve()))
            self.assertEqual(config["paths"]["final_dataset_root"], str((output_root / "webdataset").resolve()))
            self.assertNotIn("annotation_root", config["paths"])
            self.assertEqual(config["run_tag"], "run")
            self.assertTrue(config["resume"])
            self.assertFalse(config["build"]["require_annotation"])
            self.assertTrue(config["build"]["export_depth"])
            self.assertIsNone(config["build"]["source_fps"])
            self.assertEqual(config["_meta"]["default_stages"], "prepare,infer,filter,build,validate")

    def test_single_video_default_stages_include_annotate_only_with_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "demo.mp4"
            video.write_bytes(b"fake")
            config = normalize_pipeline_config(
                {
                    "video": str(video),
                    "annotation": {"command": "echo {manifest}"},
                }
            )
            self.assertEqual(config["_meta"]["default_stages"], "prepare,annotate,infer,filter,build,validate")
            self.assertIn("annotation_root", config["paths"])

    def test_single_video_defaults_to_visible_cuda_devices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "demo.mp4"
            video.write_bytes(b"fake")
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3"}):
                config = normalize_pipeline_config({"video": str(video)})
            self.assertEqual(config["infer"]["common"]["gpus"], "2,3")


if __name__ == "__main__":
    unittest.main()
