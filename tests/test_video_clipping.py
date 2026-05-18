import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lib.pipeline.video_clipping import apply_video_clipping_if_configured


class VideoClippingTests(unittest.TestCase):
    def test_no_clipping_leaves_config_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run"
            run_dir.mkdir(parents=True)
            config = {
                "dataset": {"adapter": "single_video", "source_id": "demo", "split": "train"},
                "paths": {"final_dataset_root": str(root / "wds")},
                "adapter_config": {"video": str(root / "input.mp4")},
                "clip": {"mode": "none"},
                "annotation": {"command": "echo annotate"},
                "resume": True,
            }

            summary = apply_video_clipping_if_configured(
                config=config,
                run_dir=run_dir,
                project_root=root,
            )

            self.assertIsNone(summary)
            self.assertEqual(config["dataset"]["adapter"], "single_video")
            self.assertEqual(config["adapter_config"]["video"], str(root / "input.mp4"))
            self.assertNotIn("_api_clip_completed", config["annotation"])

    def test_heuristic_clipping_redirects_single_video_to_clipped_video_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "input.mp4"
            video.write_bytes(b"fake")
            run_dir = root / "runs" / "run"
            run_dir.mkdir(parents=True)
            config = {
                "dataset": {"adapter": "single_video", "source_id": "demo", "split": "train"},
                "paths": {"final_dataset_root": str(root / "wds")},
                "adapter_config": {"video": str(video)},
                "clip": {"mode": "heuristic"},
                "resume": True,
            }

            with mock.patch(
                "lib.pipeline.video_clipping.run_heuristic_clipping",
                return_value={"summary": {"kept_clips": 2}},
            ) as run_mock:
                summary = apply_video_clipping_if_configured(
                    config=config,
                    run_dir=run_dir,
                    project_root=root,
                )

            self.assertEqual(summary["kept_clips"], 2)
            self.assertEqual(config["dataset"]["adapter"], "video_folder")
            self.assertTrue(config["adapter_config"]["extract_frames"])
            self.assertIn("clips", config["adapter_config"]["video_root"])
            self.assertEqual(run_mock.call_args.kwargs["source_root"], video.parent)

    def test_heuristic_clipping_accepts_buildai_raw_video_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_root = root / "raw_videos"
            factory_dir = video_root / "factory002"
            factory_dir.mkdir(parents=True)
            (factory_dir / "worker_clip.mp4").write_bytes(b"fake")
            (video_root / "factory003").mkdir(parents=True)
            run_dir = root / "runs" / "run"
            run_dir.mkdir(parents=True)
            config = {
                "dataset": {
                    "adapter": "buildai",
                    "source_id": "buildai_demo",
                    "split": "train",
                    "start_factory_id": 2,
                    "end_factory_id": 3,
                },
                "paths": {
                    "final_dataset_root": str(root / "wds"),
                    "video_root": str(video_root),
                },
                "adapter_config": {"stages": "1,2,3"},
                "clip": {"mode": "heuristic"},
                "resume": True,
            }

            with mock.patch(
                "lib.pipeline.video_clipping.run_heuristic_clipping",
                return_value={"summary": {"kept_clips": 1}},
            ) as run_mock:
                summary = apply_video_clipping_if_configured(
                    config=config,
                    run_dir=run_dir,
                    project_root=root,
                )

            self.assertEqual(summary["source_videos"], 1)
            self.assertEqual(config["dataset"]["adapter"], "video_folder")
            self.assertEqual(config["adapter_config"]["_original_dataset"]["adapter"], "buildai")
            self.assertEqual(run_mock.call_args.kwargs["source_root"], video_root)

    def test_api_clipping_writes_annotations_and_redirects_to_clipped_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "input.mp4"
            video.write_bytes(b"fake")
            run_dir = root / "runs" / "run"
            run_dir.mkdir(parents=True)
            config = {
                "dataset": {"adapter": "single_video", "source_id": "demo", "split": "train"},
                "paths": {"final_dataset_root": str(root / "wds")},
                "adapter_config": {"video": str(video)},
                "clip": {
                    "mode": "api",
                    "dry_run": True,
                    "annotation_suffix": "_qwen-annotation.json",
                },
                "build": {"annotation_suffix": "_qwen-annotation.json"},
                "annotation": {},
                "resume": True,
            }

            with mock.patch(
                "lib.pipeline.video_clipping.run_api_video_clipping",
                return_value={"summary": {"kept_clips": 1}},
            ) as run_mock:
                summary = apply_video_clipping_if_configured(
                    config=config,
                    run_dir=run_dir,
                    project_root=root,
                )

            self.assertEqual(summary["mode"], "api")
            self.assertEqual(config["dataset"]["adapter"], "video_folder")
            self.assertTrue(config["annotation"]["_api_clip_completed"])
            self.assertIn("annotation_root", config["paths"])
            self.assertEqual(run_mock.call_args.kwargs["annotation_suffix"], "_qwen-annotation.json")


if __name__ == "__main__":
    unittest.main()
