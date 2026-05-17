import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.pipeline.datasets import DatasetAdapterContext, get_dataset_adapter
from lib.pipeline.pipeline_config import normalize_pipeline_config


def _write_test_video(path: Path, *, fps: float = 12.5, frames: int = 4, size=(32, 24)) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise unittest.SkipTest("cv2 is required for single_video adapter tests") from exc

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        tuple(size),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV VideoWriter is unavailable in this environment")
    try:
        for idx in range(frames):
            frame = np.full((size[1], size[0], 3), idx * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class SingleVideoAdapterTests(unittest.TestCase):
    def test_prepare_extracts_frames_and_builds_descriptor_with_native_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video = tmp / "input.mp4"
            _write_test_video(video, fps=12.5, frames=4)
            config = normalize_pipeline_config(
                {
                    "video": str(video),
                    "output_root": str(tmp / "out"),
                }
            )
            run_dir = Path(config["paths"]["log_root"]) / "run"
            context = DatasetAdapterContext(
                project_root=tmp,
                run_dir=run_dir,
                manifest_path=run_dir / "clip_manifest.jsonl",
                shard_dirs_list_path=run_dir / "shard_dirs.txt",
                summary_path=run_dir / "run_summary.json",
            )
            adapter = get_dataset_adapter("single_video")

            prepared = adapter.prepare(
                dataset_cfg=config["dataset"],
                adapter_cfg=config["adapter_config"],
                paths_cfg=config["paths"],
                runtimes_cfg=config["runtimes"],
                context=context,
            )
            descriptors = list(
                adapter.build_descriptors(
                    dataset_cfg=config["dataset"],
                    adapter_cfg=config["adapter_config"],
                    paths_cfg=config["paths"],
                    context=context,
                    prepared=prepared,
                )
            )

            self.assertEqual(len(descriptors), 1)
            descriptor = descriptors[0]
            self.assertEqual(descriptor.clip_id, "input")
            self.assertEqual(descriptor.frame_count, 4)
            self.assertEqual(descriptor.width, 32)
            self.assertEqual(descriptor.height, 24)
            self.assertAlmostEqual(float(descriptor.fps), 12.5, places=1)
            self.assertEqual(descriptor.media_path, str(video.resolve()))
            self.assertTrue(Path(descriptor.frame_dir, "000000.jpg").is_file())
            self.assertTrue(Path(descriptor.seq_folder).is_dir())

            validation = adapter.validate_source(
                dataset_cfg=config["dataset"],
                adapter_cfg=config["adapter_config"],
                paths_cfg=config["paths"],
                context=context,
                prepared=prepared,
            )
            self.assertTrue(validation.ok)
            self.assertEqual(validation.summary["frame_count"], 4)

    def test_resume_skips_reextraction_only_when_frames_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video = tmp / "input.mp4"
            _write_test_video(video, fps=12.5, frames=4)
            config = normalize_pipeline_config(
                {"video": str(video), "output_root": str(tmp / "out")}
            )
            run_dir = Path(config["paths"]["log_root"]) / "run"
            context = DatasetAdapterContext(
                project_root=tmp,
                run_dir=run_dir,
                manifest_path=run_dir / "clip_manifest.jsonl",
                shard_dirs_list_path=run_dir / "shard_dirs.txt",
                summary_path=run_dir / "run_summary.json",
            )
            adapter = get_dataset_adapter("single_video")
            adapter_cfg = config["adapter_config"]

            def _prepare():
                return adapter.prepare(
                    dataset_cfg=config["dataset"],
                    adapter_cfg=adapter_cfg,
                    paths_cfg=config["paths"],
                    runtimes_cfg=config["runtimes"],
                    context=context,
                )

            first = _prepare()
            self.assertNotEqual(first.payload.get("resumed"), True)

            adapter_cfg["resume"] = True
            resumed = _prepare()
            self.assertTrue(resumed.payload.get("resumed"))

            frame_dir = Path(resumed.payload["descriptor"].frame_dir)
            next(frame_dir.glob("*.jpg")).unlink()
            reextracted = _prepare()
            self.assertNotEqual(reextracted.payload.get("resumed"), True)

            adapter_cfg["resume"] = False
            never_resumes = _prepare()
            self.assertNotEqual(never_resumes.payload.get("resumed"), True)


if __name__ == "__main__":
    unittest.main()
