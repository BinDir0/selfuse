import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **_kwargs: iterable if iterable is not None else []
    sys.modules["tqdm"] = tqdm_module


if "joblib" not in sys.modules:
    joblib_module = types.ModuleType("joblib")
    joblib_module.load = lambda *_args, **_kwargs: None
    joblib_module.dump = lambda *_args, **_kwargs: None
    sys.modules["joblib"] = joblib_module


if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")
    torch_module.manual_seed = lambda *_args, **_kwargs: None
    torch_module.cuda = types.SimpleNamespace(
        manual_seed_all=lambda *_args, **_kwargs: None,
    )
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
    )
    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_data_module = types.ModuleType("torch.utils.data")
    torch_utils_data_module.Dataset = type("Dataset", (), {})
    torch_utils_data_module.get_worker_info = lambda: None
    torch_utils_module.data = torch_utils_data_module
    torch_module.utils = torch_utils_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = torch_utils_module
    sys.modules["torch.utils.data"] = torch_utils_data_module


from lib.pipeline.batch.worker_pool import StageWorkerPool, _prefetch_video_data
from lib.pipeline.datasets.descriptors import ClipDescriptor


class _DummyConfig:
    def __init__(self, descriptors):
        self._descriptor_map = {descriptor.video_key: descriptor for descriptor in descriptors}
        self.resume = True

    @property
    def descriptor_map(self):
        return self._descriptor_map


class BatchWorkerPoolTests(unittest.TestCase):
    def test_motion_prioritizes_shard_locality(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shard_a = os.path.join(tmp_dir, "shard_a.tar")
            shard_b = os.path.join(tmp_dir, "shard_b.tar")
            open(shard_a, "wb").close()
            open(shard_b, "wb").close()

            descriptors = [
                ClipDescriptor.from_tar_shard(
                    clip_id="clip_a",
                    clip_name="clip_a",
                    root_dir=tmp_dir,
                    seq_folder=os.path.join(tmp_dir, "outputs", "clip_a"),
                    shard_path=shard_a,
                    frame_names=["clip_a_f000000.jpg"] * 30,
                    frame_offsets=None,
                ),
                ClipDescriptor.from_tar_shard(
                    clip_id="clip_b",
                    clip_name="clip_b",
                    root_dir=tmp_dir,
                    seq_folder=os.path.join(tmp_dir, "outputs", "clip_b"),
                    shard_path=shard_a,
                    frame_names=["clip_b_f000000.jpg"] * 29,
                    frame_offsets=None,
                ),
                ClipDescriptor.from_tar_shard(
                    clip_id="clip_c",
                    clip_name="clip_c",
                    root_dir=tmp_dir,
                    seq_folder=os.path.join(tmp_dir, "outputs", "clip_c"),
                    shard_path=shard_b,
                    frame_names=["clip_c_f000000.jpg"] * 50,
                    frame_offsets=None,
                ),
            ]

            pool = StageWorkerPool(_DummyConfig(descriptors))
            ordered = pool._prioritize_videos([descriptor.video_key for descriptor in descriptors], "motion")

            self.assertEqual(ordered, [descriptors[0].video_key, descriptors[1].video_key, descriptors[2].video_key])

    def test_slam_prefetch_builds_frame_source_only(self):
        marker = object()
        config = _DummyConfig([])

        class _DummyTask:
            seq_folder = Path("/tmp/slam-prefetch")

            def build_frame_source(self):
                return marker

        with mock.patch("lib.pipeline.batch.worker_pool._build_pipeline_task", return_value=_DummyTask()), \
            mock.patch("lib.pipeline.batch.worker_pool.is_stage_complete", return_value=False):
            prefetched = _prefetch_video_data("video_key", "slam", {}, config)

        self.assertEqual(prefetched, {"frame_source": marker})


if __name__ == "__main__":
    unittest.main()
