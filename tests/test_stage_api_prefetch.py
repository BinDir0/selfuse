import sys
import types
import unittest
from pathlib import Path
from unittest import mock


if "joblib" not in sys.modules:
    joblib_module = types.ModuleType("joblib")
    joblib_module.load = lambda *_args, **_kwargs: None
    joblib_module.dump = lambda *_args, **_kwargs: None
    sys.modules["joblib"] = joblib_module


if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **_kwargs: iterable if iterable is not None else []
    sys.modules["tqdm"] = tqdm_module


if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")
    torch_module.manual_seed = lambda *_args, **_kwargs: None
    torch_module.cuda = types.SimpleNamespace(manual_seed_all=lambda *_args, **_kwargs: None)
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


from lib.pipeline.stage_api import StageExecutionConfig, _run_infiller_stage, run_pipeline_stage


class _DummyTask:
    def __init__(self):
        self.seq_folder = Path("/tmp/prefetch-task")
        self.video_path = "dummy_video"
        self.descriptor = None
        self.build_calls = 0

    def build_frame_source(self):
        self.build_calls += 1
        raise AssertionError("build_frame_source should not be called when prefetched_data provides frame_source")


class StageApiPrefetchTests(unittest.TestCase):
    def test_run_pipeline_stage_reuses_prefetched_frame_source(self):
        task = _DummyTask()
        prefetched_frame_source = object()
        captured = {}

        def _fake_run_non_detect_stage(stage, task_obj, stage_args, config, runtime, frame_source, profiler, prefetched_data, force):
            captured["stage"] = stage
            captured["frame_source"] = frame_source
            captured["prefetched_data"] = prefetched_data
            return 0, 1, {"timing": {"3_depth": 1.0}, "stats": {"frame_count": 2}}

        with mock.patch("lib.pipeline.stage_api.is_stage_complete", return_value=False), \
            mock.patch("lib.pipeline.stage_api._ensure_runtime_for_stage", return_value=None), \
            mock.patch("lib.pipeline.stage_api._run_non_detect_stage", side_effect=_fake_run_non_detect_stage), \
            mock.patch(
                "lib.pipeline.stage_api._finalize_stage_run",
                return_value={"status": "success", "wall_sec": 0.0},
            ):
            result = run_pipeline_stage(
                "slam",
                task,
                StageExecutionConfig(),
                runtime=object(),
                prefetched_data={"frame_source": prefetched_frame_source},
                resume=False,
            )

        self.assertEqual(result["status"], "success")
        self.assertIs(captured["frame_source"], prefetched_frame_source)
        self.assertEqual(task.build_calls, 0)

    def test_infiller_does_not_build_frame_source_when_descriptor_frame_count_exists(self):
        task = _DummyTask()
        task.descriptor = types.SimpleNamespace(frame_count=123)
        captured = {}

        def _fake_run_non_detect_stage(stage, task_obj, stage_args, config, runtime, frame_source, profiler, prefetched_data, force):
            captured["stage"] = stage
            captured["frame_source"] = frame_source
            return 0, 1, {"timing": {"model_forward": 1.0}, "stats": {"frame_count": 123}}

        with mock.patch("lib.pipeline.stage_api.is_stage_complete", return_value=False), \
            mock.patch("lib.pipeline.stage_api._ensure_runtime_for_stage", return_value=None), \
            mock.patch("lib.pipeline.stage_api._run_non_detect_stage", side_effect=_fake_run_non_detect_stage), \
            mock.patch(
                "lib.pipeline.stage_api._finalize_stage_run",
                return_value={"status": "success", "wall_sec": 0.0},
            ):
            result = run_pipeline_stage(
                "infiller",
                task,
                StageExecutionConfig(),
                runtime=object(),
                resume=False,
            )

        self.assertEqual(result["status"], "success")
        self.assertIsNone(captured["frame_source"])
        self.assertEqual(task.build_calls, 0)

    def test_run_infiller_stage_reuses_prefetched_chunks_and_cache(self):
        task = types.SimpleNamespace(
            seq_folder=Path("/tmp/prefetch-task"),
            descriptor=types.SimpleNamespace(frame_count=77),
        )
        stage_args = object()
        runtime = types.SimpleNamespace(infiller_runner=object())
        frame_chunks_all = {"chunks": [1]}
        cam_space_cache = {"cache": True}
        expected_metrics = {"timing": {"model_forward": 1.0}, "stats": {"frame_count": 77}}
        fake_hawor_video = types.ModuleType("lib.pipeline.stages.hawor_video")
        fake_hawor_video.run_infiller_for_video = mock.Mock(return_value=expected_metrics)

        with mock.patch.dict(sys.modules, {"lib.pipeline.stages.hawor_video": fake_hawor_video}), \
            mock.patch("lib.pipeline.stage_api.joblib.load") as load_mock:
            metrics = _run_infiller_stage(
                task,
                stage_args,
                runtime,
                frame_source=None,
                prefetched_data={
                    "frame_chunks_all": frame_chunks_all,
                    "cam_space_cache": cam_space_cache,
                },
                start_idx=0,
                end_idx=1,
            )

        self.assertEqual(metrics, expected_metrics)
        load_mock.assert_not_called()
        fake_hawor_video.run_infiller_for_video.assert_called_once_with(
            stage_args,
            0,
            1,
            frame_chunks_all,
            infiller_runner=runtime.infiller_runner,
            frame_source=None,
            seq_folder=str(task.seq_folder),
            num_frames=77,
            cam_space_cache=cam_space_cache,
            return_timing=True,
        )


if __name__ == "__main__":
    unittest.main()
