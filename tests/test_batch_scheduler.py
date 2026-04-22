import unittest
import sys
import types


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

from lib.pipeline.batch.scheduler import BatchScheduler


class BatchSchedulerTests(unittest.TestCase):
    def test_summarize_hotspot_uses_timing_only(self):
        metrics = {
            "timing": {
                "3_depth": 12.5,
                "frame_count": 2265,
                "cache_hit": 1,
            },
            "stats": {
                "frame_count": 2265,
                "dense_depth_cache_hit": 1,
            },
        }

        hotspot = BatchScheduler._summarize_hotspot(metrics)

        self.assertEqual(hotspot, "3_depth:12.5s")

    def test_summarize_hotspot_returns_dash_when_no_valid_timing(self):
        metrics = {
            "timing": {
                "frame_count": 2265,
                "dense_depth_cache_hit": 1,
                "is_ready": False,
            },
            "stats": {
                "frame_count": 2265,
            },
        }

        hotspot = BatchScheduler._summarize_hotspot(metrics)

        self.assertEqual(hotspot, "-")


if __name__ == "__main__":
    unittest.main()
