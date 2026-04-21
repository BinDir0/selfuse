import unittest
from argparse import Namespace
from pathlib import Path

from lib.pipeline.batch.config import BatchRunConfig


class BatchConfigTests(unittest.TestCase):
    def test_from_namespace_applies_defaults_and_env_overrides(self):
        config = BatchRunConfig.from_namespace(
            Namespace(gpus="0,2", stages="slam"),
            video_paths=["clip_a"],
            descriptors=None,
            run_dir=Path("/tmp/run"),
        )

        self.assertEqual(config.gpus, [0, 2])
        self.assertEqual(config.stages, ["slam"])
        self.assertEqual(config.any4d_batch_size, 32)
        self.assertEqual(config.infer_profile, "standard")
        self.assertEqual(
            config.worker_env_overrides(),
            {
                "HAWOR_LOCAL_CACHE_ROOT": None,
                "HAWOR_LOCAL_CACHE_QUOTA_GB": None,
                "HAWOR_LOCAL_CACHE_MODE": "off",
                "HAWOR_LOCAL_CACHE_MIN_FRAMES": "1",
            },
        )


if __name__ == "__main__":
    unittest.main()
