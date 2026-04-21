import unittest

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


if __name__ == "__main__":
    unittest.main()
