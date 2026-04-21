import unittest

from lib.pipeline.batch.cli import build_batch_infer_parser, normalize_batch_infer_args
from lib.pipeline.orchestrator.helpers import cli_args_from_mapping
from lib.pipeline.orchestrator.stage_selection import selected_stages


class OrchestratorStageSelectionTests(unittest.TestCase):
    def test_selected_stages_expands_official_aliases(self):
        result = selected_stages("prepare,infer,build")
        self.assertEqual(result["requested_public"], ["prepare", "infer", "build"])
        self.assertEqual(
            result["internal"],
            ["preprocess", "manifest", "detect_motion", "slam", "infiller", "build"],
        )
        self.assertEqual(result["deprecated"], [])

    def test_selected_stages_keeps_legacy_compatibility(self):
        result = selected_stages("manifest,slam")
        self.assertEqual(result["requested_public"], ["prepare", "infer"])
        self.assertEqual(result["internal"], ["manifest", "slam"])
        self.assertEqual(sorted(result["deprecated"]), ["manifest", "slam"])

    def test_cli_args_from_mapping_supports_negative_bool_flags(self):
        args = cli_args_from_mapping(
            {
                "resume": False,
                "foo": True,
                "bar": ["x", "y"],
                "baz": 3,
                "skip": None,
            },
            negative_bool_flags={"resume"},
        )
        self.assertEqual(args, ["--no-resume", "--foo", "--bar", "x", "y", "--baz", "3"])

    def test_throughput_profile_enables_safe_defaults(self):
        parser = build_batch_infer_parser()
        args = parser.parse_args(["--descriptor_manifest", "/tmp/fake.jsonl", "--infer_profile", "throughput_80gb"])
        notes = normalize_batch_infer_args(args, raw_argv=["--descriptor_manifest", "/tmp/fake.jsonl", "--infer_profile", "throughput_80gb"])
        self.assertEqual(args.detect_track_workers_per_gpu, 2)
        self.assertEqual(args.motion_workers_per_gpu, 1)
        self.assertEqual(args.slam_workers_per_gpu, 1)
        self.assertEqual(args.infiller_workers_per_gpu, 2)
        self.assertEqual(args.local_cache_mode, "all")
        self.assertEqual(args.local_cache_quota_gb, 2000.0)
        self.assertGreaterEqual(args.local_cache_min_frames, 96)
        self.assertTrue(any("throughput_80gb" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
