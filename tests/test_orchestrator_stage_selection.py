import unittest

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


if __name__ == "__main__":
    unittest.main()
