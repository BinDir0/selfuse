import unittest
from types import SimpleNamespace

from lib.pipeline.orchestrator.pipeline import (
    _apply_single_video_native_fps_defaults,
    _resolve_effective_resume,
)


class ResolveEffectiveResumeTests(unittest.TestCase):
    def test_defaults_to_disabled_when_no_cli_and_no_config(self):
        self.assertFalse(_resolve_effective_resume(None, {}))

    def test_config_value_used_when_cli_unset(self):
        self.assertTrue(_resolve_effective_resume(None, {"resume": True}))
        self.assertFalse(_resolve_effective_resume(None, {"resume": False}))

    def test_cli_overrides_config(self):
        self.assertTrue(_resolve_effective_resume(True, {"resume": False}))
        self.assertFalse(_resolve_effective_resume(False, {"resume": True}))


class NativeBuildFpsDefaultsTests(unittest.TestCase):
    def _descriptor(self, fps):
        return SimpleNamespace(fps=fps)

    def test_no_build_section_still_persists_fps(self):
        # Regression: previously `config.get("build") or {}` dropped the
        # mutation when no build section existed.
        config = {"_meta": {"default_build_fps_from_video": True}}
        _apply_single_video_native_fps_defaults(
            config, descriptors=[self._descriptor(30.0)]
        )
        self.assertEqual(config["build"]["source_fps"], 30.0)
        self.assertEqual(config["build"]["target_fps"], 30.0)

    def test_existing_fps_not_overwritten(self):
        config = {
            "_meta": {"default_build_fps_from_video": True},
            "build": {"source_fps": 24.0},
        }
        _apply_single_video_native_fps_defaults(
            config, descriptors=[self._descriptor(30.0)]
        )
        self.assertEqual(config["build"]["source_fps"], 24.0)
        self.assertEqual(config["build"]["target_fps"], 30.0)

    def test_disabled_meta_is_noop(self):
        config = {"_meta": {}}
        _apply_single_video_native_fps_defaults(
            config, descriptors=[self._descriptor(30.0)]
        )
        self.assertNotIn("build", config)

    def test_prepared_payload_source(self):
        config = {"_meta": {"default_build_fps_from_video": True}}
        prepared = SimpleNamespace(payload={"fps": 25.0, "descriptor": None})
        _apply_single_video_native_fps_defaults(config, prepared=prepared)
        self.assertEqual(config["build"]["source_fps"], 25.0)

    def test_nonpositive_fps_ignored(self):
        config = {"_meta": {"default_build_fps_from_video": True}}
        _apply_single_video_native_fps_defaults(
            config, descriptors=[self._descriptor(0.0), self._descriptor(60.0)]
        )
        self.assertEqual(config["build"]["source_fps"], 60.0)


if __name__ == "__main__":
    unittest.main()
