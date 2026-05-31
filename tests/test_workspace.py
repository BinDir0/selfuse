"""Unit tests for lib.pipeline.workspace path resolution.

Pure-path logic, so these run without torch / cv2 / GPU.
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline import workspace


class _EnvGuard:
    """Context manager that isolates the workspace-related env vars."""

    _VARS = (
        workspace.ENV_OUTPUT_ROOT,
        workspace.ENV_LEGACY_SEQ_FOLDER,
        workspace.ENV_STAGE3_TMP_ROOT,
        workspace.ENV_BATCH_TMPDIR,
    )

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Descriptor:
    def __init__(self, seq_folder):
        self.seq_folder = seq_folder


class ResolveSeqFolderTests(unittest.TestCase):
    def test_descriptor_seq_folder_wins(self):
        with _EnvGuard():
            got = workspace.resolve_seq_folder(
                descriptor=_Descriptor("/explicit/seq"),
                video_path="/data/clip.mp4",
                output_root="/somewhere/else",
            )
            self.assertEqual(got, Path("/explicit/seq"))

    def test_default_is_consolidated_sibling_not_next_to_video(self):
        with _EnvGuard():
            got = workspace.resolve_seq_folder(video_path="/data/clip.mp4")
            # NOT /data/clip (the legacy next-to-video layout)
            self.assertNotEqual(got, Path("/data/clip"))
            self.assertEqual(
                got, Path("/data/clip.hawor_pipeline/stage_outputs/clip").resolve()
            )

    def test_explicit_output_root(self):
        with _EnvGuard():
            got = workspace.resolve_seq_folder(
                video_path="/data/clip.mp4", output_root="/scratch/run1"
            )
            self.assertEqual(got, Path("/scratch/run1/stage_outputs/clip"))

    def test_env_output_root(self):
        with _EnvGuard():
            os.environ[workspace.ENV_OUTPUT_ROOT] = "/scratch/shared"
            got = workspace.resolve_seq_folder(video_path="/data/clip.mp4")
            self.assertEqual(
                got, Path("/scratch/shared/clip.hawor_pipeline/stage_outputs/clip").resolve()
            )

    def test_legacy_flag_opt_in(self):
        with _EnvGuard():
            got = workspace.resolve_seq_folder(
                video_path="/data/clip.mp4", legacy_next_to_video=True
            )
            self.assertEqual(got, Path("/data/clip"))

    def test_legacy_env_opt_in(self):
        with _EnvGuard():
            os.environ[workspace.ENV_LEGACY_SEQ_FOLDER] = "1"
            got = workspace.resolve_seq_folder(video_path="/data/clip.mp4")
            self.assertEqual(got, Path("/data/clip"))

    def test_explicit_flag_overrides_env(self):
        with _EnvGuard():
            os.environ[workspace.ENV_LEGACY_SEQ_FOLDER] = "1"
            got = workspace.resolve_seq_folder(
                video_path="/data/clip.mp4", legacy_next_to_video=False
            )
            self.assertNotEqual(got, Path("/data/clip"))

    def test_requires_video_or_descriptor(self):
        with _EnvGuard():
            with self.assertRaises(ValueError):
                workspace.resolve_seq_folder()


class ResolveTmpRootTests(unittest.TestCase):
    def test_unset_required_raises(self):
        with _EnvGuard():
            with self.assertRaises(ValueError):
                workspace.resolve_tmp_root(args=None, required=True)

    def test_unset_not_required_returns_none(self):
        with _EnvGuard():
            self.assertIsNone(workspace.resolve_tmp_root(args=None, required=False))

    def test_env_batch_tmpdir(self):
        import tempfile

        with _EnvGuard(), tempfile.TemporaryDirectory() as tmp:
            os.environ[workspace.ENV_BATCH_TMPDIR] = tmp
            got = workspace.resolve_tmp_root(required=True)
            self.assertEqual(Path(got), Path(tmp).resolve())

    def test_stage3_tmp_root_env_takes_priority_over_batch_tmpdir(self):
        import tempfile

        with _EnvGuard(), tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            os.environ[workspace.ENV_STAGE3_TMP_ROOT] = a
            os.environ[workspace.ENV_BATCH_TMPDIR] = b
            got = workspace.resolve_tmp_root(required=True)
            self.assertEqual(Path(got), Path(a).resolve())

    def test_args_take_priority(self):
        import argparse
        import tempfile

        with _EnvGuard(), tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            os.environ[workspace.ENV_STAGE3_TMP_ROOT] = b
            args = argparse.Namespace(stage3_tmp_root=a)
            got = workspace.resolve_tmp_root(args, required=True)
            self.assertEqual(Path(got), Path(a).resolve())


class Stage3FrameCacheDirTests(unittest.TestCase):
    def test_deterministic(self):
        a = workspace.stage3_frame_cache_dir("/tmp/root", "/data/seqA", 0, 100)
        b = workspace.stage3_frame_cache_dir("/tmp/root", "/data/seqA", 0, 100)
        self.assertEqual(a, b)

    def test_same_stem_different_path_no_collision(self):
        a = workspace.stage3_frame_cache_dir("/tmp/root", "/data/x/clip", 0, 100)
        b = workspace.stage3_frame_cache_dir("/tmp/root", "/data/y/clip", 0, 100)
        self.assertNotEqual(a, b)


class WorkspaceLayoutTests(unittest.TestCase):
    def test_subdirs(self):
        with _EnvGuard():
            layout = workspace.WorkspaceLayout.build(
                video_path="/data/clip.mp4", output_root="/scratch/run1"
            )
            seq = Path("/scratch/run1/stage_outputs/clip")
            self.assertEqual(layout.seq_folder, seq)
            self.assertEqual(layout.slam_dir, seq / "SLAM")
            self.assertEqual(layout.cam_space_dir, seq / "cam_space")
            self.assertEqual(layout.tracks_dir(0, 100), seq / "tracks_0_100")
            self.assertEqual(layout.final_result_path, seq / "world_space_res.pth")
            self.assertEqual(layout.stage_done_marker("slam"), seq / ".slam.done")
            self.assertIsNone(layout.tmp_root)


if __name__ == "__main__":
    unittest.main()
