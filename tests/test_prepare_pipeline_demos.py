from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.ops.prepare_pipeline_demos import parse_render_modes, peak_score, resolve_seq_folder, stable_inverse_score
from lib.pipeline.viewer_backend import scan_sample_summaries


class PreparePipelineDemosTests(unittest.TestCase):
    def test_parse_render_modes_deduplicates_and_normalizes(self):
        self.assertEqual(parse_render_modes(" keypoint,mano,keypoint "), ["keypoint", "mano"])

    def test_peak_score_peaks_inside_interval(self):
        self.assertEqual(peak_score(0.5, low=0.0, peak=0.5, high=1.0), 1.0)
        self.assertEqual(peak_score(-1.0, low=0.0, peak=0.5, high=1.0), 0.0)
        self.assertEqual(peak_score(2.0, low=0.0, peak=0.5, high=1.0), 0.0)

    def test_stable_inverse_score_is_monotonic(self):
        self.assertGreater(stable_inverse_score(0.0, 1.0), stable_inverse_score(1.0, 1.0))
        self.assertGreater(stable_inverse_score(1.0, 1.0), stable_inverse_score(2.0, 1.0))

    def test_resolve_seq_folder_prefers_manifest_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_root = mock.Mock()
            resolved = resolve_seq_folder(
                "f238_w016_v00006_i001",
                seq_folder_lookup={"f238_w016_v00006_i001": tmpdir},
                buildai_processed_root=processed_root,
            )
            self.assertEqual(resolved, tmpdir)

    def test_resolve_seq_folder_can_autodiscover_buildai_processed_root(self):
        clip_id = "f238_w016_v00006_i001"
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_folder = Path(tmpdir) / "factory_238" / "worker_016" / "processed" / clip_id
            seq_folder.mkdir(parents=True)
            resolved = resolve_seq_folder(
                clip_id,
                seq_folder_lookup={},
                buildai_processed_root=Path(tmpdir),
            )
            self.assertEqual(resolved, str(seq_folder.resolve()))

    def test_resolve_seq_folder_returns_none_for_unknown_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = resolve_seq_folder(
                "hot3d_ep000001_part000",
                seq_folder_lookup={},
                buildai_processed_root=Path(tmpdir),
            )
            self.assertIsNone(resolved)

    def test_scan_sample_summaries_stops_when_episode_limit_is_reached(self):
        headers = [
            {"episode_key": "ep1", "key": "sample1"},
            {"episode_key": "ep1", "key": "sample2"},
            {"episode_key": "ep2", "key": "sample3"},
            {"episode_key": "ep2", "key": "sample4"},
            {"episode_key": "ep3", "key": "sample5"},
        ]

        def build_summary(header, shard_path, sample_id):
            return mock.Mock(
                episode_key=header["episode_key"],
                key=header["key"],
                clip_id=header["episode_key"],
                instruction_preview="",
                shard_name="shard-000000.tar",
                presence=3,
            )

        with mock.patch("lib.pipeline.viewer_backend.iter_shard_sample_headers", return_value=headers), mock.patch(
            "lib.pipeline.viewer_backend.build_sample_summary_from_header",
            side_effect=build_summary,
        ):
            summaries = scan_sample_summaries(
                ["dummy.tar"],
                sample_limit=None,
                episode_limit=2,
                filter_key="",
                filter_presence=None,
            )

        self.assertEqual([item.episode_key for item in summaries], ["ep1", "ep1", "ep2", "ep2"])


if __name__ == "__main__":
    unittest.main()
