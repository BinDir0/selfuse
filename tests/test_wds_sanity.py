import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.pipeline.quality_metrics import parse_instruction_metadata
from lib.pipeline.wds_sanity import LOWDIM_DIMENSION_NAMES, analyze_webdataset


def _encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _add_tar_member(tar_writer: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar_writer.addfile(info, io.BytesIO(payload))


class WebdatasetSanityTests(unittest.TestCase):
    def test_parse_instruction_metadata_handles_sparse_and_string_inputs(self):
        sparse = parse_instruction_metadata({"instruction_num": 2, "instruction": ["a", " ", "b"]})
        self.assertEqual((sparse["instruction_num"], sparse["instructions"]), (2, ["a"]))
        self.assertTrue(sparse["instruction_num_mismatch"])
        string_input = parse_instruction_metadata({"instruction_num": 1, "instruction": "pick object"})
        self.assertEqual((string_input["instruction_num"], string_input["instructions"]), (1, ["pick object"]))
        missing = parse_instruction_metadata({"instruction_num": 0, "instruction": ["a"]})
        self.assertEqual((missing["instruction_num"], missing["instructions"]), (0, []))
        self.assertTrue(missing["missing_instruction"])

    def test_dimension_names_cover_full_lowdim(self):
        self.assertEqual(len(LOWDIM_DIMENSION_NAMES), 116)
        self.assertEqual(LOWDIM_DIMENSION_NAMES[0], "state.left_wrist.x")
        self.assertEqual(LOWDIM_DIMENSION_NAMES[-1], "camera_intrinsic.cy")

    def test_analyze_webdataset_reports_nonfinite_and_missing_instruction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_path = root / "shard-000000.tar"

            valid_lowdim = np.arange(116, dtype=np.float32)
            valid_lowdim[112:116] = np.asarray([100.0, 100.0, 192.0, 192.0], dtype=np.float32)
            nonfinite_lowdim = valid_lowdim.copy()
            nonfinite_lowdim[5] = np.nan

            meta_ok = {
                "clip_id": "clip_a",
                "instruction": ["pick up tool"],
                "instruction_num": 1,
                "presence": 3,
            }
            meta_missing_instruction = {
                "clip_id": "clip_b",
                "instruction": [],
                "instruction_num": 0,
                "presence": 3,
            }

            with tarfile.open(shard_path, "w") as tar_writer:
                _add_tar_member(tar_writer, "clip_a_ep000000_f000000.image.jpg", b"not-a-real-image")
                _add_tar_member(tar_writer, "clip_a_ep000000_f000000.lowdim.npy", _encode_npy(valid_lowdim))
                _add_tar_member(tar_writer, "clip_a_ep000000_f000000.meta.json", json.dumps(meta_ok).encode("utf-8"))

                _add_tar_member(tar_writer, "clip_b_ep000001_f000000.image.jpg", b"not-a-real-image")
                _add_tar_member(tar_writer, "clip_b_ep000001_f000000.lowdim.npy", _encode_npy(valid_lowdim + 1.0))
                _add_tar_member(
                    tar_writer,
                    "clip_b_ep000001_f000000.meta.json",
                    json.dumps(meta_missing_instruction).encode("utf-8"),
                )

                _add_tar_member(tar_writer, "clip_b_ep000001_f000001.image.jpg", b"not-a-real-image")
                _add_tar_member(tar_writer, "clip_b_ep000001_f000001.lowdim.npy", _encode_npy(nonfinite_lowdim))
                _add_tar_member(
                    tar_writer,
                    "clip_b_ep000001_f000001.meta.json",
                    json.dumps(meta_missing_instruction).encode("utf-8"),
                )

            report = analyze_webdataset(
                source_shard_dir=str(root),
                decode_images=False,
                render_dir=None,
                render_episodes=0,
            )

            self.assertEqual(report["summary"]["samples_total"], 3)
            self.assertEqual(report["summary"]["episodes_total"], 2)
            self.assertEqual(report["checks"]["missing_instruction_frames"], 2)
            self.assertEqual(report["checks"]["nonfinite_lowdim_frames"], 1)
            self.assertEqual(report["summary"]["hard_filter_drop_episodes"], 1)
            self.assertEqual(report["hard_filter_reason_counts"]["missing_instruction"], 1)
            self.assertEqual(report["hard_filter_reason_counts"]["nonfinite_lowdim"], 1)
            self.assertEqual(report["summary"]["valid_lowdim_frames"], 2)
            self.assertEqual(report["lowdim_stats"]["count"], 2)
            self.assertAlmostEqual(report["lowdim_stats"]["dimensions"][0]["min"], 0.0)
            self.assertAlmostEqual(report["lowdim_stats"]["dimensions"][0]["max"], 1.0)
            self.assertAlmostEqual(report["lowdim_stats"]["dimensions"][0]["avg"], 0.5)
            self.assertGreaterEqual(report["summary"]["issue_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
