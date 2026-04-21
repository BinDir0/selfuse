import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.filter_webdataset import analyze_shard, rewrite_shard
from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar
from lib.pipeline.quality_metrics import decide_clip_quality


def _lowdim_bytes(*, nonfinite: bool = False) -> bytes:
    lowdim = np.zeros((116,), dtype=np.float32)
    lowdim[96:112] = np.eye(4, dtype=np.float32).reshape(-1)
    if nonfinite:
        lowdim[0] = np.nan
    buffer = io.BytesIO()
    np.save(buffer, lowdim, allow_pickle=False)
    return buffer.getvalue()


def _meta_bytes(clip_id: str, instruction, instruction_num: int | None = None) -> bytes:
    meta = {
        "clip_id": clip_id,
        "presence": 3,
        "instruction": instruction,
        "instruction_num": len(instruction) if instruction_num is None else instruction_num,
    }
    return json.dumps(meta).encode("utf-8")


def _add_sample(
    tar_writer: tarfile.TarFile,
    sample_key: str,
    clip_id: str,
    instruction,
    *,
    instruction_num: int | None = None,
    nonfinite: bool = False,
) -> None:
    write_sample_to_tar(
        tar_writer,
        sample_key,
        b"jpg",
        _lowdim_bytes(nonfinite=nonfinite),
        _meta_bytes(clip_id, instruction, instruction_num),
    )


def _criteria() -> dict:
    return {
        "min_instruction_num": None,
        "min_presence_ratio": None,
        "max_hand_translation_step": None,
        "max_camera_translation_step": None,
        "max_camera_rotation_step": None,
        "max_camera_space_wrist_abs": None,
        "max_camera_space_hand_abs": None,
        "camera_space_wrist_bounds": None,
        "camera_space_hand_bounds": None,
        "camera_space_axis_abs_cap": None,
    }


class FilterWebdatasetTests(unittest.TestCase):
    def test_rewrite_drops_whole_clip_for_any_bad_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            shard_path = tmpdir_path / "shard-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                _add_sample(tar_writer, "clip_missing_f000000", "clip_missing", ["pick"])
                _add_sample(tar_writer, "clip_missing_f000001", "clip_missing", [], instruction_num=0)
                _add_sample(tar_writer, "clip_empty_f000000", "clip_empty", ["   "], instruction_num=1)
                _add_sample(tar_writer, "clip_nan_f000000", "clip_nan", ["pick"], nonfinite=True)
                _add_sample(tar_writer, "clip_good_f000000", "clip_good", ["pick"])
                _add_sample(tar_writer, "clip_good_f000001", "clip_good", ["place"])

            analysis = analyze_shard(
                str(shard_path),
                compute_motion_metrics=False,
                compute_camera_space_metrics=False,
            )
            decisions = {}
            for item in analysis["clip_metrics"]:
                keep, reasons = decide_clip_quality(item["metrics"], _criteria())
                decisions[item["clip_id"]] = (keep, reasons)

            self.assertFalse(decisions["clip_missing"][0])
            self.assertIn("missing_instruction_frame", decisions["clip_missing"][1])
            self.assertFalse(decisions["clip_empty"][0])
            self.assertIn("empty_instruction_frame", decisions["clip_empty"][1])
            self.assertFalse(decisions["clip_nan"][0])
            self.assertIn("nonfinite_lowdim", decisions["clip_nan"][1])
            self.assertTrue(decisions["clip_good"][0])

            keep_by_clip = {clip_id: keep for clip_id, (keep, _) in decisions.items()}
            output_dir = tmpdir_path / "filtered"
            result = rewrite_shard(str(shard_path), str(output_dir), keep_by_clip)

            self.assertEqual(result["clips_written"], 1)
            self.assertEqual(result["frames_written"], 2)
            output_samples = list(iter_shard_samples(str(output_dir / shard_path.name)))
            self.assertEqual([sample["key"] for sample in output_samples], ["clip_good_f000000", "clip_good_f000001"])


if __name__ == "__main__":
    unittest.main()
