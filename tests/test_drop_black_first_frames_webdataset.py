import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar
from scripts.drop_black_first_frames_webdataset import audit_zero_intrinsic_frames, drop_black_first_frames


def _jpeg_bytes(value: int) -> bytes:
    image = Image.new("RGB", (8, 8), color=(value, value, value))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _npy_bytes(*, zero_intrinsic: bool = False, partial_zero_intrinsic: bool = False) -> bytes:
    buffer = io.BytesIO()
    lowdim = np.zeros((116,), dtype=np.float32)
    if not zero_intrinsic:
        lowdim[112:116] = np.asarray([100.0, 100.0, 4.0, 4.0], dtype=np.float32)
    if partial_zero_intrinsic:
        lowdim[112:116] = np.asarray([100.0, 0.0, 4.0, 4.0], dtype=np.float32)
    np.save(buffer, lowdim, allow_pickle=False)
    return buffer.getvalue()


def _meta_bytes(clip_id: str) -> bytes:
    return json.dumps({"clip_id": clip_id, "instruction": ["pick"], "instruction_num": 1}).encode("utf-8")


def _add_sample(
    tar_writer: tarfile.TarFile,
    sample_key: str,
    clip_id: str,
    image_value: int,
    *,
    zero_intrinsic: bool = False,
    partial_zero_intrinsic: bool = False,
) -> None:
    write_sample_to_tar(
        tar_writer,
        sample_key,
        _jpeg_bytes(image_value),
        _npy_bytes(zero_intrinsic=zero_intrinsic, partial_zero_intrinsic=partial_zero_intrinsic),
        _meta_bytes(clip_id),
    )


class DropBlackFirstFramesWebdatasetTests(unittest.TestCase):
    def test_audits_zero_intrinsic_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            source_dir.mkdir()

            with tarfile.open(source_dir / "shard-000000.tar", "w") as tar_writer:
                _add_sample(tar_writer, "clip_a_f000000", "clip_a", 80, zero_intrinsic=True)
                _add_sample(tar_writer, "clip_a_f000001", "clip_a", 80)
                _add_sample(tar_writer, "clip_a_f000002", "clip_a", 80, partial_zero_intrinsic=True)

            report = audit_zero_intrinsic_frames(
                source_dir,
                workers=1,
                detail_limit=100,
                progress_every_shards=0,
            )

            summary = report["summary"]
            self.assertEqual(summary["source_shards"], 1)
            self.assertEqual(summary["lowdim_frames_total"], 3)
            self.assertEqual(summary["any_zero_intrinsic_frames"], 2)
            self.assertEqual(summary["all_zero_intrinsic_frames"], 1)
            self.assertEqual(summary["zero_fx_frames"], 1)
            self.assertEqual(summary["zero_fy_frames"], 2)
            self.assertEqual(summary["zero_cx_frames"], 1)
            self.assertEqual(summary["zero_cy_frames"], 1)
            self.assertEqual(summary["decode_errors"], 0)
            self.assertEqual(len(report["details"]), 2)

    def test_drops_only_black_first_frame_per_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            source_dir.mkdir()
            output_dir = root / "output"
            shard_path = source_dir / "shard-000000.tar"

            with tarfile.open(shard_path, "w") as tar_writer:
                _add_sample(tar_writer, "clip_a_f000000", "clip_a", 80, zero_intrinsic=True)
                _add_sample(tar_writer, "clip_a_f000001", "clip_a", 80)
                _add_sample(tar_writer, "clip_b_f000000", "clip_b", 80)
                _add_sample(tar_writer, "clip_b_f000001", "clip_b", 80, zero_intrinsic=True)

            report = drop_black_first_frames(
                source_dir,
                output_dir,
                dry_run=False,
                max_mean=2.0,
                max_pixel=10,
                min_dark_ratio=0.999,
                detail_limit=100,
            )

            self.assertEqual(report["summary"]["episodes_seen"], 2)
            self.assertEqual(report["summary"]["black_first_frames_dropped"], 1)
            self.assertEqual(report["summary"]["frames_written"], 3)

            output_samples = list(iter_shard_samples(str(output_dir / "shard-000000.tar")))
            self.assertEqual(
                [sample["key"] for sample in output_samples],
                ["clip_a_f000000", "clip_b_f000000", "clip_b_f000001"],
            )

    def test_parallel_drops_only_global_first_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            source_dir.mkdir()
            output_dir = root / "output"

            with tarfile.open(source_dir / "shard-000000.tar", "w") as tar_writer:
                _add_sample(tar_writer, "clip_a_f000000", "clip_a", 80, zero_intrinsic=True)
                _add_sample(tar_writer, "clip_b_f000001", "clip_b", 80, zero_intrinsic=True)

            with tarfile.open(source_dir / "shard-000001.tar", "w") as tar_writer:
                _add_sample(tar_writer, "clip_a_f000001", "clip_a", 80)
                _add_sample(tar_writer, "clip_b_f000000", "clip_b", 80)

            with tarfile.open(source_dir / "shard-000002.tar", "w") as tar_writer:
                _add_sample(tar_writer, "clip_c_f000000", "clip_c", 80)

            report = drop_black_first_frames(
                source_dir,
                output_dir,
                dry_run=False,
                max_mean=2.0,
                max_pixel=10,
                min_dark_ratio=0.999,
                detail_limit=100,
                workers=2,
            )

            self.assertEqual(report["summary"]["episodes_seen"], 3)
            self.assertEqual(report["summary"]["black_first_frames_dropped"], 1)
            self.assertEqual(report["summary"]["frames_written"], 4)
            self.assertTrue(report["shards"]["shard-000002.tar"].get("hardlinked"))
            self.assertEqual(
                (source_dir / "shard-000002.tar").stat().st_ino,
                (output_dir / "shard-000002.tar").stat().st_ino,
            )

            output_samples = []
            for shard_path in sorted(output_dir.glob("*.tar")):
                output_samples.extend(sample["key"] for sample in iter_shard_samples(str(shard_path)))
            self.assertEqual(
                output_samples,
                ["clip_b_f000001", "clip_a_f000000", "clip_b_f000000", "clip_c_f000000"],
            )

    def test_drops_when_any_first_frame_intrinsic_value_is_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            source_dir.mkdir()
            output_dir = root / "output"

            with tarfile.open(source_dir / "shard-000000.tar", "w") as tar_writer:
                _add_sample(tar_writer, "clip_partial_f000000", "clip_partial", 80, partial_zero_intrinsic=True)
                _add_sample(tar_writer, "clip_partial_f000001", "clip_partial", 80)

            report = drop_black_first_frames(
                source_dir,
                output_dir,
                dry_run=False,
                max_mean=2.0,
                max_pixel=10,
                min_dark_ratio=0.999,
                detail_limit=100,
            )

            self.assertEqual(report["summary"]["black_first_frames_dropped"], 1)
            output_samples = list(iter_shard_samples(str(output_dir / "shard-000000.tar")))
            self.assertEqual([sample["key"] for sample in output_samples], ["clip_partial_f000000"])


if __name__ == "__main__":
    unittest.main()
