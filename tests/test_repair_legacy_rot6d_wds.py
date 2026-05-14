import unittest
import json
import tempfile
import tarfile
from pathlib import Path

import numpy as np

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar
from scripts.repair_legacy_rot6d_wds import (
    REPAIR_MARKER_KEY,
    REPAIR_MARKER_VALUE,
    dirty_state_action_scale_lowdim,
    encode_npy,
    mark_meta_repaired,
    repair_shard,
    repair_legacy_rot6d_vector,
    repair_lowdim,
)


class RepairLegacyRot6DWdsTests(unittest.TestCase):
    def test_repair_legacy_rot6d_vector_transposes_3x2_flatten(self):
        old = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        repaired = repair_legacy_rot6d_vector(old)
        np.testing.assert_array_equal(
            repaired,
            np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], dtype=np.float32),
        )

    def test_repair_lowdim_updates_all_wrist_state_and_action_rot6d_slots(self):
        lowdim = np.arange(116, dtype=np.float32)
        repaired = repair_lowdim(lowdim)

        for start in (6, 12, 54, 60):
            expected = lowdim[start:start + 6].reshape(3, 2).T.reshape(6)
            np.testing.assert_array_equal(repaired[start:start + 6], expected)

        untouched = np.ones(116, dtype=bool)
        for start in (6, 12, 54, 60):
            untouched[start:start + 6] = False
        np.testing.assert_array_equal(repaired[untouched], lowdim[untouched])

    def test_dirty_state_action_scale_updates_hand_and_wrist_translation_only(self):
        lowdim = np.arange(1, 117, dtype=np.float32)
        scaled = dirty_state_action_scale_lowdim(
            lowdim,
            seed="test-seed",
            sample_key="sample-000000",
            scale_min=2.0,
            scale_max=2.0,
        )

        expected = lowdim.copy()
        for start, end in ((0, 6), (18, 48), (48, 54), (66, 96)):
            expected[start:end] *= 2.0
        np.testing.assert_array_equal(scaled, expected)
        for start, end in ((6, 18), (54, 66)):
            np.testing.assert_array_equal(scaled[start:end], lowdim[start:end])

    def test_mark_meta_repaired_records_layout_and_prevents_double_repair(self):
        repaired = mark_meta_repaired(json.dumps({"clip_id": "clip"}).encode("utf-8"))
        meta = json.loads(repaired.decode("utf-8"))
        self.assertEqual(meta[REPAIR_MARKER_KEY], REPAIR_MARKER_VALUE)
        self.assertEqual(meta["lowdim_rot6d_layout"], "column_major_2x3")

        with self.assertRaisesRegex(ValueError, "already has"):
            mark_meta_repaired(repaired)

    def test_repair_shard_writes_progress_and_repaired_output(self):
        lowdim = np.zeros(116, dtype=np.float32)
        lowdim[6:12] = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        meta_bytes = json.dumps({"clip_id": "clip"}).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            out_dir = root / "out"
            src_dir.mkdir()
            shard_path = src_dir / "shard-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                write_sample_to_tar(
                    tar_writer,
                    "sample-000000",
                    b"jpg",
                    encode_npy(lowdim),
                    meta_bytes,
                )

            progress_path = root / "progress.jsonl"
            result = repair_shard(
                str(shard_path),
                str(out_dir),
                resume=True,
                dry_run=False,
                progress_out=str(progress_path),
                progress_interval=1,
                progress_run_id="test-run",
            )

            self.assertEqual(result["status"], "rewritten")
            progress_events = [
                json.loads(line)["event"]
                for line in progress_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(progress_events, ["start", "sample", "finish"])

            sample = next(iter_shard_samples(str(out_dir / shard_path.name)))
            repaired_lowdim = np.load(__import__("io").BytesIO(sample["lowdim_bytes"]), allow_pickle=False)
            np.testing.assert_array_equal(
                repaired_lowdim[6:12],
                np.array([1, 3, 5, 2, 4, 6], dtype=np.float32),
            )
            repaired_meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            self.assertEqual(repaired_meta[REPAIR_MARKER_KEY], REPAIR_MARKER_VALUE)

    def test_repair_shard_can_leave_rot6d_legacy_and_replace_instructions(self):
        lowdim = np.zeros(116, dtype=np.float32)
        lowdim[6:12] = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        meta_bytes = json.dumps(
            {"clip_id": "clip", "instruction": ["pick"], "instruction_num": 1}
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            out_dir = root / "out"
            src_dir.mkdir()
            shard_path = src_dir / "shard-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                write_sample_to_tar(
                    tar_writer,
                    "sample-000000",
                    b"jpg",
                    encode_npy(lowdim),
                    meta_bytes,
                )

            result = repair_shard(
                str(shard_path),
                str(out_dir),
                resume=True,
                dry_run=False,
                keep_legacy_rot6d_episode_fraction=1.0,
                dirty_instruction_episode_fraction=1.0,
                dirty_instruction_mode="generic",
                generic_instruction="do something useful",
            )

            self.assertEqual(result["legacy_rot6d_samples"], 1)
            self.assertEqual(result["dirty_instruction_samples"], 1)
            sample = next(iter_shard_samples(str(out_dir / shard_path.name)))
            output_lowdim = np.load(__import__("io").BytesIO(sample["lowdim_bytes"]), allow_pickle=False)
            np.testing.assert_array_equal(output_lowdim[6:12], lowdim[6:12])
            output_meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            self.assertEqual(output_meta[REPAIR_MARKER_KEY], "skipped_for_dirty_ablation")
            self.assertEqual(output_meta["lowdim_rot6d_layout"], "legacy_3x2_row_major")
            self.assertEqual(output_meta["instruction"], ["do something useful"])
            self.assertEqual(output_meta["instruction_num"], 1)
            self.assertIn("legacy_rot6d", output_meta["dirty_ablation_flags"])
            self.assertIn("generic_instruction", output_meta["dirty_ablation_flags"])

    def test_repair_shard_can_scale_hand_and_wrist_translation_dirty(self):
        lowdim = np.arange(1, 117, dtype=np.float32)
        for start in (6, 12, 54, 60):
            lowdim[start:start + 6] = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        meta_bytes = json.dumps({"clip_id": "clip", "instruction": ["pick"], "instruction_num": 1}).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            out_dir = root / "out"
            src_dir.mkdir()
            shard_path = src_dir / "shard-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                write_sample_to_tar(
                    tar_writer,
                    "sample-000000",
                    b"jpg",
                    encode_npy(lowdim),
                    meta_bytes,
                )

            result = repair_shard(
                str(shard_path),
                str(out_dir),
                resume=True,
                dry_run=False,
                dirty_state_action_scale_episode_fraction=1.0,
                dirty_state_action_scale_min=2.0,
                dirty_state_action_scale_max=2.0,
            )

            self.assertEqual(result["dirty_state_action_scale_samples"], 1)
            sample = next(iter_shard_samples(str(out_dir / shard_path.name)))
            output_lowdim = np.load(__import__("io").BytesIO(sample["lowdim_bytes"]), allow_pickle=False)

            repaired_without_scale = repair_lowdim(lowdim)
            expected = repaired_without_scale.copy()
            for start, end in ((0, 6), (18, 48), (48, 54), (66, 96)):
                expected[start:end] *= 2.0
            np.testing.assert_array_equal(output_lowdim, expected)
            for start, end in ((6, 18), (54, 66)):
                np.testing.assert_array_equal(output_lowdim[start:end], repaired_without_scale[start:end])

            output_meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            self.assertIn("state_action_scale", output_meta["dirty_ablation_flags"])
            self.assertEqual(output_meta["dirty_state_action_scale_range"], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
