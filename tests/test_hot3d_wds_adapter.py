import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from lib.pipeline.clip_manifest import build_manifest_records_from_descriptors, write_clip_manifest
from lib.pipeline.datasets.hot3d_wds import HOT3DWDSDatasetAdapter
from scripts.extract_hot3d_wds_annotations import main as extract_annotations_main


def _npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _add_bytes(tar_writer, name, payload):
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar_writer.addfile(info, io.BytesIO(payload))


def _write_sample(tar_writer, sample_key, instruction):
    _add_bytes(tar_writer, f"{sample_key}.image.jpg", b"\xff\xd8\xff\xd9")
    _add_bytes(
        tar_writer,
        f"{sample_key}.meta.json",
        json.dumps(
            {
                "dataset_name": "hot3d",
                "episode_index": 0,
                "instruction": instruction,
                "instruction_num": len(instruction),
                "presence": 3,
            }
        ).encode("utf-8"),
    )
    _add_bytes(tar_writer, f"{sample_key}.bbox.npy", _npy_bytes(np.zeros((2, 4), dtype=np.float32)))
    _add_bytes(tar_writer, f"{sample_key}.lowdim.npy", _npy_bytes(np.zeros((116,), dtype=np.float32)))
    _add_bytes(tar_writer, f"{sample_key}.mano.npy", _npy_bytes(np.zeros((2, 55), dtype=np.float32)))


class HOT3DWDSDatasetAdapterTests(unittest.TestCase):
    def test_builds_episode_descriptors_and_extracts_annotations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shards"
            shard_dir.mkdir()
            shard_path = shard_dir / "shard-w0000-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                _write_sample(tar_writer, "hot3d_ep000000_f00000", [])
                _write_sample(tar_writer, "hot3d_ep000000_f00001", ["pick up the object", "pick up the object with the right hand"])
                _write_sample(tar_writer, "hot3d_ep000001_f00000", ["open the drawer"])

            adapter = HOT3DWDSDatasetAdapter()
            descriptors = list(
                adapter.build_descriptors(
                    dataset_cfg={"source_id": "hot3d"},
                    adapter_cfg={"shard_dir": str(shard_dir), "seq_folder_root": str(root / "outputs")},
                    paths_cfg={},
                )
            )

            self.assertEqual([descriptor.clip_id for descriptor in descriptors], ["hot3d_ep000000", "hot3d_ep000001"])
            self.assertEqual(descriptors[0].frame_count, 2)
            self.assertEqual(descriptors[0].frame_names[0], "hot3d_ep000000_f00000.image.jpg")
            self.assertEqual(descriptors[0].extra["frame_start_idx"], 0)
            self.assertEqual(descriptors[0].extra["frame_end_idx"], 1)

            manifest_path = root / "manifest.jsonl"
            records = build_manifest_records_from_descriptors(descriptors, source_id="hot3d", split="train")
            write_clip_manifest(records, manifest_path)

            annotation_root = root / "annotations"
            original_argv = list(sys.argv)
            try:
                sys.argv = [
                    "extract_hot3d_wds_annotations.py",
                    "--descriptor_manifest",
                    str(manifest_path),
                    "--annotation_root",
                    str(annotation_root),
                ]
                with redirect_stdout(io.StringIO()):
                    extract_annotations_main()
            finally:
                sys.argv = original_argv

            annotation = json.loads((annotation_root / "hot3d_ep000000.annotation.json").read_text(encoding="utf-8"))
            self.assertEqual(annotation["instruction_num"], 2)
            self.assertEqual(annotation["instruction"][0], "pick up the object")


if __name__ == "__main__":
    unittest.main()
