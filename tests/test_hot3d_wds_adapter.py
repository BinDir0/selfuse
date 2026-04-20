import io
import importlib.util
import json
import sys
import tarfile
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

if importlib.util.find_spec("tqdm") is None and "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **_kwargs: iterable if iterable is not None else []
    sys.modules["tqdm"] = tqdm_module

if importlib.util.find_spec("torch") is None and "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = type("Tensor", (), {})
    torch_module.device = lambda value: value
    sys.modules["torch"] = torch_module

if importlib.util.find_spec("joblib") is None and "joblib" not in sys.modules:
    joblib_module = types.ModuleType("joblib")
    joblib_module.load = lambda *_args, **_kwargs: None
    joblib_module.dump = lambda *_args, **_kwargs: None
    sys.modules["joblib"] = joblib_module

if importlib.util.find_spec("scipy") is None and "scipy" not in sys.modules:
    scipy_module = types.ModuleType("scipy")
    spatial_module = types.ModuleType("scipy.spatial")
    transform_module = types.ModuleType("scipy.spatial.transform")

    class _FakeRotation:
        @classmethod
        def from_quat(cls, _quat):
            return cls()

        def as_matrix(self):
            return np.eye(3, dtype=np.float32)

    class _FakeSlerp:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, _times):
            return _FakeRotation()

    transform_module.Rotation = _FakeRotation
    transform_module.Slerp = _FakeSlerp
    scipy_module.spatial = spatial_module
    spatial_module.transform = transform_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.spatial"] = spatial_module
    sys.modules["scipy.spatial.transform"] = transform_module

from lib.pipeline.clip_manifest import build_manifest_records_from_descriptors, write_clip_manifest
from lib.pipeline.datasets.hot3d_wds import HOT3DWDSDatasetAdapter
from lib.pipeline.exporters.manifest_build.writer import build_manifest_meta_prefix
from lib.pipeline.exporters.manifest_vla import load_descriptor_episode_features, prepare_manifest_record_for_build
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
    lowdim = np.zeros((116,), dtype=np.float32)
    lowdim[96:112] = np.eye(4, dtype=np.float32).reshape(-1)
    lowdim[112:116] = np.array([500.0, 500.0, 704.0, 704.0], dtype=np.float32)
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
    _add_bytes(tar_writer, f"{sample_key}.lowdim.npy", _npy_bytes(lowdim))
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
            self.assertEqual(descriptors[0].extra["native_feature_source"], "wds_lowdim_mano_v1")

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

            episode, error_code = prepare_manifest_record_for_build(
                records[0],
                require_annotation=True,
                annotation_root=str(annotation_root),
                annotation_suffix=".annotation.json",
                source_fps=30.0,
                target_fps=30.0,
                interpolate_labels=False,
            )
            self.assertIsNone(error_code)
            self.assertEqual(episode["num_valid_frames"], 2)

            episode_data = load_descriptor_episode_features(
                episode,
                None,
                None,
                None,
                feature_cache_dir=None,
                mano_dir=None,
                source_fps=30.0,
                target_fps=30.0,
                interpolate_labels=False,
            )
            self.assertEqual(episode_data["lowdim_all"].shape, (2, 116))
            self.assertEqual(episode_data["mano_all"].shape, (2, 2, 55))
            self.assertTrue(np.all(episode_data["lowdim_all"][:, 112:114] > 0))

            meta_prefix = build_manifest_meta_prefix({**episode, "episode_index": 0})
            meta = json.loads(meta_prefix + b"3}")
            self.assertEqual(meta["lowdim_schema"], "hot3d_wrist_world_v1")
            self.assertEqual(meta["native_feature_source"], "wds_lowdim_mano_v1")
            self.assertEqual(meta["mano_schema"], "hot3d_mano_2x55_v1")


if __name__ == "__main__":
    unittest.main()
