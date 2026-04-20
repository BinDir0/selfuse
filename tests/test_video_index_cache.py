import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **_kwargs: iterable if iterable is not None else []
    sys.modules["tqdm"] = tqdm_module

from lib.pipeline.video_index import CLIP_INDEX_FORMAT_VERSION, load_clip_frame_offsets, load_or_build_index


class VideoIndexCacheTests(unittest.TestCase):
    def test_manifest_index_reuses_clip_cache_without_offset_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            factory_dir = Path(tmpdir)
            shard_path = factory_dir / "shard-000000.tar"
            with tarfile.open(shard_path, "w"):
                pass

            index_payload = {
                "format_version": CLIP_INDEX_FORMAT_VERSION,
                "clips": {
                    "clip_a": {
                        "shard": shard_path.name,
                        "video_name": "clip_a",
                        "frame_count": 1,
                        "frame_ext": ".jpg",
                        "frame_index_width": 5,
                        "frame_start_idx": 0,
                    }
                },
                "shards": [shard_path.name],
                "num_videos": 1,
                "num_shards": 1,
            }
            (factory_dir / "_clip_index.json").write_text(json.dumps(index_payload), encoding="utf-8")

            with patch("lib.pipeline.video_index.build_video_index") as build_video_index:
                loaded = load_or_build_index(str(factory_dir))

            build_video_index.assert_not_called()
            self.assertEqual(loaded["clips"]["clip_a"]["frame_count"], 1)
            self.assertIsNone(load_clip_frame_offsets(str(factory_dir), "clip_a"))


if __name__ == "__main__":
    unittest.main()
