import json
import tempfile
import tarfile
import unittest
from pathlib import Path

import numpy as np

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar
from scripts.repair_nonfinite_lowdim_wds import (
    REPAIR_FLAG,
    encode_npy,
    rewrite_shard,
)


class RepairNonfiniteLowdimWdsTests(unittest.TestCase):
    def test_rewrite_shard_replaces_nonfinite_lowdim_values_and_marks_meta(self):
        lowdim = np.arange(116, dtype=np.float32)
        lowdim[0] = np.nan
        lowdim[18] = np.inf
        lowdim[66] = -np.inf
        meta_bytes = json.dumps({"clip_id": "clip"}).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dst_dir = root / "dst"
            dst_dir.mkdir()
            shard_path = dst_dir / "shard-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                write_sample_to_tar(
                    tar_writer,
                    "sample-000000",
                    b"jpg",
                    encode_npy(lowdim),
                    meta_bytes,
                )

            stats = rewrite_shard(
                src_shard=str(root / "src" / shard_path.name),
                dst_dir=dst_dir,
                sample_keys={"sample-000000"},
                seed="test-seed",
                replacement_min=0.5,
                replacement_max=0.5,
                dry_run=False,
                overwrite=True,
            )

            self.assertEqual(stats["values_replaced"], 3)
            self.assertEqual(stats["sample_keys_repaired"], ["sample-000000"])
            sample = next(iter_shard_samples(str(shard_path)))
            repaired = np.load(__import__("io").BytesIO(sample["lowdim_bytes"]), allow_pickle=False)
            self.assertTrue(np.isfinite(repaired).all())
            self.assertEqual(float(repaired[0]), 0.5)
            self.assertEqual(float(repaired[18]), 0.5)
            self.assertEqual(float(repaired[66]), 0.5)

            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            self.assertIn(REPAIR_FLAG, meta["dirty_ablation_flags"])
            self.assertEqual(meta["nonfinite_lowdim_repair"]["values_replaced"], 3)


if __name__ == "__main__":
    unittest.main()
