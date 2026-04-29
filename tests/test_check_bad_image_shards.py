import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.check_bad_image_shards import check_one_mmap


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (10, 20, 30)).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _add_image(tar_writer: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar_writer.addfile(info, io.BytesIO(payload))


class CheckBadImageShardsTests(unittest.TestCase):
    def test_non_strict_flags_cv2_decode_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "shard-000000.tar"
            with tarfile.open(shard_path, "w") as tar_writer:
                _add_image(tar_writer, "clip_f000000.image.jpg", b"not a jpeg")

            _path, issues, images_checked = check_one_mmap(str(shard_path), strict=False)

            self.assertEqual(images_checked, 1)
            self.assertIn(("clip_f000000.image.jpg", "cv2_imdecode_none"), issues)

    def test_strict_flags_cv2_decoder_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "shard-000000.tar"
            payload = bytearray(_jpeg_bytes())
            for offset in range(200, 220):
                payload[offset] = 0
            with tarfile.open(shard_path, "w") as tar_writer:
                _add_image(tar_writer, "clip_f000000.image.jpg", bytes(payload))

            _path, non_strict_issues, images_checked = check_one_mmap(str(shard_path), strict=False)
            self.assertEqual(images_checked, 1)
            self.assertEqual(non_strict_issues, [])

            _path, strict_issues, images_checked = check_one_mmap(str(shard_path), strict=True)
            self.assertEqual(images_checked, 1)
            self.assertTrue(any(reason.startswith("decoder_stderr:") for _name, reason in strict_issues))


if __name__ == "__main__":
    unittest.main()
