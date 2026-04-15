import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from lib.pipeline.clip_manifest import build_manifest_records_from_descriptors, load_clip_manifest, write_clip_manifest
from lib.pipeline.datasets.descriptors import ClipDescriptor

sys.modules.setdefault("joblib", mock.Mock())

from lib.pipeline.orchestrator import pipeline as pipeline_module


class OrchestratorPartialInferTests(unittest.TestCase):
    def test_pipeline_uses_config_run_tag_when_cli_run_tag_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log_root = tmp / "logs"
            config_run_tag = "config-run-tag"
            config_path = tmp / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "run_tag": config_run_tag,
                        "dataset": {"adapter": "buildai"},
                        "paths": {
                            "log_root": str(log_root),
                            "final_dataset_root": str(tmp / "final_dataset"),
                        },
                        "runtimes": {
                            "hawor_python": "/usr/bin/python3",
                            "slam_python": "/usr/bin/python3",
                        },
                        "infer": {
                            "common": {"resume": True},
                        },
                        "build": {},
                    }
                ),
                encoding="utf-8",
            )

            descriptor = ClipDescriptor.from_image_sequence(
                clip_id="clip_a",
                clip_name="clip_a",
                root_dir=str(tmp / "seqs"),
                seq_folder=str(tmp / "seqs" / "clip_a"),
                frame_dir=str(tmp / "seqs" / "clip_a" / "frames"),
                frame_names=["000000.jpg", "000001.jpg"],
            )
            adapter = mock.Mock()
            adapter.prepare.return_value = None
            adapter.build_descriptors.return_value = [descriptor]

            args = argparse.Namespace(
                config=str(config_path),
                stages="prepare",
                run_tag=None,
                resume=False,
            )

            with mock.patch.object(pipeline_module, "get_dataset_adapter", return_value=adapter):
                pipeline_module.run_pipeline(args)

            expected_manifest = log_root / config_run_tag / "clip_manifest.jsonl"
            self.assertTrue(expected_manifest.exists())

    def test_infer_requires_existing_manifest_for_infer_only_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log_root = tmp / "logs"
            config_path = tmp / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"adapter": "buildai"},
                        "paths": {
                            "log_root": str(log_root),
                            "final_dataset_root": str(tmp / "final_dataset"),
                        },
                        "runtimes": {
                            "hawor_python": "/usr/bin/python3",
                            "slam_python": "/usr/bin/python3",
                        },
                        "infer": {
                            "common": {"resume": True},
                        },
                        "build": {},
                    }
                ),
                encoding="utf-8",
            )

            args = argparse.Namespace(
                config=str(config_path),
                stages="infer",
                run_tag="missing-manifest",
                resume=False,
            )

            with mock.patch.object(pipeline_module, "get_dataset_adapter", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "requires descriptor manifest"):
                    pipeline_module.run_pipeline(args)

    def test_partial_infer_failure_narrows_manifest_for_downstream_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log_root = tmp / "logs"
            run_tag = "partial-infer"
            run_dir = log_root / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            descriptors = []
            for clip_id in ("clip_a", "clip_b", "clip_c"):
                seq_folder = tmp / "seqs" / clip_id
                seq_folder.mkdir(parents=True, exist_ok=True)
                descriptors.append(
                    ClipDescriptor.from_image_sequence(
                        clip_id=clip_id,
                        clip_name=clip_id,
                        root_dir=str(tmp / "seqs"),
                        seq_folder=str(seq_folder),
                        frame_dir=str(seq_folder / "frames"),
                        frame_names=["000000.jpg", "000001.jpg"],
                    )
                )
            manifest_records = build_manifest_records_from_descriptors(
                descriptors,
                source_id="test",
                split="train",
            )
            manifest_path = run_dir / "clip_manifest.jsonl"
            write_clip_manifest(manifest_records, manifest_path)

            config_path = tmp / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"adapter": "buildai"},
                        "paths": {
                            "log_root": str(log_root),
                            "final_dataset_root": str(tmp / "final_dataset"),
                        },
                        "runtimes": {
                            "hawor_python": "/usr/bin/python3",
                            "slam_python": "/usr/bin/python3",
                        },
                        "infer": {
                            "common": {"resume": True},
                        },
                        "build": {},
                    }
                ),
                encoding="utf-8",
            )

            build_manifests = []

            def fake_stream_command(name, cmd, log_path, *, cwd=None, env=None, raise_on_error=True):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("$ fake\n", encoding="utf-8")
                status_path = run_dir / "status.json"

                if name == "detect_motion":
                    current_manifest = Path(cmd[cmd.index("--descriptor_manifest") + 1])
                    self.assertEqual(current_manifest.resolve(), manifest_path.resolve())
                    tasks = {
                        "clip_a": {"stage_status": {"motion": "completed"}},
                        "clip_b": {"stage_status": {"motion": "completed"}},
                        "clip_c": {"stage_status": {"motion": "failed"}},
                    }
                    status_path.write_text(
                        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    for clip_id in ("clip_a", "clip_b"):
                        (tmp / "seqs" / clip_id / ".motion.done").touch()
                    return 1

                if name == "slam":
                    current_manifest = Path(cmd[cmd.index("--descriptor_manifest") + 1])
                    self.assertEqual(
                        current_manifest.name,
                        "clip_manifest.motion.completed.jsonl",
                    )
                    self.assertEqual(
                        [record.clip_id for record in load_clip_manifest(current_manifest)],
                        ["clip_a", "clip_b"],
                    )
                    tasks = {
                        "clip_a": {"stage_status": {"slam": "completed"}},
                        "clip_b": {"stage_status": {"slam": "failed"}},
                    }
                    status_path.write_text(
                        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (tmp / "seqs" / "clip_a" / ".slam.done").touch()
                    return 1

                if name == "infiller":
                    current_manifest = Path(cmd[cmd.index("--descriptor_manifest") + 1])
                    self.assertEqual(
                        current_manifest.name,
                        "clip_manifest.motion.completed.slam.completed.jsonl",
                    )
                    self.assertEqual(
                        [record.clip_id for record in load_clip_manifest(current_manifest)],
                        ["clip_a"],
                    )
                    tasks = {
                        "clip_a": {"stage_status": {"infiller": "completed"}},
                    }
                    status_path.write_text(
                        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (tmp / "seqs" / "clip_a" / ".infiller.done").touch()
                    return 0

                if name == "build":
                    current_manifest = Path(cmd[cmd.index("--descriptor_manifest") + 1])
                    build_manifests.append(current_manifest)
                    self.assertEqual(
                        current_manifest.name,
                        "clip_manifest.motion.completed.slam.completed.infiller.completed.jsonl",
                    )
                    self.assertEqual(
                        [record.clip_id for record in load_clip_manifest(current_manifest)],
                        ["clip_a"],
                    )
                    return 0

                raise AssertionError(f"Unexpected stage: {name}")

            args = argparse.Namespace(
                config=str(config_path),
                stages="infer,build",
                run_tag=run_tag,
                resume=True,
            )

            with mock.patch.object(pipeline_module, "get_dataset_adapter", return_value=object()), mock.patch.object(
                pipeline_module,
                "stream_command",
                side_effect=fake_stream_command,
            ):
                pipeline_module.run_pipeline(args)

            self.assertEqual(len(build_manifests), 1)
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["active_manifest_path"].endswith(".infiller.completed.jsonl"))
            self.assertEqual(summary["infer_stage_manifests"]["detect_motion"]["completed"], 2)
            self.assertEqual(summary["infer_stage_manifests"]["slam"]["completed"], 1)
            self.assertEqual(summary["infer_stage_manifests"]["infiller"]["completed"], 1)


if __name__ == "__main__":
    unittest.main()
