"""Resolve repo-level Python runtimes for the dataset pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HAWOR_ENV = "hawor"
DEFAULT_ANY4D_ENV = "any4d"


@dataclass(frozen=True)
class PipelineRuntimes:
    hawor_python: str | None
    slam_python: str | None
    source: dict[str, str]


def _compatible_existing_python_path(raw_path: str | None, *, runtime_name: str) -> str | None:
    if raw_path is None:
        return None
    path_text = str(raw_path)
    candidate = Path(path_text)
    if candidate.exists():
        return path_text

    fixed_text = path_text.replace("/envs/any4/", "/envs/any4d/")
    if fixed_text != path_text and Path(fixed_text).exists():
        print(
            f"[runtime] {runtime_name} python not found at {path_text}; using compatible fallback {fixed_text}",
            flush=True,
        )
        return fixed_text
    return path_text


def _conda_like_commands() -> list[str]:
    return [cmd for cmd in ("conda", "mamba", "micromamba") if shutil.which(cmd)]


def _parse_env_list_json(stdout: str) -> list[Path]:
    payload = json.loads(stdout)
    envs = payload.get("envs") if isinstance(payload, dict) else None
    if not isinstance(envs, list):
        return []
    return [Path(str(path)).expanduser() for path in envs if str(path).strip()]


def _parse_env_list_text(stdout: str) -> list[Path]:
    envs = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in line.replace("*", " ").split() if part]
        path_token = next((part for part in reversed(parts) if "/" in part), None)
        if path_token:
            envs.append(Path(path_token).expanduser())
    return envs


def _list_conda_env_paths(command_runner=subprocess.run) -> list[Path]:
    paths: list[Path] = []
    seen = set()
    for command in _conda_like_commands():
        json_result = command_runner(
            [command, "env", "list", "--json"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        parsed = []
        if getattr(json_result, "returncode", 1) == 0:
            try:
                parsed = _parse_env_list_json(json_result.stdout)
            except Exception:
                parsed = []
        if not parsed:
            text_result = command_runner(
                [command, "env", "list"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if getattr(text_result, "returncode", 1) == 0:
                parsed = _parse_env_list_text(text_result.stdout)
        for path in parsed:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def _env_path_matches(path: Path, env_name: str) -> bool:
    if path.name == env_name:
        return True
    parts = path.parts
    return len(parts) >= 2 and parts[-2] == "envs" and parts[-1] == env_name


def resolve_conda_env_python(env_name: str, *, command_runner=subprocess.run) -> str:
    for env_path in _list_conda_env_paths(command_runner=command_runner):
        if not _env_path_matches(env_path, env_name):
            continue
        python_path = env_path / "bin" / "python"
        if python_path.exists():
            return str(python_path)
        raise FileNotFoundError(f"Conda env `{env_name}` was found at {env_path}, but {python_path} does not exist.")

    commands = ", ".join(_conda_like_commands()) or "conda/mamba/micromamba"
    raise FileNotFoundError(
        f"Required conda env `{env_name}` was not found via `{commands} env list`.\n"
        f"Install or create it first, then verify with `conda env list | grep {env_name}`."
    )


def resolve_pipeline_runtimes(
    runtimes_cfg: dict | None,
    *,
    require_hawor: bool,
    require_slam: bool,
    command_runner=subprocess.run,
) -> PipelineRuntimes:
    runtimes_cfg = dict(runtimes_cfg or {})
    source = {}

    hawor_python = _compatible_existing_python_path(runtimes_cfg.get("hawor_python"), runtime_name="hawor")
    if hawor_python:
        source["hawor"] = "config"
    elif require_hawor:
        hawor_python = resolve_conda_env_python(DEFAULT_HAWOR_ENV, command_runner=command_runner)
        source["hawor"] = f"conda:{DEFAULT_HAWOR_ENV}"

    slam_python = _compatible_existing_python_path(
        runtimes_cfg.get("slam_python") or runtimes_cfg.get("any4d_python"),
        runtime_name="any4d",
    )
    if slam_python:
        source["slam"] = "config"
    elif require_slam:
        slam_python = resolve_conda_env_python(DEFAULT_ANY4D_ENV, command_runner=command_runner)
        source["slam"] = f"conda:{DEFAULT_ANY4D_ENV}"
    elif hawor_python:
        slam_python = hawor_python
        source["slam"] = source.get("hawor", "config")

    return PipelineRuntimes(
        hawor_python=hawor_python,
        slam_python=slam_python,
        source=source,
    )
