# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py

import asyncio
from contextlib import nullcontext
import http
import logging
import pathlib
import time
import traceback
from typing import Any, Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from . import msgpack_numpy
from .serving_recorder import ConnectionRecorder, ServingRecorder


logger = logging.getLogger(__name__)


class RuntimeEngine:
    """
    The Runtime Engine.
    It manages the execution context (Device, Autocast) for a Policy.
    """

    def __init__(
        self,
        policy: Any,
        device: torch.device,
        use_autocast: bool,
        warmup_image_shape: tuple[int, int, int],
        warmup_depth_shape: tuple[int, int, int],
        warmup_intrinsic: np.ndarray,
    ) -> None:
        self.policy = policy
        self.device = device
        self.use_autocast = use_autocast
        self.metadata = policy.metadata
        self.warmup_image_shape = tuple(int(x) for x in warmup_image_shape)
        self.warmup_depth_shape = tuple(int(x) for x in warmup_depth_shape)
        self.warmup_intrinsic = np.asarray(warmup_intrinsic, dtype=np.float64)
        self._profiler = None
        self._profile_steps = 0
        self._profile_max_steps = 0
        self._profile_output_dir: pathlib.Path | None = None

        self.policy.to(device)
        if hasattr(self.policy, "maybe_compile_model"):
            self.policy.maybe_compile_model()

    def _move_to_device(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        return data.to(self.device) if torch.is_tensor(data) else data

    def _autocast_context(self):
        if not self.use_autocast:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.policy.dtype)

    def enable_profiling(self, output_dir: str | pathlib.Path, steps: int, skip_first: int) -> None:
        if steps <= 0:
            return
        output_dir = pathlib.Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=steps, repeat=1, skip_first=skip_first),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self._profiler.__enter__()
        self._profile_steps = 0
        self._profile_max_steps = skip_first + steps
        self._profile_output_dir = output_dir
        logger.info(
            "Enabled inference profiler for %d requests after skipping %d requests. Output dir: %s",
            steps,
            skip_first,
            output_dir,
        )

    def _step_profiler(self) -> None:
        if self._profiler is None:
            return
        self._profiler.step()
        self._profile_steps += 1
        if self._profile_steps >= self._profile_max_steps:
            self._finalize_profiler()

    def _finalize_profiler(self) -> None:
        if self._profiler is None or self._profile_output_dir is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sort_by = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        summary = self._profiler.key_averages().table(sort_by=sort_by, row_limit=30)
        summary_path = self._profile_output_dir / "summary.txt"
        trace_path = self._profile_output_dir / "trace.json"
        summary_path.write_text(summary, encoding="utf-8")
        self._profiler.export_chrome_trace(str(trace_path))
        self._profiler.__exit__(None, None, None)
        logger.info("Saved inference profiler summary to %s", summary_path)
        logger.info("Saved inference profiler trace to %s", trace_path)
        self._profiler = None
        self._profile_output_dir = None

    def _get_shape_meta(self) -> Dict[str, Any]:
        if hasattr(self.policy, "shape_meta"):
            return self.policy.shape_meta
        if hasattr(self.policy, "model") and hasattr(self.policy.model, "shape_meta"):
            return self.policy.model.shape_meta
        raise AttributeError("Policy does not expose model.shape_meta for warmup")

    def _build_dummy_obs(self, instruction: str) -> Dict[str, Any]:
        shape_meta = self._get_shape_meta()
        rgb_meta = shape_meta["obs"]["rgb"]
        state_meta = shape_meta["obs"]["state"]

        image = np.zeros((rgb_meta["horizon"], *self.warmup_image_shape), dtype=np.uint8)
        states = np.zeros((state_meta["horizon"], state_meta["shape"][0]), dtype=np.float32)
        intrinsic = self.warmup_intrinsic.copy()

        obs = {
            "image": image,
            "intrinsic": intrinsic,
            "instruction": instruction,
            "states": states,
        }

        depth_meta = shape_meta["obs"].get("depth")
        if depth_meta is not None:
            obs["depth"] = np.zeros((depth_meta["horizon"], *self.warmup_depth_shape), dtype=np.uint16)

        return obs

    def warmup(self, warmup_iters: int, instruction: str) -> None:
        if warmup_iters <= 0:
            return

        dummy_obs = self._build_dummy_obs(instruction=instruction)
        logger.info("Running %d warmup inference iterations", warmup_iters)
        start_time = time.monotonic()
        for _ in range(warmup_iters):
            self.infer(dummy_obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logger.info("Warmup finished in %.3f ms", (time.monotonic() - start_time) * 1000.0)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """The high-level entry point for inference."""
        inputs = self.policy.prepare_process(obs)
        inputs = self._move_to_device(inputs)

        with self._autocast_context(), torch.inference_mode():
            pred_actions = self.policy(inputs)
        pred_actions = self.policy.post_process(pred_actions.cpu())
        output = {"pred_actions": pred_actions.cpu().float().numpy()[0]}
        self._step_profiler()
        return output


class EnvWrapper:
    def __init__(
        self,
        policy: Any,
        image_key: str = "image",
        depth_key: str = "depth_image",
        intrinsic_key: str = "camera_intrinsics",
        instruction_key: str = "instruction",
        states_key: str = "states",
        prev_action_chunk_key: str = "action_rtc",
    ) -> None:
        self.policy = policy
        self.image_key = image_key
        self.depth_key = depth_key
        self.intrinsic_key = intrinsic_key
        self.instruction_key = instruction_key
        self.states_key = states_key
        self.prev_action_chunk_key = prev_action_chunk_key
        self.metadata = getattr(policy, "metadata", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self.policy)) | set(super().__dir__()))

    def infer(self, obs: dict) -> dict:
        mapped_obs = {
            "image": obs.get(self.image_key),
            "depth": obs.get(self.depth_key),
            "intrinsic": obs.get(self.intrinsic_key),
            "instruction": obs.get(self.instruction_key),
            "states": obs.get(self.states_key),
            # RTC condition: executed action prefix (None on first step)
            "prev_action_chunk": obs.get(self.prev_action_chunk_key),
        }
        return self.policy.infer(mapped_obs)


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_engine(policy_cfg: Any, serving_cfg: Any) -> Any:
    return RuntimeEngine(
        hydra.utils.instantiate(policy_cfg),
        device=_resolve_device(serving_cfg.device),
        use_autocast=serving_cfg.autocast,
        warmup_image_shape=tuple(serving_cfg.warmup_image_shape),
        warmup_depth_shape=tuple(serving_cfg.warmup_depth_shape),
        warmup_intrinsic=np.asarray(serving_cfg.warmup_intrinsic, dtype=np.float64),
    )


def create_env_wrapper(policy: Any, wrapper_cfg: Any) -> Any:
    return EnvWrapper(
        policy=policy,
        image_key=wrapper_cfg.image_key,
        depth_key=wrapper_cfg.depth_key,
        intrinsic_key=wrapper_cfg.intrinsic_key,
        instruction_key=wrapper_cfg.instruction_key,
        states_key=wrapper_cfg.states_key,
        prev_action_chunk_key=wrapper_cfg.prev_action_chunk_key,
    )


class WebsocketPolicyServer:
    """Serve a policy with the websocket protocol."""

    def __init__(
        self,
        policy: Any,
        recorder: ServingRecorder | None,
        log_obs_details: bool,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._recorder = recorder
        self._log_obs_details = log_obs_details
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @staticmethod
    def _format_obs_details(obs: Dict[str, Any]) -> str:
        import numpy as np

        lines = ["\n观测数据详情:"]
        lines.append(f"  总字段数: {len(obs)}")
        lines.append(f"  字段列表: {list(obs.keys())}")
        lines.append("-" * 80)

        for key, value in obs.items():
            lines.append(f"\n字段: '{key}'")

            if value is None:
                lines.append("  类型: None")
                continue

            value_type = type(value).__name__
            lines.append(f"  Python类型: {value_type}")

            if hasattr(value, "shape"):
                lines.append(f"  Shape: {value.shape}")

                if hasattr(value, "dtype"):
                    lines.append(f"  Dtype: {value.dtype}")

                if hasattr(value, "nbytes"):
                    size_bytes = value.nbytes
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} bytes"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.2f} KB"
                    else:
                        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                    lines.append(f"  内存大小: {size_str}")

                try:
                    if hasattr(value, "size") and value.size == 0:
                        lines.append("  状态: 空数组")
                    elif np.issubdtype(value.dtype, np.number):
                        lines.append("  数值统计:")
                        lines.append(f"    - Min: {float(value.min()):.6f}")
                        lines.append(f"    - Max: {float(value.max()):.6f}")
                        lines.append(f"    - Mean: {float(value.mean()):.6f}")
                        if hasattr(value, "std"):
                            lines.append(f"    - Std: {float(value.std()):.6f}")
                    else:
                        lines.append("  数据类型: 非数值型")
                except Exception as exc:
                    lines.append(f"  统计信息: 无法计算 ({str(exc)})")

            elif isinstance(value, (list, tuple)):
                lines.append(f"  长度: {len(value)}")
                if len(value) > 0:
                    lines.append(f"  首元素类型: {type(value[0]).__name__}")

            elif isinstance(value, str):
                lines.append(f"  长度: {len(value)} 字符")
                preview = value[:50] + "..." if len(value) > 50 else value
                lines.append(f"  内容预览: {preview}")

            elif isinstance(value, (int, float)):
                lines.append(f"  值: {value}")

            else:
                lines.append(f"  描述: {str(value)[:100]}")

        lines.append("-" * 80)
        return "\n".join(lines)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        connection_recorder: ConnectionRecorder | None = None

        if self._recorder is not None:
            try:
                connection_recorder = self._recorder.open_connection(websocket.remote_address)
            except Exception:
                logger.exception("Failed to initialize serving recorder for %s", websocket.remote_address)

        await websocket.send(packer.pack(self._metadata))

        prev_send_time = None
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                recv_start = time.monotonic()
                raw_obs = await websocket.recv()
                recv_wait_time = time.monotonic() - recv_start
                obs = msgpack_numpy.unpackb(raw_obs)

                if self._log_obs_details:
                    logger.info("Received observation from %s%s", websocket.remote_address, self._format_obs_details(obs))

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                action["server_timing"] = {
                    "recv_wait_ms": recv_wait_time * 1000,
                    "infer_ms": infer_time * 1000,
                }
                if prev_send_time is not None:
                    action["server_timing"]["prev_send_ms"] = prev_send_time * 1000
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                pack_start = time.monotonic()
                packed_action = packer.pack(action)
                pack_time = time.monotonic() - pack_start
                action["server_timing"]["pack_ms"] = pack_time * 1000
                action["server_timing"]["pre_send_total_ms"] = (time.monotonic() - start_time) * 1000

                record_time = 0.0
                if connection_recorder is not None:
                    try:
                        record_start = time.monotonic()
                        connection_recorder.record(obs, action)
                        record_time = time.monotonic() - record_start
                    except Exception:
                        logger.exception("Failed to record request for %s", websocket.remote_address)

                action["server_timing"]["record_ms"] = record_time * 1000

                packed_action = packer.pack(action)
                send_start = time.monotonic()
                await websocket.send(packed_action)
                prev_send_time = time.monotonic() - send_start
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise



def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
