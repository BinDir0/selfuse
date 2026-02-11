# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py

import asyncio
import http
import logging
import time
import traceback
from typing import Any, Dict

import websockets.asyncio.server as _server
import websockets.frames
import hydra
import torch
import torch.nn as nn

from . import msgpack_numpy


logger = logging.getLogger(__name__)


class RuntimeEngine:
    """
    The Runtime Engine. 
    It manages the execution context (Device, Autocast) for a Policy.
    """
    def __init__(self, policy: Any, device: torch.device, use_autocast: bool = True) -> None:
        self.policy = policy
        self.device = device
        self.use_autocast = use_autocast
        self.metadata = policy.metadata
        
        self.policy.to(device)

    def _move_to_device(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        return data.to(self.device) if torch.is_tensor(data) else data

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """The high-level entry point for inference."""
        prepared = self.policy.prepare_process(obs)
        inputs = self.policy.build_model_inputs(prepared)
        
        inputs = self._move_to_device(inputs)
        
        with torch.autocast(device_type=self.device.type, dtype=self.policy.dtype):
            pred_actions = self.policy(inputs)

        pred_actions = self.policy.post_process(pred_actions.cpu())
        
        return {"pred_actions": pred_actions}


class EnvWrapper:
    def __init__(
        self,
        policy: Any,
        image_key: str = "image",
        depth_key: str = "depth_image",
        intrinsic_key: str = "camera_intrinsics",
        instruction_key: str = "instruction",
        states_key: str = "states",
    ) -> None:
        self._policy = policy
        self._image_key = image_key
        self._depth_key = depth_key
        self._intrinsic_key = intrinsic_key
        self._instruction_key = instruction_key
        self._states_key = states_key
        self.metadata = getattr(policy, "metadata", {})

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attributes/methods to the wrapped policy.
        return getattr(self._policy, name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._policy)) | set(super().__dir__()))

    def infer(self, obs: dict) -> dict:
        mapped_obs = {
            "image": obs.get(self._image_key),
            "depth": obs.get(self._depth_key),
            "intrinsic": obs.get(self._intrinsic_key),
            "instruction": obs.get(self._instruction_key),
            "states": obs.get(self._states_key),
        }
        return self._policy.infer(mapped_obs)


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_engine(policy_cfg: Any, serving_cfg: Any) -> Any:
    policy = hydra.utils.instantiate(policy_cfg)
    device = _resolve_device(getattr(serving_cfg, "device", "auto"))
    use_autocast = bool(getattr(serving_cfg, "autocast", True))
    return RuntimeEngine(policy, device=device, use_autocast=use_autocast)


def create_env_wrapper(policy: Any, wrapper_cfg: Any) -> Any:
    return EnvWrapper(
        policy=policy,
        image_key=wrapper_cfg.image_key,
        depth_key=wrapper_cfg.depth_key,
        intrinsic_key=wrapper_cfg.intrinsic_key,
        instruction_key=wrapper_cfg.instruction_key,
        states_key=wrapper_cfg.states_key,
    )

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

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
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
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
    # Continue with the normal request handling.
    return None