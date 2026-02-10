# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py

import asyncio
import http
import logging
import time
import traceback
from typing import Any

import hydra
import torch
from src.serving.msgpack_numpy import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class DevicePolicyWrapper:
    def __init__(self, policy: Any, device: torch.device, use_autocast: bool = True) -> None:
        self._policy = policy
        self._device = device
        self._use_autocast = use_autocast
        self._autocast_dtype = getattr(policy, "dtype", torch.float32)
        self.metadata = getattr(policy, "metadata", {})
        self._move_model_to_device()

    def _move_model_to_device(self) -> None:
        if hasattr(self._policy, "model") and isinstance(self._policy.model, torch.nn.Module):
            self._policy.model.to(self._device)
        elif isinstance(self._policy, torch.nn.Module):
            self._policy.to(self._device)

    def _move_inputs_to_device(self, inputs: dict) -> dict:
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self._device)
        return inputs

    def _run_infer(self, inputs: dict) -> dict:
        if self._policy.mode == "flow":
            pred_actions = self._policy.model("infer_action", inputs)
        elif self._policy.mode == "ar":
            pred_actions = self._policy.model(
                "infer_vla",
                inputs,
                max_new_tokens=self._policy.ar_max_new_tokens,
                temperature=self._policy.ar_temperature,
                cfg=self._policy.ar_cfg,
            )
        else:
            raise ValueError(f"Unsupported inference mode: {self._policy.mode}")
        pred_actions = self._policy._unnormalize_actions(pred_actions)
        return {"pred_actions": pred_actions}

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        prepared = self._policy._prepare_processor_inputs(obs)
        inputs = self._policy._build_model_inputs(prepared)
        inputs = self._move_inputs_to_device(inputs)

        if self._use_autocast and self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
                return self._run_infer(inputs)
        return self._run_infer(inputs)


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_policy(policy_cfg: Any, serving_cfg: Any) -> Any:
    policy = hydra.utils.instantiate(policy_cfg)
    device = _resolve_device(getattr(serving_cfg, "device", "auto"))
    use_autocast = bool(getattr(serving_cfg, "autocast", True))
    return DevicePolicyWrapper(policy, device=device, use_autocast=use_autocast)


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