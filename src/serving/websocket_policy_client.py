import argparse
import asyncio
import time
from typing import Any, Dict, Tuple

import numpy as np
import websockets.asyncio.client as _client

from . import msgpack_numpy


def _parse_shape(value: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must be a comma-separated list of ints")
    try:
        shape = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be a comma-separated list of ints") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dims must be > 0")
    return shape


def _random_obs(
    image_shape: Tuple[int, ...],
    depth_shape: Tuple[int, ...],
    intrinsic_shape: Tuple[int, ...],
    state_horizon: int,
    state_dim: int,
    instruction: str,
) -> Dict[str, Any]:
    return {
        "image": np.random.rand(*image_shape).astype(np.float32),
        "depth_image": np.random.rand(*depth_shape).astype(np.float32),
        "camera_intrinsics": np.random.rand(*intrinsic_shape).astype(np.float32),
        "instruction": instruction,
        "states": np.random.rand(state_horizon, state_dim).astype(np.float32),
    }


async def _run_client(args: argparse.Namespace) -> None:
    uri = f"ws://{args.host}:{args.port}"
    packer = msgpack_numpy.Packer()
    image_shape = _parse_shape(args.image_shape)
    depth_shape = _parse_shape(args.depth_shape)
    intrinsic_shape = _parse_shape(args.intrinsic_shape)

    async with _client.connect(uri, max_size=None, compression=None) as websocket:
        metadata = msgpack_numpy.unpackb(await websocket.recv())
        print("server metadata:", metadata)

        for idx in range(args.num_requests):
            obs = _random_obs(
                image_shape=image_shape,
                depth_shape=depth_shape,
                intrinsic_shape=intrinsic_shape,
                state_horizon=args.state_horizon,
                state_dim=args.state_dim,
                instruction=args.instruction,
            )
            start_time = time.monotonic()
            await websocket.send(packer.pack(obs))
            response = msgpack_numpy.unpackb(await websocket.recv())
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            timing = response.get("server_timing", {})
            infer_ms = timing.get("infer_ms")
            if infer_ms is None:
                print(f"[{idx}] infer_ms missing, response keys: {list(response.keys())}")
            else:
                print(f"[{idx}] infer_ms={infer_ms:.3f}ms, rtt_ms={elapsed_ms:.3f}ms")

            if args.sleep_ms > 0:
                await asyncio.sleep(args.sleep_ms / 1000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WebSocket client that sends random observations and prints inference time."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, required=True, help="Server port.")
    parser.add_argument("--num-requests", type=int, default=5, help="Number of requests to send.")
    parser.add_argument("--sleep-ms", type=int, default=200, help="Sleep between requests in ms.")
    parser.add_argument("--image-shape", default="1,224,224,3", help="Image shape, e.g. 1,224,224,3")
    parser.add_argument("--depth-shape", default="1,224,224,1", help="Depth shape, e.g. 1,224,224,1")
    parser.add_argument("--intrinsic-shape", default="3,3", help="Intrinsic shape, e.g. 3,3")
    parser.add_argument("--state-horizon", type=int, default=16, help="State horizon.")
    parser.add_argument("--state-dim", type=int, default=48, help="State vector dim.")
    parser.add_argument("--instruction", default="random instruction", help="Instruction string.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_client(args))


if __name__ == "__main__":
    main()

