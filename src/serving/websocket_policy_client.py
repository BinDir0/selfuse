import argparse
import asyncio
import time
from typing import Any, Dict, Tuple

import numpy as np
import websockets.asyncio.client as _client

from . import msgpack_numpy


DEFAULT_CAMERA_INTRINSICS = np.array(
    [
        [388.0, 0.0, 320.0],
        [0.0, 388.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# Built-in presets so a single flag flips all defaults at once. Add new entries
# here when a recurring server config (resolution / horizon / camera setup)
# emerges; CLI users only need to remember --preset NAME.
PRESETS: Dict[str, Dict[str, Any]] = {
    "prod-480x640-dual": {
        # Matches the production inference config:
        #   target_image_size = [480, 640], image_horizon = 6,
        #   camera_setup_mode = "both", state_horizon = 6, state_dim = 48,
        #   action_horizon = 32, action_dim = 48, RTC enabled.
        "image_shape": "6,480,640,3",
        "depth_shape": "6,480,640,1",
        "intrinsic_shape": "3,3",
        "camera_setup": "both",
        "image_mode": "rgb",
        "state_horizon": 6,
        "state_dim": 48,
        "action_horizon": 32,
        "action_dim": 48,
        "rtc_delay": 32,
        "fixed_state_horizon": True,
    },
    "prod-480x640-single": {
        "image_shape": "6,480,640,3",
        "depth_shape": "6,480,640,1",
        "intrinsic_shape": "3,3",
        "camera_setup": "single",
        "image_mode": "rgb",
        "state_horizon": 6,
        "state_dim": 48,
        "action_horizon": 32,
        "action_dim": 48,
        "rtc_delay": 32,
        "fixed_state_horizon": True,
    },
}


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



def _random_rgb_image(shape: Tuple[int, ...]) -> np.ndarray:
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)



def _random_depth_image(shape: Tuple[int, ...]) -> np.ndarray:
    depth = np.random.randint(300, 1500, size=shape, dtype=np.uint16)
    invalid_mask = np.random.rand(*shape[:-1], 1) < 0.01
    saturated_mask = np.random.rand(*shape[:-1], 1) < 0.01
    depth = np.where(invalid_mask, 0, depth)
    depth = np.where(saturated_mask, np.iinfo(np.uint16).max, depth)
    return depth.astype(np.uint16)



def _camera_intrinsics(shape: Tuple[int, ...]) -> np.ndarray:
    if shape == (3, 3):
        return DEFAULT_CAMERA_INTRINSICS.copy()
    return np.random.rand(*shape).astype(np.float64)


def _camera_extrinsics() -> np.ndarray:
    # Keep a simple identity world->cam transform for smoke tests.
    return np.eye(4, dtype=np.float64)



def _random_states(state_horizon: int, state_dim: int) -> np.ndarray:
    states = np.random.normal(loc=0.0, scale=0.35, size=(state_horizon, state_dim))
    return np.clip(states, -1.0, 1.0).astype(np.float32)



def _state_horizon_candidates(state_horizon: int) -> Tuple[int, ...]:
    candidates = {
        1,
        max(1, state_horizon // 4),
        max(1, state_horizon // 2),
        state_horizon,
    }
    return tuple(sorted(candidates))


def _build_action_rtc(rtc_delay: int, action_dim: int, action_horizon: int) -> np.ndarray:
    """Construct an action_rtc payload of shape [delay, action_dim].

    Server treats len(action_rtc) as inference_delay and pads to action_horizon
    internally, so any 1 <= delay <= action_horizon is a valid value to test
    the RTC code path.
    """
    if rtc_delay < 1:
        raise ValueError(f"rtc_delay must be >= 1, got {rtc_delay}")
    if rtc_delay > action_horizon:
        raise ValueError(
            f"rtc_delay={rtc_delay} > action_horizon={action_horizon}; server cannot accept this."
        )
    return np.zeros((rtc_delay, action_dim), dtype=np.float32)


def _random_obs(
    image_shape: Tuple[int, ...],
    depth_shape: Tuple[int, ...],
    intrinsic_shape: Tuple[int, ...],
    state_horizon: int,
    state_dim: int,
    instruction: str,
    camera_setup: str,
    image_mode: str,
    include_camera_extrinsics: bool,
    rtc_delay: int | None,
    action_dim: int,
    action_horizon: int,
    fixed_state_horizon: bool,
) -> Tuple[Dict[str, Any], int]:
    if fixed_state_horizon:
        sampled_state_horizon = state_horizon
    else:
        sampled_state_horizon = int(np.random.choice(_state_horizon_candidates(state_horizon)))

    states = _random_states(sampled_state_horizon, state_dim)
    action_rtc = (
        _build_action_rtc(rtc_delay, action_dim, action_horizon)
        if rtc_delay is not None
        else None
    )

    if camera_setup == "both":
        head_rgb = _random_rgb_image(image_shape)
        chest_rgb = _random_rgb_image(image_shape)
        obs: Dict[str, Any] = {
            "image": {
                "head": head_rgb,
                "chest": chest_rgb,
            },
            "camera_intrinsics": {
                "head": _camera_intrinsics(intrinsic_shape),
                "chest": _camera_intrinsics(intrinsic_shape),
            },
            "instruction": instruction,
            "states": states,
            "action_rtc": action_rtc,
        }
        if image_mode == "rgbd":
            obs["depth_image"] = {
                "head": _random_depth_image(depth_shape),
                "chest": _random_depth_image(depth_shape),
            }
        if include_camera_extrinsics:
            obs["camera_extrinsics"] = {
                "head": _camera_extrinsics(),
                "chest": _camera_extrinsics(),
            }
        return obs, sampled_state_horizon

    # single camera legacy payload
    obs = {
        "image": _random_rgb_image(image_shape),
        "camera_intrinsics": _camera_intrinsics(intrinsic_shape),
        "instruction": instruction,
        "states": states,
        "action_rtc": action_rtc,
    }
    if image_mode == "rgbd":
        obs["depth_image"] = _random_depth_image(depth_shape)
    if include_camera_extrinsics:
        obs["camera_extrinsics"] = _camera_extrinsics()
    return obs, sampled_state_horizon


async def _run_client(args: argparse.Namespace) -> None:
    uri = f"ws://{args.host}:{args.port}"
    packer = msgpack_numpy.Packer()
    image_shape = _parse_shape(args.image_shape)
    depth_shape = _parse_shape(args.depth_shape)
    intrinsic_shape = _parse_shape(args.intrinsic_shape)
    rtc_delay: int | None = None if args.rtc_delay <= 0 else int(args.rtc_delay)

    print(
        f"[client] preset={args.preset or 'none'} camera_setup={args.camera_setup} "
        f"image_mode={args.image_mode} image_shape={image_shape} "
        f"state_horizon={args.state_horizon}({'fixed' if args.fixed_state_horizon else 'sampled'}) "
        f"state_dim={args.state_dim} rtc_delay={rtc_delay} "
        f"action_horizon={args.action_horizon} action_dim={args.action_dim}"
    )

    async with _client.connect(uri, max_size=None, compression=None) as websocket:
        metadata = msgpack_numpy.unpackb(await websocket.recv())
        print("server metadata:", metadata)

        for idx in range(args.num_requests):
            obs, sampled_state_horizon = _random_obs(
                image_shape=image_shape,
                depth_shape=depth_shape,
                intrinsic_shape=intrinsic_shape,
                state_horizon=args.state_horizon,
                state_dim=args.state_dim,
                instruction=args.instruction,
                camera_setup=args.camera_setup,
                image_mode=args.image_mode,
                include_camera_extrinsics=args.include_camera_extrinsics,
                rtc_delay=rtc_delay,
                action_dim=args.action_dim,
                action_horizon=args.action_horizon,
                fixed_state_horizon=args.fixed_state_horizon,
            )
            start_time = time.monotonic()
            await websocket.send(packer.pack(obs))
            response = msgpack_numpy.unpackb(await websocket.recv())
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            timing = response.get("server_timing", {})
            infer_ms = timing.get("infer_ms")
            if infer_ms is None:
                print(
                    f"[{idx}] state_horizon={sampled_state_horizon}, infer_ms missing, response keys: {list(response.keys())}"
                )
            else:
                print(
                    f"[{idx}] state_horizon={sampled_state_horizon}, infer_ms={infer_ms:.3f}ms, rtt_ms={elapsed_ms:.3f}ms"
                )

            if args.sleep_ms > 0:
                await asyncio.sleep(args.sleep_ms / 1000.0)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WebSocket client that sends representative observations and prints inference time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, required=True, help="Server port.")
    parser.add_argument("--num-requests", type=int, default=5, help="Number of requests to send.")
    parser.add_argument("--sleep-ms", type=int, default=200, help="Sleep between requests in ms.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="One-shot defaults bundle. CLI flags below override preset values.",
    )
    parser.add_argument("--camera-setup", choices=["single", "both"], default="both",
                        help="single: one camera payload; both: head/chest dict payload.")
    parser.add_argument("--image-mode", choices=["rgb", "rgbd"], default="rgb",
                        help="rgb: no depth_image field; rgbd: include depth_image field.")
    parser.add_argument("--image-shape", default="6,480,640,3",
                        help="RGB image shape per camera, e.g. '6,480,640,3' (T=horizon, H, W, C).")
    parser.add_argument("--depth-shape", default="6,480,640,1",
                        help="Depth image shape per camera, e.g. '6,480,640,1'.")
    parser.add_argument("--intrinsic-shape", default="3,3",
                        help="Camera intrinsic shape; '3,3' uses the built-in default fx/fy/cx/cy.")
    parser.add_argument("--state-horizon", type=int, default=6,
                        help="State history length; sent as states.shape[0] when --fixed-state-horizon, "
                             "otherwise the upper bound for random sampling.")
    parser.add_argument("--state-dim", type=int, default=48, help="State vector dim.")
    parser.add_argument("--fixed-state-horizon", action=argparse.BooleanOptionalAction, default=True,
                        help="When set (default), every request uses the configured --state-horizon. "
                             "When --no-fixed-state-horizon, randomly sample 1/horizon//4/horizon//2/horizon "
                             "to mimic real client traffic with growing history.")
    parser.add_argument("--action-horizon", type=int, default=32,
                        help="Action chunk horizon, used only when --rtc-delay > 0.")
    parser.add_argument("--action-dim", type=int, default=48,
                        help="Action vector dim, used only when --rtc-delay > 0.")
    parser.add_argument("--rtc-delay", type=int, default=0,
                        help="If > 0, send action_rtc payload of length rtc_delay (zeros). "
                             "0 disables RTC (sends action_rtc=None).")
    parser.add_argument("--instruction", default="grasp the yellow toy", help="Instruction string.")
    parser.add_argument("--include-camera-extrinsics", action="store_true",
                        help="Include camera_extrinsics in payload for protocol parity tests.")

    args = parser.parse_args()

    # Apply preset by patching args. Only override values left at their argparse defaults
    # so explicit CLI flags continue to win.
    if args.preset:
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            if not hasattr(args, key):
                continue
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    return args



def main() -> None:
    args = _parse_args()
    asyncio.run(_run_client(args))


if __name__ == "__main__":
    main()
