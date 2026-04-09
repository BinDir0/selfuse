"""
WebSocket replay server for ground-truth action playback.

Loads one episode from a WebDataset shard, pre-computes action chunks
using the same pipeline as training (without normalization), and serves
them over the existing WebSocket + msgpack protocol. This allows
verifying that the inference client correctly receives and interprets
action chunks without needing a trained model.

Usage:
    python -m src.serving.replay_server \
        --shard-path /path/to/shard-00000.tar \
        --episode-index 0 \
        --port 8765 \
        --step-interval 12
"""

import argparse
import asyncio
import http
import logging
import time
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server as _server

from . import msgpack_numpy
from ..dataset.wds_dataset import (
    build_wds_pipeline,
    WindowConfig,
    LOWDIM_SLICES,
)
from ..dataset.data_transforms import process_state_action

logger = logging.getLogger(__name__)

# Match training config: unified_wds.yaml + inference.yaml
DEFAULT_WINDOW_CONFIG = WindowConfig(
    action_horizon=32,
    action_stride=1,
    state_horizon=6,
    state_stride=30,
    image_horizon=6,
    image_stride=30,
    history_pad_mode="repeat",
    future_pad_mode="truncate",
)


def load_episode_samples(
    shard_path: str,
    episode_nth: int = 0,
    window_config: WindowConfig | None = None,
) -> list[dict[str, Any]]:
    """Load all windowed samples for the N-th episode in a shard.

    Uses a simple counter to identify episodes by order of appearance,
    independent of the actual episode_index value stored in metadata.
    """
    if window_config is None:
        window_config = DEFAULT_WINDOW_CONFIG

    pipeline = build_wds_pipeline(
        shard_urls=[shard_path],
        config=window_config,
        lowdim_slices=LOWDIM_SLICES,
        mode="val",
        use_sliding_window=True,
        lowdim_only=True,
    )

    samples = []
    current_ep_key = None
    ep_counter = -1

    for sample in pipeline:
        ep_key = (sample.get("dataset_name", ""), sample["episode_index"])
        if ep_key != current_ep_key:
            current_ep_key = ep_key
            ep_counter += 1
            if ep_counter > episode_nth:
                break

        if ep_counter == episode_nth:
            samples.append(sample)

    if not samples:
        raise ValueError(
            f"Shard only contains {ep_counter + 1} episode(s), "
            f"requested the {episode_nth}-th (0-based)"
        )

    logger.info(
        "Loaded %d windowed samples for the %d-th episode from %s",
        len(samples), episode_nth, shard_path,
    )
    return samples


def precompute_action_chunks(
    samples: list[dict[str, Any]],
    step_interval: int = 12,
    hand_ndim: int = 15,
    motion_type: str = "fingertips",
    use_relative_action: bool = True,
    action_horizon: int = 32,
) -> list[np.ndarray]:
    """Process windowed samples into action chunks, sub-sampled at step_interval.

    Each returned array has shape [action_horizon, 48].
    Actions are in physical space (no normalization), matching the
    output format of RuntimeEngine.infer() after post_process().
    """
    selected = samples[::step_interval]
    chunks = []

    for sample in selected:
        _, action = process_state_action(
            wrist_state=sample["wrist_state"],
            hand_state=sample["hand_state"],
            wrist_action=sample["wrist_action"],
            hand_action=sample["hand_action"],
            extrinsic=sample["extrinsic"].reshape(4, 4),
            hand_ndim=hand_ndim,
            normalizer=None,
            motion_type=motion_type,
            use_relative_action=use_relative_action,
        )

        # Pad truncated actions at episode end to fixed horizon
        if action.shape[0] < action_horizon:
            pad = np.tile(action[-1:], (action_horizon - action.shape[0], 1))
            action = np.concatenate([action, pad], axis=0)

        chunks.append(action.astype(np.float32))

    logger.info(
        "Pre-computed %d action chunks (step_interval=%d, relative=%s)",
        len(chunks), step_interval, use_relative_action,
    )
    return chunks


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class ReplayServer:
    """Serve pre-computed action chunks over the WebSocket protocol."""

    def __init__(
        self,
        action_chunks: list[np.ndarray],
        host: str = "0.0.0.0",
        port: int = 8765,
        metadata: dict | None = None,
    ) -> None:
        self.action_chunks = action_chunks
        self.host = host
        self.port = port
        self.metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        step_index = 0
        num_chunks = len(self.action_chunks)

        # Protocol handshake: send metadata
        await websocket.send(packer.pack(self.metadata))

        while True:
            try:
                start_time = time.monotonic()

                # Receive observation (content ignored for replay)
                raw_obs = await websocket.recv()
                recv_time = time.monotonic() - start_time
                _ = msgpack_numpy.unpackb(raw_obs)

                # Look up pre-computed action chunk
                chunk_idx = step_index % num_chunks
                if step_index > 0 and chunk_idx == 0:
                    logger.info(
                        "Episode finished for %s, looping from start",
                        websocket.remote_address,
                    )

                action = self.action_chunks[chunk_idx]
                elapsed = time.monotonic() - start_time

                response = {
                    "pred_actions": action,
                    "server_timing": {
                        "recv_wait_ms": recv_time * 1000,
                        "infer_ms": 0.0,
                        "pack_ms": 0.0,
                        "pre_send_total_ms": elapsed * 1000,
                    },
                    "replay_step": step_index,
                    "is_replay": True,
                }

                packed = packer.pack(response)
                await websocket.send(packed)

                logger.info(
                    "[%s] step=%d chunk=%d/%d",
                    websocket.remote_address, step_index, chunk_idx, num_chunks,
                )
                step_index += 1

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WebSocket replay server for ground-truth action playback."
    )
    parser.add_argument(
        "--shard-path", required=True,
        help="Path to a WebDataset .tar shard",
    )
    parser.add_argument(
        "--episode-nth", type=int, default=0,
        help="Use the N-th episode in the shard, 0-based (default: 0)",
    )
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--step-interval", type=int, default=12,
        help="Frames between served action chunks (default: 12)",
    )
    parser.add_argument(
        "--no-relative-action", dest="use_relative_action",
        action="store_false", default=True,
        help="Disable relative action (default: enabled)",
    )
    parser.add_argument(
        "--hand-ndim", type=int, default=15,
        help="Per-hand fingertip dimensions (default: 15)",
    )
    parser.add_argument(
        "--motion-type", default="fingertips",
        help="Motion representation type (default: fingertips)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    logger.info("Loading the %d-th episode from %s", args.episode_nth, args.shard_path)
    samples = load_episode_samples(args.shard_path, args.episode_nth)

    action_chunks = precompute_action_chunks(
        samples,
        step_interval=args.step_interval,
        hand_ndim=args.hand_ndim,
        motion_type=args.motion_type,
        use_relative_action=args.use_relative_action,
    )

    metadata = {
        "mode": "replay",
        "action_horizon": DEFAULT_WINDOW_CONFIG.action_horizon,
        "action_dim": 48,
        "is_replay": True,
        "num_chunks": len(action_chunks),
        "step_interval": args.step_interval,
    }

    server = ReplayServer(
        action_chunks=action_chunks,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )

    logger.info(
        "Serving %d action chunks on %s:%d (step_interval=%d, relative=%s)",
        len(action_chunks), args.host, args.port,
        args.step_interval, args.use_relative_action,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
