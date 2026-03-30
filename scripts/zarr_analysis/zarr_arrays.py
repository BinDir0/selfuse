"""从 teleop 风格 Zarr（meta + data/*）读取分析所需的 numpy 字典。"""
from __future__ import annotations

from typing import Dict

import numpy as np
import zarr


def load_episode_data(zarr_path: str) -> Dict[str, np.ndarray]:
    print(f"Loading data from: {zarr_path}")
    root = zarr.open(zarr_path, mode="r")

    episode_names = root["meta"]["episode_names"][:]
    episode_ends = root["meta"]["episode_ends"][:]

    print(f"  - Number of episodes: {len(episode_names)}")
    print(f"  - Total frames: {episode_ends[-1] if len(episode_ends) > 0 else 0}")

    print("  - Loading instruction data...")
    instruction = root["data"]["instruction"][:]
    instruction_num = root["data"]["instruction_num"][:]

    print("  - Loading camera parameters...")
    intrinsic_h = root["data"]["intrinsic"]["head"][:]
    intrinsic_b = root["data"]["intrinsic"]["breast"][:]
    extrinsic_bl = root["data"]["extrinsic-calib"]["breast-left"][:]
    extrinsic_br = root["data"]["extrinsic-calib"]["breast-right"][:]
    extrinsic_hl = root["data"]["extrinsic-calib"]["head-left"][:]
    extrinsic_hr = root["data"]["extrinsic-calib"]["head-right"][:]

    print("  - Loading joint data...")
    joint_state = root["data"]["state"]["joint"][:]
    joint_action = root["data"]["action"]["joint"][:]

    print("  - Loading wrist and fingertips data...")
    wrist_state_h = root["data"]["state"]["wrist-head"][:]
    wrist_action_h = root["data"]["action"]["wrist-head"][:]
    wrist_state_b = root["data"]["state"]["wrist-breast"][:]
    wrist_action_b = root["data"]["action"]["wrist-breast"][:]
    fingertips_state_h = root["data"]["state"]["fingertips-head"][:]
    fingertips_action_h = root["data"]["action"]["fingertips-head"][:]
    fingertips_state_b = root["data"]["state"]["fingertips-breast"][:]
    fingertips_action_b = root["data"]["action"]["fingertips-breast"][:]

    return {
        "episode_names": episode_names,
        "episode_ends": episode_ends,
        "instruction": instruction,
        "instruction_num": instruction_num,
        "intrinsic_h": intrinsic_h,
        "intrinsic_b": intrinsic_b,
        "extrinsic_hl": extrinsic_hl,
        "extrinsic_hr": extrinsic_hr,
        "extrinsic_bl": extrinsic_bl,
        "extrinsic_br": extrinsic_br,
        "joint_state": joint_state,
        "joint_action": joint_action,
        "wrist_state_h": wrist_state_h,
        "wrist_action_h": wrist_action_h,
        "wrist_state_b": wrist_state_b,
        "wrist_action_b": wrist_action_b,
        "fingertips_state_h": fingertips_state_h,
        "fingertips_action_h": fingertips_action_h,
        "fingertips_state_b": fingertips_state_b,
        "fingertips_action_b": fingertips_action_b,
    }
