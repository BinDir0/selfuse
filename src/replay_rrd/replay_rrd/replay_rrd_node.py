#!/usr/bin/env python3

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np
import rclpy
import rerun as rr
from ament_index_python.packages import get_package_prefix

try:
    import rerun_bindings as rrb
except Exception:
    rrb = None
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R


def _repo_root_from_this_file() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / 'assets' / 'PsiRobot_DC_02_OnlyArm' / 'meshes' / 'psi_robot_scene_transformed.xml'
        if candidate.exists():
            return parent
    return Path('/home/admin01/Documents/projects/LegendaryVLA/LegendVLA-Inference')


def _default_arm_xml_path() -> str:
    repo_root = _repo_root_from_this_file()
    return str(repo_root / 'assets' / 'PsiRobot_DC_02_OnlyArm' / 'meshes' / 'psi_robot_scene_transformed.xml')


def _load_rrd_recording(rrd_path: str):
    if hasattr(rr, 'recording') and hasattr(rr.recording, 'load_recording'):
        return rr.recording.load_recording(rrd_path)

    if hasattr(rr, 'dataframe') and hasattr(rr.dataframe, 'load_recording'):
        return rr.dataframe.load_recording(rrd_path)

    if hasattr(rr, 'load_recording'):
        return rr.load_recording(rrd_path)

    if rrb is not None and hasattr(rrb, 'load_recording'):
        return rrb.load_recording(rrd_path)

    raise RuntimeError('当前 rerun 版本不支持 load_recording，请安装包含 dataframe/recording API 的 rerun-sdk。')


def _resolve_hand_mjcf_file(hand_type: str) -> Path:
    hand_prefix = 'Left' if hand_type == 'left' else 'Right'
    relative_path = Path('RuiYan') / '0611_v1.4' / 'Version_3.0' / f'RuiYan_Hand_{hand_prefix}_Mimic' / 'meshes' / f'RuiYan_Hand_{hand_prefix}_Mimic_scene.xml'

    candidates = []

    repo_root = _repo_root_from_this_file()
    candidates.append(repo_root / 'src' / 'hand' / 'resource' / relative_path)

    try:
        hand_prefix_path = Path(get_package_prefix('hand'))
        candidates.append(hand_prefix_path / 'lib' / 'resource' / relative_path)
        candidates.append(hand_prefix_path / 'share' / 'hand' / 'resource' / relative_path)
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f'未找到 {hand_type} 手 MJCF 文件，已检查: ' + ', '.join(str(path) for path in candidates)
    )


@dataclass
class ReplayFrame:
    left_arm: np.ndarray
    right_arm: np.ndarray
    left_hand_raw: np.ndarray
    right_hand_raw: np.ndarray
    left_hand_cmd: np.ndarray
    right_hand_cmd: np.ndarray
    timestamp_sec: float


class ArmFKSolver:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.left_wrist_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'left_wrist')
        self.right_wrist_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'right_wrist')
        self.left_arm_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'arm1_link0')
        self.right_arm_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'arm2_link0')

    def compute_fk(self, left_arm_joints: np.ndarray, right_arm_joints: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        joint_angles = np.concatenate([left_arm_joints, right_arm_joints], axis=0).astype(np.float64)
        if joint_angles.shape[0] != self.model.nq:
            raise ValueError(f'Arm joint dimension mismatch: expected {self.model.nq}, got {joint_angles.shape[0]}')

        self.data.qpos[:] = joint_angles
        mujoco.mj_forward(self.model, self.data)

        left_base_pos = self.data.body(self.left_arm_base_id).xpos.copy()
        left_base_mat = self.data.body(self.left_arm_base_id).xmat.copy().reshape(3, 3)
        right_base_pos = self.data.body(self.right_arm_base_id).xpos.copy()
        right_base_mat = self.data.body(self.right_arm_base_id).xmat.copy().reshape(3, 3)

        left_tcp_pos = self.data.site(self.left_wrist_site_id).xpos.copy()
        left_tcp_mat = self.data.site(self.left_wrist_site_id).xmat.copy().reshape(3, 3)
        right_tcp_pos = self.data.site(self.right_wrist_site_id).xpos.copy()
        right_tcp_mat = self.data.site(self.right_wrist_site_id).xmat.copy().reshape(3, 3)

        left_tcp_pos_in_base = left_base_mat.T @ (left_tcp_pos - left_base_pos)
        left_tcp_mat_in_base = left_base_mat.T @ left_tcp_mat
        right_tcp_pos_in_base = right_base_mat.T @ (right_tcp_pos - right_base_pos)
        right_tcp_mat_in_base = right_base_mat.T @ right_tcp_mat

        return {
            'left': {
                'position': left_tcp_pos_in_base,
                'orientation': R.from_matrix(left_tcp_mat_in_base).as_quat(),
            },
            'right': {
                'position': right_tcp_pos_in_base,
                'orientation': R.from_matrix(right_tcp_mat_in_base).as_quat(),
            },
        }


class HandFKSolver:
    def __init__(self, hand_type='left'):
        self.hand_type = hand_type
        self.model = mujoco.MjModel.from_xml_path(str(_resolve_hand_mjcf_file(hand_type)))
        self.data = mujoco.MjData(self.model)
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        self._setup_joint_info()
        self._record_joint_metadata()
        self._setup_site_info()
        mujoco.mj_forward(self.model, self.data)

    def _setup_joint_info(self):
        all_joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        active_candidates = [
            f'{prefix}_joint_link_1_1', f'{prefix}_joint_link_1_2',
            f'{prefix}_joint_link_2_1', f'{prefix}_joint_link_3_1',
            f'{prefix}_joint_link_4_1', f'{prefix}_joint_link_5_1',
        ]
        self.joint_names = [joint_name for joint_name in active_candidates if joint_name in all_joints]

    def _record_joint_metadata(self):
        self.joint_metadata = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            lower, upper = self.model.jnt_range[joint_id]
            self.joint_metadata.append({
                'name': name,
                'adr': qpos_adr,
                'low': lower,
                'range': (upper - lower) if (upper - lower) != 0 else 1.0,
            })

    def _setup_site_info(self):
        side_prefix = self.hand_type
        self.site_ids_list = []
        for finger_name in self.finger_names:
            site_name = f'{side_prefix}_{finger_name}_tip'
            try:
                self.site_ids_list.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name))
            except Exception:
                pass

    def _get_mimic_relations(self):
        prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        return [
            (f'{prefix}_joint_link_1_2', f'{prefix}_joint_link_1_3', 1.675, 0.0),
            (f'{prefix}_joint_link_2_1', f'{prefix}_joint_link_2_2', 1.0, 0.0),
            (f'{prefix}_joint_link_3_1', f'{prefix}_joint_link_3_2', 1.0, 0.0),
            (f'{prefix}_joint_link_4_1', f'{prefix}_joint_link_4_2', 1.0, 0.0),
            (f'{prefix}_joint_link_5_1', f'{prefix}_joint_link_5_2', 1.0, 0.0),
        ]

    def _apply_mimic_joints(self):
        for leader_name, follower_name, multiplier, offset in self._get_mimic_relations():
            try:
                leader_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name)
                follower_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                leader_adr = self.model.jnt_qposadr[leader_id]
                follower_adr = self.model.jnt_qposadr[follower_id]
                self.data.qpos[follower_adr] = offset + multiplier * self.data.qpos[leader_adr]
            except Exception:
                pass

    def compute_fk(self, normalized_positions: np.ndarray):
        if len(normalized_positions) != len(self.joint_metadata):
            return None

        for index, metadata in enumerate(self.joint_metadata):
            value_rad = metadata['low'] + normalized_positions[index] * metadata['range']
            self.data.qpos[metadata['adr']] = value_rad

        self._apply_mimic_joints()
        mujoco.mj_forward(self.model, self.data)

        results = []
        for site_id in self.site_ids_list:
            pos = self.data.site(site_id).xpos.copy()
            quat = R.from_matrix(self.data.site(site_id).xmat.copy().reshape(3, 3)).as_quat()
            results.append((pos, quat))
        return results


class ReplayRRDNode(Node):
    def __init__(self):
        super().__init__('replay_rrd_node')

        self.declare_parameter('rrd_file_path', '')
        self.declare_parameter('target_hz', 30.0)
        self.declare_parameter('loop_playback', False)
        self.declare_parameter('arm_xml_path', _default_arm_xml_path())

        self.rrd_file_path = self.get_parameter('rrd_file_path').value
        self.target_hz = float(self.get_parameter('target_hz').value)
        self.loop_playback = bool(self.get_parameter('loop_playback').value)
        self.arm_xml_path = self.get_parameter('arm_xml_path').value

        if not self.rrd_file_path:
            raise RuntimeError('参数 rrd_file_path 不能为空，请在配置文件中指定待回放的 .rrd 文件。')
        if not os.path.exists(self.rrd_file_path):
            raise RuntimeError(f'RRD 文件不存在: {self.rrd_file_path}')
        if not os.path.exists(self.arm_xml_path):
            raise RuntimeError(f'机械臂 FK XML 不存在: {self.arm_xml_path}')

        self.arm_action_pub = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 10)
        self.left_hand_action_pub = self.create_publisher(PoseArray, '/action/left_hand/keypoints', 10)
        self.right_hand_action_pub = self.create_publisher(PoseArray, '/action/right_hand/keypoints', 10)

        self.arm_fk_solver = ArmFKSolver(self.arm_xml_path)
        self.left_hand_fk_solver = HandFKSolver(hand_type='left')
        self.right_hand_fk_solver = HandFKSolver(hand_type='right')

        self.frames = self._load_replay_frames(self.rrd_file_path, self.target_hz)
        self.frame_index = 0

        self.timer = self.create_timer(1.0 / self.target_hz, self._timer_callback)

        duration_sec = self.frames[-1].timestamp_sec - self.frames[0].timestamp_sec if len(self.frames) > 1 else 0.0
        self.get_logger().info(
            f'ReplayRRDNode ready: {len(self.frames)} frames, {self.target_hz:.1f} Hz, duration {duration_sec:.2f}s, file={self.rrd_file_path}'
        )

    def _load_replay_frames(self, rrd_path: str, target_hz: float) -> List[ReplayFrame]:
        recording = _load_rrd_recording(rrd_path)
        table = recording.view(index='timestamp', contents='/**').select().read_all()
        timestamps_sec = table['timestamp'].to_numpy().astype(np.float64) / 1e9

        left_arm = self._extract_arm_stream(table, timestamps_sec, 'left')
        right_arm = self._extract_arm_stream(table, timestamps_sec, 'right')
        left_hand = self._extract_hand_stream(table, timestamps_sec, 'left')
        right_hand = self._extract_hand_stream(table, timestamps_sec, 'right')

        common_start = max(left_arm['timestamp'][0], right_arm['timestamp'][0], left_hand['timestamp'][0], right_hand['timestamp'][0])
        common_end = min(left_arm['timestamp'][-1], right_arm['timestamp'][-1], left_hand['timestamp'][-1], right_hand['timestamp'][-1])
        if common_end <= common_start:
            raise RuntimeError('无法构造公共回放时间轴：四路数据没有有效重叠区间。')

        dt = 1.0 / target_hz
        target_timestamps = np.arange(common_start, common_end + 0.5 * dt, dt, dtype=np.float64)
        if target_timestamps.size == 0:
            raise RuntimeError('降采样后的时间轴为空，请检查 RRD 数据和 target_hz 参数。')

        left_arm_samples = self._resample_stream(left_arm, target_timestamps)
        right_arm_samples = self._resample_stream(right_arm, target_timestamps)
        left_hand_raw_samples = self._resample_stream(left_hand, target_timestamps)
        right_hand_raw_samples = self._resample_stream(right_hand, target_timestamps)

        frames: List[ReplayFrame] = []
        for ts, l_arm, r_arm, l_hand_raw, r_hand_raw in zip(
            target_timestamps,
            left_arm_samples,
            right_arm_samples,
            left_hand_raw_samples,
            right_hand_raw_samples,
        ):
            frames.append(
                ReplayFrame(
                    left_arm=np.asarray(l_arm, dtype=np.float64),
                    right_arm=np.asarray(r_arm, dtype=np.float64),
                    left_hand_raw=np.asarray(l_hand_raw, dtype=np.float64),
                    right_hand_raw=np.asarray(r_hand_raw, dtype=np.float64),
                    left_hand_cmd=self._reorder_hand_action_for_control(l_hand_raw),
                    right_hand_cmd=self._reorder_hand_action_for_control(r_hand_raw),
                    timestamp_sec=float(ts),
                )
            )

        return frames

    def _extract_scalar_value(self, value):
        while isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return np.nan
            value = value[0]

        if value is None:
            return np.nan

        try:
            return float(value)
        except Exception:
            return np.nan

    def _find_columns_by_tokens(self, column_names, joint_tokens, prefix_candidates):
        matched_columns = []
        for token in joint_tokens:
            token_matches = []
            for column_name in column_names:
                if not any(column_name.startswith(prefix) for prefix in prefix_candidates):
                    continue
                if token not in column_name:
                    continue
                if ':Scalars:' not in column_name and ':Scalar' not in column_name:
                    continue
                token_matches.append(column_name)

            token_matches = sorted(token_matches, key=len)
            if not token_matches:
                return None
            matched_columns.append(token_matches[0])

        return matched_columns

    def _extract_joint_stream(self, table, timestamps_sec: np.ndarray, candidate_specs, stream_name: str) -> Dict[str, np.ndarray]:
        column_names = list(table.column_names)
        chosen_columns = None

        for spec in candidate_specs:
            exact_columns = spec.get('exact_columns')
            if exact_columns is not None and all(column in column_names for column in exact_columns):
                chosen_columns = exact_columns
                break

            token_columns = self._find_columns_by_tokens(
                column_names=column_names,
                joint_tokens=spec.get('joint_tokens', []),
                prefix_candidates=spec.get('prefixes', []),
            )
            if token_columns is not None:
                chosen_columns = token_columns
                break

        if chosen_columns is None:
            related_columns = [
                name for name in column_names
                if any(prefix in name for spec in candidate_specs for prefix in spec.get('prefixes', []))
            ]
            raise RuntimeError(
                f'RRD 缺少 {stream_name} 列。候选组均未匹配成功: {candidate_specs}\n'
                f'{stream_name} 相关可用列: {related_columns[:80]}'
            )

        data_list = []
        for column in chosen_columns:
            processed = [self._extract_scalar_value(value) for value in table[column].to_pylist()]
            data_list.append(np.asarray(processed, dtype=np.float64))

        joint_data = np.column_stack(data_list)
        valid_mask = ~np.isnan(joint_data).any(axis=1)

        if valid_mask.sum() == 0:
            raise RuntimeError(f'{stream_name} 数据为空。')

        self.get_logger().info(f'{stream_name} 使用列: {chosen_columns}')
        return {
            'timestamp': timestamps_sec[valid_mask],
            'data': joint_data[valid_mask],
        }

    def _extract_arm_stream(self, table, timestamps_sec: np.ndarray, side: str) -> Dict[str, np.ndarray]:
        if side == 'left':
            action_joint_names = [f'arm1_joint_link{i+1}' for i in range(7)]
            state_joint_names = [f'left_joint_{i+1}' for i in range(7)]
            state_prefix = '/left_arm/joint_states/position'
        else:
            action_joint_names = [f'arm2_joint_link{i+1}' for i in range(7)]
            state_joint_names = [f'right_joint_{i+1}' for i in range(7)]
            state_prefix = '/right_arm/joint_states/position'

        candidate_specs = [
            {
                'exact_columns': [f'/ik_output/position/{name}:Scalars:scalars' for name in action_joint_names],
                'prefixes': ['/ik_output'],
                'joint_tokens': action_joint_names,
            },
            {
                'exact_columns': [f'{state_prefix}/{name}:Scalars:scalars' for name in state_joint_names],
                'prefixes': [f'/{side}_arm/joint_states', state_prefix],
                'joint_tokens': state_joint_names,
            },
        ]
        return self._extract_joint_stream(table, timestamps_sec, candidate_specs, f'{side} arm')

    def _extract_hand_stream(self, table, timestamps_sec: np.ndarray, side: str) -> Dict[str, np.ndarray]:
        action_joint_names = ['thumb_bend', 'thumb_rotate', 'index_bend', 'middle_bend', 'ring_bend', 'pinky_bend']
        state_joint_names = ['thumb_bend', 'thumb_rotation', 'index', 'middle', 'ring', 'pinky']

        candidate_specs = [
            {
                'exact_columns': [f'/ry_hand/{side}/set_angles/position/{name}:Scalars:scalars' for name in action_joint_names],
                'prefixes': [f'/ry_hand/{side}/set_angles'],
                'joint_tokens': action_joint_names,
            },
            {
                'exact_columns': [f'/ry_hand/{side}/joint_states/position/{name}:Scalars:scalars' for name in state_joint_names],
                'prefixes': [f'/ry_hand/{side}/joint_states'],
                'joint_tokens': state_joint_names,
            },
        ]
        return self._extract_joint_stream(table, timestamps_sec, candidate_specs, f'{side} hand')

    def _resample_stream(self, stream: Dict[str, np.ndarray], target_timestamps: np.ndarray) -> List[np.ndarray]:
        source_timestamps = stream['timestamp']
        source_data = stream['data']
        samples = []
        for target_ts in target_timestamps:
            idx = np.searchsorted(source_timestamps, target_ts, side='right') - 1
            idx = max(0, min(idx, len(source_timestamps) - 1))
            samples.append(source_data[idx])
        return samples

    def _reorder_hand_action_for_control(self, raw_hand_action: np.ndarray) -> np.ndarray:
        reordered = np.asarray(raw_hand_action, dtype=np.float64).copy()
        reordered[0] = raw_hand_action[1]
        reordered[1] = raw_hand_action[0]
        return reordered

    def _timer_callback(self):
        if self.frame_index >= len(self.frames):
            if self.loop_playback:
                self.frame_index = 0
            else:
                self.get_logger().info('Replay completed.')
                self.timer.cancel()
                return

        frame = self.frames[self.frame_index]
        now = self.get_clock().now().to_msg()

        arm_fk = self.arm_fk_solver.compute_fk(frame.left_arm, frame.right_arm)
        left_hand_fk = self.left_hand_fk_solver.compute_fk(frame.left_hand_cmd)
        right_hand_fk = self.right_hand_fk_solver.compute_fk(frame.right_hand_cmd)

        self.arm_action_pub.publish(self._build_arm_pose_array(now, arm_fk))
        self.left_hand_action_pub.publish(self._build_hand_pose_array(now, 'left', left_hand_fk))
        self.right_hand_action_pub.publish(self._build_hand_pose_array(now, 'right', right_hand_fk))

        self.frame_index += 1

    def _build_arm_pose_array(self, stamp, arm_fk: Dict[str, Dict[str, np.ndarray]]) -> PoseArray:
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'

        for side in ('left', 'right'):
            pose = Pose()
            pose.position.x = float(arm_fk[side]['position'][0])
            pose.position.y = float(arm_fk[side]['position'][1])
            pose.position.z = float(arm_fk[side]['position'][2])
            pose.orientation.x = float(arm_fk[side]['orientation'][0])
            pose.orientation.y = float(arm_fk[side]['orientation'][1])
            pose.orientation.z = float(arm_fk[side]['orientation'][2])
            pose.orientation.w = float(arm_fk[side]['orientation'][3])
            msg.poses.append(pose)

        return msg

    def _build_hand_pose_array(self, stamp, side: str, pose_list) -> PoseArray:
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = f'{side}_wrist_link'

        if pose_list is None:
            return msg

        for pos, quat in pose_list:
            pose = Pose()
            pose.position.x = float(pos[0])
            pose.position.y = float(pos[1])
            pose.position.z = float(pos[2])
            msg.poses.append(pose)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ReplayRRDNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
