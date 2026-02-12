"""
Hand IK Solver - 灵巧手逆运动学求解器

基于 MuJoCo + Mink 实现的纯 IK 计算模块，无 ROS2 依赖。
为 HandIKNode 提供核心逆运动学求解功能。

输入: 5 个指尖的 3D 目标位置 (wrist frame, 单位: m)
输出: 6 个关节的归一化角度 (0-1)

关节顺序 (与遥操作代码 ry_hand_node 保持一致):
    [thumb_rotation, thumb_bend, index, middle, ring, pinky]

参考:
    - psirobot_visualizer/ruiyan_hand/hand_ik.py (原始 IK 实现)
    - mj-controller/src/haptic_hand_control (遥操作关节命名规范)
"""

import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import mink
import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 自由度数量（与遥操作代码 ry_hand_node.py 的 self.dof = 6 保持一致）
DOF = 6

# 手指名称（顺序固定，与 site 一一对应）
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# 输出关节名称（与遥操作代码 ry_hand_node.py 保持一致）
# ry_hand_node.py line 152: ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
# ry_hand_485_node.py line 190-196: same mapping
JOINT_NAMES = (
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
)

# MuJoCo 模型中的关节前缀映射
_MUJOCO_JOINT_PREFIX = {"left": "hand1", "right": "hand2"}


# ---------------------------------------------------------------------------
# HandIKSolver
# ---------------------------------------------------------------------------


class HandIKSolver:
    """
    灵巧手逆运动学求解器

    功能:
        - 接收 5 个指尖 3D 目标位置 (wrist frame)
        - 使用 MuJoCo + Mink 求解 IK
        - 返回 6 个归一化关节角度 (0-1)

    特点:
        - 线程安全（内部使用 threading.Lock）
        - 无 ROS2 / GUI 依赖
        - 支持增量求解（保持上一次构型作为初始值，运动更平滑）

    使用方式:
        solver = HandIKSolver(hand_type='left', mjcf_path='/path/to/scene.xml')
        joints, info = solver.compute_ik({
            'thumb': [0.01, 0.02, 0.03],
            'index': [0.04, 0.05, 0.06],
        })
    """

    # 6 个主动关节在 MuJoCo 模型中的后缀（顺序与 JOINT_NAMES 一一对应）
    _ACTIVE_JOINT_SUFFIXES = (
        "_joint_link_1_1",  # thumb_rotation
        "_joint_link_1_2",  # thumb_bend
        "_joint_link_2_1",  # index
        "_joint_link_3_1",  # middle
        "_joint_link_4_1",  # ring
        "_joint_link_5_1",  # pinky
    )

    # Mimic 关系: (leader_suffix, follower_suffix, multiplier, offset)
    _MIMIC_RELATIONS = (
        ("_joint_link_1_2", "_joint_link_1_3", 1.675, 0.0),  # thumb
        ("_joint_link_2_1", "_joint_link_2_2", 1.0, 0.0),    # index
        ("_joint_link_3_1", "_joint_link_3_2", 1.0, 0.0),    # middle
        ("_joint_link_4_1", "_joint_link_4_2", 1.0, 0.0),    # ring
        ("_joint_link_5_1", "_joint_link_5_2", 1.0, 0.0),    # pinky
    )

    def __init__(
        self,
        hand_type: str = "left",
        mjcf_path: Optional[str] = None,
        solver: str = "daqp",
        frequency: float = 80.0,
    ):
        """
        初始化 IK 求解器

        Args:
            hand_type: 'left' 或 'right'
            mjcf_path: MJCF 场景文件的绝对路径。
                       为 None 时使用默认路径 (hand/models/ 下)。
            solver: QP 求解器类型 ('daqp', 'quadprog', 'proxqp')
            frequency: IK 求解频率 (Hz)，影响积分步长 dt
        """
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got '{hand_type}'")

        self.hand_type = hand_type
        self.solver = solver
        self.dt = 1.0 / frequency
        self._data_lock = threading.Lock()

        # 加载 MuJoCo 模型
        resolved_path = self._resolve_mjcf_path(mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(str(resolved_path))
        self.configuration = mink.Configuration(self.model)

        # 初始化各组件
        self._setup_active_joints()
        self._setup_fingertip_sites()
        self._setup_ik_tasks()
        self._setup_joint_limits()
        self._setup_mimic_joints()

        # 初始化构型并预热
        self._init_configuration()
        self._warm_up()

    # ------------------------------------------------------------------
    # 初始化方法
    # ------------------------------------------------------------------

    def _resolve_mjcf_path(self, mjcf_path: Optional[str]) -> Path:
        """
        解析 MJCF 文件路径

        查找顺序:
            1. 用户显式指定的 mjcf_path
            2. ROS2 包 share 目录下的 models/ (colcon build 后)
            3. 源码相对路径 (开发模式 / Docker 直接运行)
            4. Repository assets 目录 (新增: /path/to/LegendVLA-Inference/assets)
        """
        if mjcf_path is not None:
            path = Path(mjcf_path)
            if not path.exists():
                raise FileNotFoundError(f"MJCF file not found: {path}")
            return path

        hand_prefix = "Left" if self.hand_type == "left" else "Right"
        scene_filename = f"RuiYan_Hand_{hand_prefix}_Mimic_scene.xml"
        relative_subpath = (
            Path("models")
            / f"RuiYan_Hand_{hand_prefix}_Mimic"
            / "meshes"
            / scene_filename
        )

        # 尝试 1: ROS2 包 share 目录 (colcon build 后的安装路径)
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = Path(get_package_share_directory("hand"))
            share_path = share_dir / relative_subpath
            if share_path.exists():
                return share_path
        except Exception:
            pass

        # 尝试 2: 源码相对路径
        #   hand_ik_solver.py 位于 src/hand/hand/
        #   models 目录位于        src/hand/models/
        source_path = Path(__file__).parent.parent / relative_subpath
        if source_path.exists():
            return source_path

        # 尝试 3: Repository assets 目录
        #   hand_ik_solver.py 位于 src/hand/hand/
        #   assets 目录位于        assets/
        #   路径: assets/ruiyan_hand/InspiredHand_RuiYan/0611_v1.4/Version_3.0/RuiYan_Hand_{Left|Right}_Mimic/meshes/
        repo_root = Path(__file__).parent.parent.parent.parent  # Go up to repo root
        assets_path = (
            repo_root
            / "assets"
            / "ruiyan_hand"
            / "InspiredHand_RuiYan"
            / "0611_v1.4"
            / "Version_3.0"
            / f"RuiYan_Hand_{hand_prefix}_Mimic"
            / "meshes"
            / scene_filename
        )
        if assets_path.exists():
            return assets_path

        raise FileNotFoundError(
            f"MJCF file not found in any search path:\n"
            f"  - Share dir: (ament_index lookup failed or file missing)\n"
            f"  - Source dir: {source_path}\n"
            f"  - Assets dir: {assets_path}\n"
            "Please set the 'mjcf_path' parameter to the correct absolute path."
        )

    def _setup_active_joints(self):
        """设置 6 个主动关节的名称和 ID"""
        prefix = _MUJOCO_JOINT_PREFIX[self.hand_type]

        # 获取模型中所有关节名称
        all_joint_names = set(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        )

        # 构建主动关节名称列表
        self._mujoco_joint_names = [
            f"{prefix}{suffix}" for suffix in self._ACTIVE_JOINT_SUFFIXES
        ]

        # 验证所有主动关节都存在
        missing = [j for j in self._mujoco_joint_names if j not in all_joint_names]
        if missing:
            raise RuntimeError(
                f"Active joints not found in model: {missing}\n"
                f"Available joints: {sorted(all_joint_names)}"
            )

        # 缓存关节 ID 和 qpos 地址
        self._joint_ids = []
        self._joint_qpos_addrs = []
        for name in self._mujoco_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_ids.append(jid)
            self._joint_qpos_addrs.append(self.model.jnt_qposadr[jid])

    def _setup_fingertip_sites(self):
        """设置 5 个指尖 site 的名称和 ID"""
        self._site_names = [
            f"{self.hand_type}_thumb_tip",
            f"{self.hand_type}_index_tip",
            f"{self.hand_type}_middle_tip",
            f"{self.hand_type}_ring_tip",
            f"{self.hand_type}_pinky_tip",
        ]

        self._site_ids = []
        for name in self._site_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise RuntimeError(f"Fingertip site not found in model: {name}")
            self._site_ids.append(sid)

    def _setup_ik_tasks(self):
        """设置 mink IK 任务（纯位置约束，极低姿态权重）"""
        self._fingertip_tasks = []
        for site_name in self._site_names:
            task = mink.FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=1000.0,       # 位置权重（高）
                orientation_cost=0.001,      # 姿态权重（极低，几乎不约束）
                lm_damping=0.001,            # LM 阻尼
            )
            self._fingertip_tasks.append(task)

        self._all_tasks = list(self._fingertip_tasks)

    def _setup_joint_limits(self):
        """缓存关节限位信息（用于弧度 <-> 归一化转换）"""
        n = len(self._joint_ids)
        self._limits_lower = np.zeros(n)
        self._limits_upper = np.zeros(n)
        for i, jid in enumerate(self._joint_ids):
            self._limits_lower[i] = self.model.jnt_range[jid][0]
            self._limits_upper[i] = self.model.jnt_range[jid][1]
        self._limits_range = self._limits_upper - self._limits_lower

        # mink 关节限制约束
        self._mink_limits = [mink.ConfigurationLimit(self.model)]

    def _init_configuration(self):
        """初始化 mink 构型（优先使用 home keyframe）"""
        try:
            self.configuration.update_from_keyframe("home")
        except Exception:
            self.configuration.q = np.zeros(self.model.nq)

        with self._data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)

    def _warm_up(self):
        """预热仿真，确保内部数据一致性"""
        for _ in range(10):
            with self._data_lock:
                mujoco.mj_forward(self.model, self.configuration.data)

    # ------------------------------------------------------------------
    # Mimic Joint 处理
    # ------------------------------------------------------------------

    def _setup_mimic_joints(self):
        """
        预缓存 mimic joint 的 qpos 地址和参数

        在初始化阶段完成所有 mj_name2id 查找，
        运行时 _apply_mimic_joints 直接使用缓存地址，零开销。
        """
        prefix = _MUJOCO_JOINT_PREFIX[self.hand_type]

        # 缓存格式: [(leader_qpos_addr, follower_qpos_addr, multiplier, offset), ...]
        self._mimic_cache = []
        for leader_suffix, follower_suffix, multiplier, offset in self._MIMIC_RELATIONS:
            leader_name = f"{prefix}{leader_suffix}"
            follower_name = f"{prefix}{follower_suffix}"
            try:
                leader_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name
                )
                follower_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name
                )
                self._mimic_cache.append((
                    self.model.jnt_qposadr[leader_id],
                    self.model.jnt_qposadr[follower_id],
                    multiplier,
                    offset,
                ))
            except Exception:
                pass  # 某些模型版本可能缺少部分 mimic 关节

    def _apply_mimic_joints(self):
        """
        手动应用 mimic joint 约束

        MuJoCo 的 equality 约束只在动力学仿真中生效，
        纯运动学 IK 计算中需要手动同步从动关节。
        使用初始化时预缓存的地址，运行时无额外开销。
        """
        qpos = self.configuration.data.qpos
        for leader_addr, follower_addr, multiplier, offset in self._mimic_cache:
            qpos[follower_addr] = offset + multiplier * qpos[leader_addr]

    # ------------------------------------------------------------------
    # 单位转换
    # ------------------------------------------------------------------

    def _radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        """弧度 -> 归一化值 (0-1)"""
        return (radians - self._limits_lower) / self._limits_range

    def _get_active_joint_radians(self) -> np.ndarray:
        """获取 6 个主动关节的当前弧度值（调用方需持有 _data_lock）"""
        return np.array([
            self.configuration.data.qpos[addr]
            for addr in self._joint_qpos_addrs
        ])

    # ------------------------------------------------------------------
    # 核心求解接口
    # ------------------------------------------------------------------

    def compute_ik(
        self,
        target_positions: Dict[str, np.ndarray],
        max_iterations: int = 10,
    ) -> Tuple[np.ndarray, dict]:
        """
        逆运动学求解：指尖位置 -> 归一化关节角度

        Args:
            target_positions: 目标指尖位置 (wrist frame, 单位: m)
                {
                    'thumb': np.array([x, y, z]),
                    'index': np.array([x, y, z]),
                    ...
                }
                只需包含需要控制的手指。未包含的手指保持当前位置不动。
            max_iterations: 最大迭代次数（默认 10，经测试为最优值）

        Returns:
            normalized_joints: np.ndarray, shape (6,)
                归一化关节角度 (0-1)，顺序为 JOINT_NAMES
            info: dict, 包含收敛信息
                {
                    'converged': bool,       是否收敛
                    'iterations': int,       实际迭代次数
                    'velocity_error': float, 速度范数
                    'position_error': float, 最大位置误差 (m)
                }
        """
        with self._data_lock:
            # Step 1: 将未指定手指的当前位置设为默认目标（保持不动）
            mujoco.mj_forward(self.model, self.configuration.data)
            for i, sid in enumerate(self._site_ids):
                current_pos = self.configuration.data.site(sid).xpos.copy()
                current_rot = self.configuration.data.site(sid).xmat.reshape(3, 3).copy()
                transform = np.eye(4)
                transform[:3, :3] = current_rot
                transform[:3, 3] = current_pos
                self._fingertip_tasks[i].set_target(mink.SE3.from_matrix(transform))

            # Step 2: IK 迭代求解
            vel_error = 0.0
            actual_iterations = 0
            for iteration in range(max_iterations):
                actual_iterations = iteration + 1

                mujoco.mj_forward(self.model, self.configuration.data)

                # 更新目标（只覆盖 target_positions 中指定的手指）
                for i, finger_name in enumerate(FINGER_NAMES):
                    if finger_name in target_positions:
                        pos = np.asarray(target_positions[finger_name], dtype=np.float64)
                        # 位置来自目标，姿态保持当前值（只约束位置）
                        rot_mat = self.configuration.data.site(
                            self._site_ids[i]
                        ).xmat.reshape(3, 3).copy()
                        transform = np.eye(4)
                        transform[:3, :3] = rot_mat
                        transform[:3, 3] = pos
                        self._fingertip_tasks[i].set_target(
                            mink.SE3.from_matrix(transform)
                        )

                # QP 求解
                vel = mink.solve_ik(
                    configuration=self.configuration,
                    tasks=self._all_tasks,
                    dt=self.dt,
                    solver=self.solver,
                    damping=1e-5,
                    safety_break=False,
                    limits=self._mink_limits,
                )

                # 积分更新关节位置
                self.configuration.integrate_inplace(vel, self.dt)

                # 同步 mimic joints
                self._apply_mimic_joints()

                # 收敛判断
                vel_error = float(np.linalg.norm(vel))
                if vel_error < 1e-8:
                    break

            # Step 3: 计算最终位置误差
            max_position_error = 0.0
            mujoco.mj_forward(self.model, self.configuration.data)
            for i, finger_name in enumerate(FINGER_NAMES):
                if finger_name in target_positions:
                    current_pos = self.configuration.data.site(
                        self._site_ids[i]
                    ).xpos.copy()
                    target_pos = np.asarray(target_positions[finger_name])
                    error = float(np.linalg.norm(current_pos - target_pos))
                    max_position_error = max(max_position_error, error)

            # Step 4: 提取主动关节角度 -> 归一化
            active_radians = np.array([
                self.configuration.data.qpos[addr]
                for addr in self._joint_qpos_addrs
            ])
            normalized_joints = np.clip(
                self._radians_to_normalized(active_radians), 0.0, 1.0
            )

        info = {
            "converged": vel_error < 1e-8,
            "iterations": actual_iterations,
            "velocity_error": vel_error,
            "position_error": max_position_error,
        }

        return normalized_joints, info

    def get_fingertip_positions(self) -> Dict[str, np.ndarray]:
        """
        获取当前所有指尖的 3D 位置 (wrist frame)

        Returns:
            {finger_name: np.array([x, y, z])} for each finger
        """
        positions = {}
        with self._data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)
            for finger_name, sid in zip(FINGER_NAMES, self._site_ids):
                positions[finger_name] = self.configuration.data.site(sid).xpos.copy()
        return positions
