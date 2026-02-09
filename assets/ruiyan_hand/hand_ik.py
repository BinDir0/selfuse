"""
RuiYan Hand IK Solver

基于 MuJoCo + Mink 实现的灵巧手逆运动学求解器
支持左手和右手，提供交互式命令行界面

使用方法：
    python hand_ik.py --hand left --viewer
    
功能：
    - 输入目标位姿 -> 输出关节角度（IK）
    - 支持控制单个或多个指尖
    - MuJoCo 可视化（可选）
    - 无 ROS2 依赖
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import mink
import os
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import threading


class HandIK:
    """
    灵巧手逆运动学求解器
    
    功能：
        - 根据目标位姿计算关节角度（IK）
        - 支持左手/右手
        - 支持多指尖同时控制
        - 可视化支持
    """
    
    def __init__(self, hand_type='left', enable_viewer=True, solver='daqp', frequency=100.0):
        """
        初始化IK求解器
        
        Args:
            hand_type: 'left' 或 'right'
            enable_viewer: 是否启用MuJoCo可视化
            solver: IK求解器类型 ('daqp', 'quadprog', 'proxqp')
            frequency: 求解频率(Hz)，默认100Hz（推荐）
        """
        self.hand_type = hand_type
        self.enable_viewer = enable_viewer
        self.solver = solver
        self.frequency = frequency
        self.dt = 1.0 / frequency
        
        # 获取MJCF场景文件路径 (在meshes目录，和OBJ文件同级)
        script_dir = Path(__file__).parent
        hand_prefix = 'Left' if hand_type == 'left' else 'Right'
        mjcf_dir = script_dir / 'InspiredHand_RuiYan' / '0611_v1.4' / 'Version_3.0' / f'RuiYan_Hand_{hand_prefix}_Mimic' / 'meshes'
        mjcf_file = mjcf_dir / f'RuiYan_Hand_{hand_prefix}_Mimic_scene.xml'
        
        if not mjcf_file.exists():
            raise FileNotFoundError(f"MJCF场景文件不存在: {mjcf_file}\n请先运行: python ruiyan_hand/convert_urdf_to_mjcf_v2.py <urdf_file>")
        
        print(f"[INFO] 加载{hand_type}手MJCF场景: {mjcf_file}")
        
        # 直接加载MJCF（MuJoCo原生格式，无需转换）
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_file))
        self.configuration = mink.Configuration(self.model)
        
        self.data_lock = threading.Lock()
        
        # 获取关节和site信息
        self._setup_joint_info()
        self._setup_site_info()
        
        # 设置IK任务
        self._setup_tasks()
        
        # 设置关节限制
        self._setup_limits()
        
        # 启动viewer（如果需要）
        self.viewer = None
        if self.enable_viewer:
            self._start_viewer()
        
        # 初始化mink环境
        self._init_mink_env()
        
        # 预热仿真
        self._warm_up_sim()
        
        print(f"[INFO] IK求解器初始化完成")
        print(f"[INFO] 关节数量: {len(self.joint_names)} ({self.model.nu} DOF)")
        print(f"[INFO] 指尖数量: {len(self.fingertip_sites)}")
        print(f"[INFO] 求解器: {self.solver}")
    
    def _setup_joint_info(self):
        """设置关节信息"""
        # 获取所有关节名称
        all_joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                     for i in range(self.model.njnt)]
        
        # 主动关节（非mimic）
        prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        active_joints = [
            f'{prefix}_joint_link_1_1',  # 拇指第1关节
            f'{prefix}_joint_link_1_2',  # 拇指第2关节
            f'{prefix}_joint_link_2_1',  # 食指
            f'{prefix}_joint_link_3_1',  # 中指
            f'{prefix}_joint_link_4_1',  # 无名指
            f'{prefix}_joint_link_5_1',  # 小指
        ]
        
        self.joint_names = [j for j in active_joints if j in all_joints]
        
        # 存储关节限制（用于归一化转换）
        self.joint_limits = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            lower = self.model.jnt_range[joint_id][0]
            upper = self.model.jnt_range[joint_id][1]
            self.joint_limits[joint_name] = {'lower': lower, 'upper': upper, 'range': upper - lower}
    
    def _setup_site_info(self):
        """设置指尖site信息（MJCF转换后已包含site定义）"""
        side_prefix = self.hand_type  # 'left' 或 'right'
        
        # 指尖site名称（MJCF转换后生成的site）
        self.fingertip_sites = [
            f'{side_prefix}_thumb_tip',   # 拇指site
            f'{side_prefix}_index_tip',   # 食指site
            f'{side_prefix}_middle_tip',  # 中指site
            f'{side_prefix}_ring_tip',    # 无名指site
            f'{side_prefix}_pinky_tip',   # 小指site
        ]
        
        # 获取site ID
        self.site_ids = {}
        for site_name in self.fingertip_sites:
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                self.site_ids[site_name] = site_id
                print(f"[DEBUG] 找到site: {site_name} (id={site_id})")
            except:
                print(f"[WARNING] 未找到site: {site_name}")
    
    def _setup_tasks(self):
        """
        设置IK任务
        
        最简配置 - 只保留位置约束：
        - 去除所有正则化约束（PostureTask, DampingTask, KineticEnergyTask）
        - 只保留指尖位置任务
        - 专注于精确到达目标位置
        - 使用速度范数判断收敛（||vel|| < 1e-8）
        """
        # 任务列表为空（不使用姿态正则化）
        self.tasks = []
        
        # 为每个指尖创建帧任务（使用site）
        self.hand_tasks = []
        for site_name in self.fingertip_sites:
            task = mink.FrameTask(
                frame_name=site_name,
                frame_type="site",  # 使用site（MJCF中定义的）
                position_cost=1000.0,     # 位置权重
                orientation_cost=0.001,   # 姿态权重（极低）
                lm_damping=0.001,         # LM阻尼（极低）
            )
            self.hand_tasks.append(task)
        
        self.tasks.extend(self.hand_tasks)
        
        print(f"[DEBUG] 创建了 {len(self.hand_tasks)} 个指尖IK任务（纯位置约束）")
    
    def _setup_limits(self):
        """设置关节限制"""
        mink_joint_limit = mink.ConfigurationLimit(self.model)
        self.limits = [mink_joint_limit]
    
    def _init_mink_env(self):
        """初始化mink环境"""
        try:
            self.configuration.update_from_keyframe("home")
        except:
            # 如果没有home keyframe，使用零位
            print("[WARNING] 没有找到'home' keyframe，使用零位")
            self.configuration.q = np.zeros(self.model.nq)
        
        with self.data_lock:
            # 打印初始site位置
            for site_name in self.fingertip_sites:
                if site_name in self.site_ids:
                    site_id = self.site_ids[site_name]
                    site_pos = self.configuration.data.site(site_id).xpos.copy()
                    print(f"[DEBUG] {site_name} 初始位置: {site_pos}")
    
    def _normalized_to_radians(self, normalized_positions):
        """
        将归一化值（0-1）转换为弧度
        
        Args:
            normalized_positions: 归一化关节角度数组（0-1）
        
        Returns:
            弧度关节角度数组
        """
        radians = np.zeros_like(normalized_positions)
        for i, joint_name in enumerate(self.joint_names):
            limits = self.joint_limits[joint_name]
            radians[i] = limits['lower'] + normalized_positions[i] * limits['range']
        return radians
    
    def _radians_to_normalized(self, radian_positions):
        """
        将弧度转换为归一化值（0-1）
        
        Args:
            radian_positions: 弧度关节角度数组
        
        Returns:
            归一化关节角度数组（0-1）
        """
        normalized = np.zeros_like(radian_positions)
        for i, joint_name in enumerate(self.joint_names):
            limits = self.joint_limits[joint_name]
            normalized[i] = (radian_positions[i] - limits['lower']) / limits['range']
        return normalized
    
    def _apply_mimic_joints(self):
        """
        手动应用mimic joint约束
        
        MuJoCo的equality约束只在动力学仿真中生效，
        在FK/IK的纯运动学计算中需要手动应用
        """
        hand_prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        
        # Mimic关系（从URDF提取）：
        mimic_relations = [
            (f'{hand_prefix}_joint_link_1_2', f'{hand_prefix}_joint_link_1_3', 1.675, 0.0),  # 拇指
            (f'{hand_prefix}_joint_link_2_1', f'{hand_prefix}_joint_link_2_2', 1.0, 0.0),    # 食指
            (f'{hand_prefix}_joint_link_3_1', f'{hand_prefix}_joint_link_3_2', 1.0, 0.0),    # 中指
            (f'{hand_prefix}_joint_link_4_1', f'{hand_prefix}_joint_link_4_2', 1.0, 0.0),    # 无名指
            (f'{hand_prefix}_joint_link_5_1', f'{hand_prefix}_joint_link_5_2', 1.0, 0.0),    # 小指
        ]
        
        with self.data_lock:
            for leader_name, follower_name, multiplier, offset in mimic_relations:
                try:
                    leader_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name)
                    follower_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                    
                    # 应用约束: follower = offset + multiplier * leader
                    self.configuration.data.qpos[follower_id] = offset + multiplier * self.configuration.data.qpos[leader_id]
                except:
                    pass  # 某些关节可能找不到
    
    def _start_viewer(self):
        """启动MuJoCo viewer"""
        print("[INFO] 启动MuJoCo Viewer...")
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.configuration.data,
            show_left_ui=True,
            show_right_ui=True
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        print("[INFO] MuJoCo Viewer已启动")
    
    def _draw_coordinate_frame(self, pos, rot_matrix, scale=0.02):
        """
        在MuJoCo viewer中绘制坐标系
        
        Args:
            pos: 位置 [x, y, z]
            rot_matrix: 3x3旋转矩阵
            scale: 坐标轴长度
        """
        if not self.enable_viewer or not self.viewer:
            return
        
        # 检查 viewer 是否有 user_scn 属性（不同版本的 MuJoCo 可能不同）
        if not hasattr(self.viewer, 'user_scn'):
            return
        
        scn = self.viewer.user_scn
        
        # X轴（红色）
        x_axis_end = pos + rot_matrix[:, 0] * scale
        from_pos = np.array([pos[0], pos[1], pos[2]])
        to_pos = np.array([x_axis_end[0], x_axis_end[1], x_axis_end[2]])
        
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                scale * 0.05,
                from_pos,
                to_pos
            )
            scn.geoms[scn.ngeom].rgba[:] = [1.0, 0.0, 0.0, 1.0]
            scn.ngeom += 1
        
        # Y轴（绿色）
        y_axis_end = pos + rot_matrix[:, 1] * scale
        to_pos = np.array([y_axis_end[0], y_axis_end[1], y_axis_end[2]])
        
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                scale * 0.05,
                from_pos,
                to_pos
            )
            scn.geoms[scn.ngeom].rgba[:] = [0.0, 1.0, 0.0, 1.0]
            scn.ngeom += 1
        
        # Z轴（蓝色）
        z_axis_end = pos + rot_matrix[:, 2] * scale
        to_pos = np.array([z_axis_end[0], z_axis_end[1], z_axis_end[2]])
        
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                scale * 0.05,
                from_pos,
                to_pos
            )
            scn.geoms[scn.ngeom].rgba[:] = [0.0, 0.0, 1.0, 1.0]
            scn.ngeom += 1
    
    def _warm_up_sim(self):
        """预热仿真"""
        for _ in range(10):
            with self.data_lock:
                mujoco.mj_forward(self.model, self.configuration.data)
                if self.enable_viewer and self.viewer:
                    self.viewer.sync()
                time.sleep(0.01)
    
    def compute_ik(self, target_poses, max_iterations=10, return_normalized=False):
        """
        计算逆运动学（不使用mocap，直接设置IK目标）
        
        默认迭代次数10次，经过测试为最优配置
        
        Args:
            target_poses: 目标位姿字典
                {
                    'thumb': {'pos': [x,y,z], 'quat': [x,y,z,w]},  # 可选
                    'index': {'pos': [x,y,z], 'quat': [x,y,z,w]},  # 可选
                    ...
                }
                如果不指定某个手指，则不会控制该手指
            max_iterations: 最大迭代次数
            return_normalized: 是否返回归一化值(0-1)（默认False，返回弧度）
        
        Returns:
            dict: 
                {
                    'joint_positions': 关节角度数组（弧度或归一化值，取决于return_normalized）,
                    'joint_positions_normalized': 归一化关节角度（仅当return_normalized=True时包含）,
                    'error': 残差误差,
                    'iterations': 实际迭代次数
                }
        """
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        # 先为所有手指设置当前位置作为默认目标（保持不动）
        with self.data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)
            for i, site_name in enumerate(self.fingertip_sites):
                site_id = self.site_ids[site_name]
                current_pos = self.configuration.data.site(site_id).xpos.copy()
                current_rot = self.configuration.data.site(site_id).xmat.reshape(3, 3).copy()
                
                transform = np.eye(4)
                transform[:3, :3] = current_rot
                transform[:3, 3] = current_pos
                target_se3 = mink.SE3.from_matrix(transform)
                self.hand_tasks[i].set_target(target_se3)
        
        # IK求解循环
        vel_error = 0.0
        position_error = 0.0
        for iteration in range(max_iterations):
            with self.data_lock:
                # 更新MuJoCo仿真
                mujoco.mj_forward(self.model, self.configuration.data)
                
                # 绘制坐标系（如果启用了viewer）
                if self.enable_viewer and self.viewer and hasattr(self.viewer, 'user_scn'):
                    # 清除之前的绘制
                    self.viewer.user_scn.ngeom = 0
                    
                    # 1. 绘制世界坐标系（原点在hand_base）
                    world_origin = np.array([0.0, 0.0, 0.0])
                    world_rot = np.eye(3)
                    self._draw_coordinate_frame(world_origin, world_rot, scale=0.05)
                    
                    # 2. 绘制各指尖坐标系
                    for site_name in self.fingertip_sites:
                        if site_name in self.site_ids:
                            site_id = self.site_ids[site_name]
                            pos = self.configuration.data.site(site_id).xpos.copy()
                            rot = self.configuration.data.site(site_id).xmat.reshape(3, 3).copy()
                            self._draw_coordinate_frame(pos, rot, scale=0.015)
                
                # 更新viewer
                if self.enable_viewer and self.viewer:
                    self.viewer.sync()
            
            # 更新任务目标（只更新target_poses中指定的手指）
            for i, (finger_name, site_name) in enumerate(
                zip(finger_names, self.fingertip_sites)
            ):
                if finger_name in target_poses:
                    target_pose = target_poses[finger_name]
                    
                    # 创建SE3目标
                    pos = target_pose.get('pos', np.zeros(3))
                    
                    if 'quat' in target_pose:
                        # 输入是 [x,y,z,w]，需要转换为旋转矩阵
                        quat_xyzw = target_pose['quat']
                        rot_mat = R.from_quat(quat_xyzw).as_matrix()
                    else:
                        # 如果没有姿态，使用当前姿态
                        with self.data_lock:
                            site_id = self.site_ids[site_name]
                            rot_mat = self.configuration.data.site(site_id).xmat.reshape(3, 3)
                    
                    # 创建SE3变换
                    transform = np.eye(4)
                    transform[:3, :3] = rot_mat
                    transform[:3, 3] = pos
                    
                    target_se3 = mink.SE3.from_matrix(transform)
                    self.hand_tasks[i].set_target(target_se3)
            
            # 求解IK
            vel = mink.solve_ik(
                configuration=self.configuration,
                tasks=self.tasks,
                dt=self.dt,
                solver=self.solver,
                damping=1e-5,
                safety_break=False,
                limits=self.limits,
            )
            
            # 积分更新关节位置
            self.configuration.integrate_inplace(vel, self.dt)
            
            # 应用mimic joints约束
            self._apply_mimic_joints()
            
            # 计算速度范数（用于收敛判断）
            vel_error = np.linalg.norm(vel)
            
            # 收敛判断：使用速度范数
            if vel_error < 1e-8:  # 速度范数阈值
                break
        
        # 计算最终位置误差（用于返回信息）
        with self.data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)
            max_position_error = 0.0
            per_finger_errors = {}
            total_error = 0.0
            n_active_fingers = 0
            for i, (finger_name, site_name) in enumerate(zip(finger_names, self.fingertip_sites)):
                if finger_name in target_poses:
                    site_id = self.site_ids[site_name]
                    current_pos = self.configuration.data.site(site_id).xpos.copy()
                    target_pos = target_poses[finger_name]['pos']
                    pos_error = np.linalg.norm(current_pos - target_pos)
                    max_position_error = max(max_position_error, pos_error)
                    per_finger_errors[finger_name] = pos_error
                    total_error += pos_error
                    n_active_fingers += 1
        
        position_error = max_position_error
        mean_position_error = total_error / max(n_active_fingers, 1)
        
        # 获取求解的关节角度
        with self.data_lock:
            joint_positions = self.configuration.data.qpos.copy()
        
        # 提取主动关节位置（用于FK验证）
        active_joint_positions = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            active_joint_positions.append(joint_positions[qpos_adr])
        
        active_joint_positions = np.array(active_joint_positions)
        
        result = {
            'joint_positions': joint_positions,          # 所有关节（包括mimic）
            'active_joint_positions': active_joint_positions,  # 仅主动关节
            'position_error': position_error,            # 最大指尖位置误差（m）
            'mean_position_error': mean_position_error,  # 平均指尖位置误差（m）
            'per_finger_errors': per_finger_errors,      # 每指位置误差（m）
            'velocity_error': vel_error,                 # 速度范数
            'error': vel_error,                          # 主要误差指标（速度范数，向后兼容）
            'iterations': iteration + 1,
            'converged': vel_error < 1e-8                # 是否收敛（基于速度范数）
        }
        
        # 如果需要归一化输出，添加归一化后的关节角度
        if return_normalized:
            result['joint_positions_normalized'] = self._radians_to_normalized(active_joint_positions)
        
        return result
    
    def interactive_mode(self):
        """交互模式：循环输入目标位姿并显示结果"""
        print("\n" + "="*60)
        print(f"RuiYan {self.hand_type.upper()} Hand IK 求解器 - 交互模式")
        print("="*60)
        print("\n可控制的手指：")
        print("  1. 拇指 (thumb)")
        print("  2. 食指 (index)")
        print("  3. 中指 (middle)")
        print("  4. 无名指 (ring)")
        print("  5. 小指 (pinky)")
        print("  all. 所有手指")
        print("\n输入 'q' 或 'quit' 退出\n")
        
        while True:
            try:
                # 选择要控制的手指
                finger_input = input("\n选择要移动的手指（1-5 或 all）: ").strip().lower()
                
                if finger_input in ['q', 'quit', 'exit']:
                    print("退出程序")
                    break
                
                # 解析手指选择
                finger_map = {
                    '1': 'thumb',
                    '2': 'index',
                    '3': 'middle',
                    '4': 'ring',
                    '5': 'pinky'
                }
                
                if finger_input == 'all':
                    selected_fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
                elif finger_input in finger_map:
                    selected_fingers = [finger_map[finger_input]]
                else:
                    print("[错误] 无效的选择")
                    continue
                
                # 输入目标位姿
                target_poses = {}
                
                for finger in selected_fingers:
                    finger_cn = {
                        'thumb': '拇指',
                        'index': '食指',
                        'middle': '中指',
                        'ring': '无名指',
                        'pinky': '小指'
                    }[finger]
                    
                    print(f"\n--- {finger_cn} ---")
                    pos_input = input(f"输入{finger_cn}目标位置 (x y z): ").strip()
                    
                    if not pos_input:
                        continue
                    
                    pos = [float(x) for x in pos_input.split()]
                    if len(pos) != 3:
                        print(f"[错误] 位置需要3个值")
                        continue
                    
                    quat_input = input(f"输入{finger_cn}目标姿态 (x y z w，可选，按回车跳过): ").strip()
                    
                    target_poses[finger] = {'pos': np.array(pos)}
                    
                    if quat_input:
                        quat = [float(x) for x in quat_input.split()]
                        if len(quat) != 4:
                            print(f"[错误] 四元数需要4个值")
                            continue
                        target_poses[finger]['quat'] = np.array(quat)
                
                if not target_poses:
                    print("[错误] 没有有效的目标位姿")
                    continue
                
                # 求解IK
                print(f"\n[求解中] IK求解...")
                t0 = time.time()
                result = self.compute_ik(target_poses, max_iterations=10)
                t1 = time.time()
                
                # 显示结果
                print(f"\n[结果] IK求解完成 (耗时: {(t1-t0)*1000:.2f}ms)")
                print(f"  迭代次数: {result['iterations']}")
                print(f"  速度范数: {result['velocity_error']:.6f}")
                print(f"  位置误差: {result['position_error']*1000:.3f} mm")
                print(f"  收敛状态: {'✅ 收敛' if result['converged'] else '⚠️  未完全收敛'}")
                print(f"\n求解的关节角度:")
                joint_positions = result['joint_positions']
                for i, joint_name in enumerate(self.joint_names):
                    if i < len(joint_positions):
                        print(f"  {joint_name}: {joint_positions[i]:.4f} rad ({np.degrees(joint_positions[i]):.2f}°)")
                
            except ValueError as e:
                print(f"[错误] 输入格式错误: {e}")
            except KeyboardInterrupt:
                print("\n\n收到中断信号，退出程序")
                break
            except Exception as e:
                print(f"[错误] {e}")
                import traceback
                traceback.print_exc()
    
    def close(self):
        """关闭求解器"""
        if self.viewer:
            self.viewer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RuiYan Hand IK Solver')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'],
                       help='选择左手或右手 (默认: left)')
    parser.add_argument('--viewer', action='store_true', default=True,
                       help='启用MuJoCo可视化')
    parser.add_argument('--no-viewer', action='store_false', dest='viewer',
                       help='禁用MuJoCo可视化')
    parser.add_argument('--solver', type=str, default='daqp',
                       choices=['daqp', 'quadprog', 'proxqp'],
                       help='IK求解器类型 (默认: daqp)')
    parser.add_argument('--frequency', type=float, default=100.0,
                       help='求解频率(Hz) (默认: 100)')
    
    args = parser.parse_args()
    
    try:
        # 创建IK求解器
        ik_solver = HandIK(
            hand_type=args.hand,
            enable_viewer=args.viewer,
            solver=args.solver,
            frequency=args.frequency
        )
        
        # 进入交互模式
        ik_solver.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ik_solver' in locals():
            ik_solver.close()


if __name__ == '__main__':
    main()

