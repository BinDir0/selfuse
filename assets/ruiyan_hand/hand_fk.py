"""
RuiYan Hand FK Solver

基于 MuJoCo 实现的灵巧手正运动学求解器
支持左手和右手，提供交互式命令行界面

使用方法：
    python hand_fk.py --hand left --viewer
    
功能：
    - 输入关节角度 -> 输出5个指尖的位姿
    - MuJoCo 可视化（可选）
    - 无 ROS2 依赖
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import os
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import threading


class HandFK:
    """
    灵巧手正运动学求解器
    
    功能：
        - 根据关节角度计算指尖位姿（FK）
        - 支持左手/右手
        - 可视化支持
    """
    
    def __init__(self, hand_type='left', enable_viewer=True):
        """
        初始化FK求解器
        
        Args:
            hand_type: 'left' 或 'right'
            enable_viewer: 是否启用MuJoCo可视化
        """
        self.hand_type = hand_type
        self.enable_viewer = enable_viewer
        
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
        self.data = mujoco.MjData(self.model)
        
        # 获取关节信息
        self._setup_joint_info()
        
        # 获取site信息
        self._setup_site_info()
        
        # 启动viewer（如果需要）
        self.viewer = None
        self.data_lock = threading.Lock()
        if self.enable_viewer:
            self._start_viewer()
        
        # 预热仿真
        self._warm_up_sim()
        
        print(f"[INFO] FK求解器初始化完成")
        print(f"[INFO] 关节数量: {len(self.joint_names)} ({self.model.nu} 个可控关节)")
        print(f"[INFO] 指尖数量: {len(self.fingertip_sites)}")
        print(f"[INFO] 关节名称: {self.joint_names}")
    
    def _setup_joint_info(self):
        """设置关节信息"""
        # 获取所有关节名称
        all_joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                     for i in range(self.model.njnt)]
        
        # 过滤掉固定关节（只保留有自由度的关节）
        self.joint_names = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = self.model.jnt_type[i]
            # 0=free, 1=ball, 2=slide, 3=hinge (revolute)
            if joint_type == 3:  # revolute joint
                # 检查是否有对应的actuator或自由度
                qpos_adr = self.model.jnt_qposadr[i]
                if qpos_adr < self.model.nq:
                    self.joint_names.append(joint_name)
        
        # 对于灵巧手，只保留非mimic的主动关节
        # 根据URDF，主动关节应该是：
        # hand1_joint_link_1_1, hand1_joint_link_1_2, 
        # hand1_joint_link_2_1, hand1_joint_link_3_1, 
        # hand1_joint_link_4_1, hand1_joint_link_5_1
        prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        active_joints = [
            f'{prefix}_joint_link_1_1',  # 拇指第1关节
            f'{prefix}_joint_link_1_2',  # 拇指第2关节
            f'{prefix}_joint_link_2_1',  # 食指
            f'{prefix}_joint_link_3_1',  # 中指
            f'{prefix}_joint_link_4_1',  # 无名指
            f'{prefix}_joint_link_5_1',  # 小指
        ]
        
        # 保留实际存在的主动关节
        self.joint_names = [j for j in active_joints if j in all_joints]
        
        # 存储关节限制（用于归一化转换）
        self.joint_limits = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            lower = self.model.jnt_range[joint_id][0]
            upper = self.model.jnt_range[joint_id][1]
            self.joint_limits[joint_name] = {'lower': lower, 'upper': upper, 'range': upper - lower}
        
        print(f"[DEBUG] 所有关节: {all_joints}")
        print(f"[DEBUG] 主动关节: {self.joint_names}")
    
    def _setup_site_info(self):
        """设置指尖site信息（与IK保持一致）"""
        # 使用site而不是body，确保FK和IK使用相同的参考点
        side_prefix = self.hand_type  # 'left' 或 'right'
        
        # 5个指尖site名称（MJCF转换后生成的site）
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
                    self.data.qpos[follower_id] = offset + multiplier * self.data.qpos[leader_id]
                except:
                    pass  # 某些关节可能找不到
    
    def _start_viewer(self):
        """启动MuJoCo viewer"""
        print("[INFO] 启动MuJoCo Viewer...")
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
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
                mujoco.mj_forward(self.model, self.data)
                if self.enable_viewer and self.viewer:
                    self.viewer.sync()
                time.sleep(0.01)
    
    def compute_fk(self, joint_positions, use_normalized=False):
        """
        计算正运动学
        
        Args:
            joint_positions: 关节角度数组
                            - 如果 use_normalized=False: 弧度值
                            - 如果 use_normalized=True: 归一化值(0-1)
                            长度应为主动关节数量（通常是6个）
            use_normalized: 是否使用归一化输入（默认False，使用弧度）
        
        Returns:
            dict: 每个指尖的位姿
                {
                    'thumb': {'pos': [x,y,z], 'quat': [x,y,z,w]},
                    'index': {'pos': [x,y,z], 'quat': [x,y,z,w]},
                    ...
                }
        """
        joint_positions = np.array(joint_positions)
        
        if len(joint_positions) != len(self.joint_names):
            raise ValueError(
                f"关节数量不匹配: 期望 {len(self.joint_names)}, 实际 {len(joint_positions)}"
            )
        
        # 如果是归一化输入，转换为弧度
        if use_normalized:
            joint_positions = self._normalized_to_radians(joint_positions)
        
        # 更新关节位置
        with self.data_lock:
            # 设置主动关节的位置
            for i, joint_name in enumerate(self.joint_names):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr] = joint_positions[i]
        
        # 应用mimic joints约束（在mj_forward之前）
        self._apply_mimic_joints()
        
        with self.data_lock:
            # 执行正向运动学
            mujoco.mj_forward(self.model, self.data)
            
            # 读取指尖位姿（使用body）
            results = {}
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
            for finger_name, site_name in zip(finger_names, self.fingertip_sites):
                if site_name in self.site_ids:
                    site_id = self.site_ids[site_name]
                    
                    # 获取site位置（与IK保持一致）
                    pos = self.data.site(site_id).xpos.copy()
                    
                    # 获取site姿态（旋转矩阵 -> 四元数）
                    xmat = self.data.site(site_id).xmat.copy().reshape(3, 3)
                    quat = R.from_matrix(xmat).as_quat()  # [x, y, z, w]
                    
                    results[finger_name] = {
                        'pos': pos,
                        'quat': quat,
                        'rot': xmat  # 保存旋转矩阵用于绘制坐标系
                    }
            
            # 绘制坐标系（如果启用了viewer）
            if self.enable_viewer and self.viewer and hasattr(self.viewer, 'user_scn'):
                # 清除之前的绘制
                self.viewer.user_scn.ngeom = 0
                
                # 1. 绘制世界坐标系（原点在hand_base）
                world_origin = np.array([0.0, 0.0, 0.0])
                world_rot = np.eye(3)
                self._draw_coordinate_frame(world_origin, world_rot, scale=0.05)
                
                # 2. 绘制各指尖坐标系
                for finger_name in finger_names:
                    if finger_name in results:
                        pos = results[finger_name]['pos']
                        rot = results[finger_name]['rot']
                        self._draw_coordinate_frame(pos, rot, scale=0.015)
            
            # 更新viewer
            if self.enable_viewer and self.viewer:
                self.viewer.sync()
        
        return results
    
    def interactive_mode(self):
        """交互模式：循环输入关节角度并显示结果"""
        print("\n" + "="*60)
        print(f"RuiYan {self.hand_type.upper()} Hand FK 求解器 - 交互模式")
        print("="*60)
        
        # 询问用户使用哪种输入方式
        print("\n选择输入方式：")
        print("  1. 归一化值 (0-1，推荐) - 0=完全伸展，1=完全弯曲")
        print("  2. 弧度值 (高级)")
        
        while True:
            input_mode = input("\n请选择 (1 或 2，默认为 1): ").strip()
            if input_mode == '' or input_mode == '1':
                use_normalized = True
                input_unit = "归一化值 (0-1)"
                example_input = "0.3 0.5 0.8 0.8 0.8 0.8"
                break
            elif input_mode == '2':
                use_normalized = False
                input_unit = "弧度"
                example_input = "0.5 0.3 1.2 1.2 1.2 1.2"
                break
            else:
                print("[错误] 请输入 1 或 2")
        
        print(f"\n✓ 已选择：{input_unit}")
        print(f"\n需要输入 {len(self.joint_names)} 个关节角度")
        print("关节顺序：")
        for i, joint_name in enumerate(self.joint_names):
            limits = self.joint_limits[joint_name]
            if use_normalized:
                print(f"  {i+1}. {joint_name} (0.0 ~ 1.0)")
            else:
                print(f"  {i+1}. {joint_name} ({limits['lower']:.3f} ~ {limits['upper']:.3f} rad)")
        
        print(f"\n示例输入: {example_input}")
        print("输入 'q' 或 'quit' 退出\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input(f"\n请输入{len(self.joint_names)}个关节角度（空格分隔）: ").strip()
                
                # 检查退出命令
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("退出程序")
                    break
                
                # 解析输入
                joint_values = [float(x) for x in user_input.split()]
                
                if len(joint_values) != len(self.joint_names):
                    print(f"[错误] 需要{len(self.joint_names)}个值，实际输入了{len(joint_values)}个")
                    continue
                
                # 检查归一化值范围
                if use_normalized:
                    if any(v < 0 or v > 1 for v in joint_values):
                        print(f"[警告] 归一化值应该在 0-1 之间，当前输入: {joint_values}")
                        confirm = input("是否继续？(y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                
                # 计算FK
                t0 = time.time()
                results = self.compute_fk(joint_values, use_normalized=use_normalized)
                t1 = time.time()
                
                # 显示结果
                print(f"\n[结果] FK计算完成 (耗时: {(t1-t0)*1000:.2f}ms)")
                print("-" * 60)
                
                for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                    if finger_name in results:
                        pos = results[finger_name]['pos']
                        quat = results[finger_name]['quat']
                        
                        finger_cn = {
                            'thumb': '拇指',
                            'index': '食指',
                            'middle': '中指',
                            'ring': '无名指',
                            'pinky': '小指'
                        }[finger_name]
                        
                        print(f"{finger_cn:>4}: pos=[{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}], "
                              f"quat=[{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]")
                
                print("-" * 60)
                
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
    parser = argparse.ArgumentParser(description='RuiYan Hand FK Solver')
    parser.add_argument('--hand', type=str, default='left', choices=['left', 'right'],
                       help='选择左手或右手 (默认: left)')
    parser.add_argument('--viewer', action='store_true', default=True,
                       help='启用MuJoCo可视化')
    parser.add_argument('--no-viewer', action='store_false', dest='viewer',
                       help='禁用MuJoCo可视化')
    
    args = parser.parse_args()
    
    try:
        # 创建FK求解器
        fk_solver = HandFK(hand_type=args.hand, enable_viewer=args.viewer)
        
        # 进入交互模式
        fk_solver.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'fk_solver' in locals():
            fk_solver.close()


if __name__ == '__main__':
    main()

