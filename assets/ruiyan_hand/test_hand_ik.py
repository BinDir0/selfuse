"""
Hand IK 交互测试脚本

在 MuJoCo Viewer 中拖拽彩色目标球来测试手部 IK 精度。
手指会实时跟随目标球移动。

使用方法:
    python test_hand_ik.py --hand left
    python test_hand_ik.py --hand right

操作说明:
    1. 双击一个彩色球体选中它
    2. Ctrl + 鼠标右键拖拽 → 在相机平面内移动球体
    3. Ctrl + 鼠标右键 + Shift → 沿相机视线方向移动
    4. 手指会实时跟踪目标球位置

    颜色对应:
        红色 = 拇指    绿色 = 食指    蓝色 = 中指
        黄色 = 无名指  紫色 = 小指
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import mink
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# ============================================================
# 常量定义
# ============================================================
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_CN = {
    'thumb': '拇指', 'index': '食指', 'middle': '中指',
    'ring': '无名指', 'pinky': '小指'
}
# mocap球体的颜色 (r g b a)，与原始site颜色对应
TARGET_COLORS = [
    '1 0.3 0.3 0.7',     # 红 - 拇指
    '0.3 1 0.3 0.7',     # 绿 - 食指
    '0.3 0.3 1 0.7',     # 蓝 - 中指
    '1 1 0.3 0.7',       # 黄 - 无名指
    '1 0.3 1 0.7',       # 紫 - 小指
]


# ============================================================
# XML 增强: 在原始场景中添加可拖拽的 mocap 目标球体
# ============================================================
def create_augmented_xml(hand_type):
    """
    读取原始 MJCF，在 worldbody 中添加 5 个 mocap 目标球体，
    保存为临时文件并返回路径。
    """
    hand_prefix = 'Left' if hand_type == 'left' else 'Right'
    script_dir = Path(__file__).parent
    mjcf_dir = (script_dir / 'InspiredHand_RuiYan' / '0611_v1.4' / 'Version_3.0'
                / f'RuiYan_Hand_{hand_prefix}_Mimic' / 'meshes')
    mjcf_file = mjcf_dir / f'RuiYan_Hand_{hand_prefix}_Mimic_scene.xml'

    if not mjcf_file.exists():
        raise FileNotFoundError(f"MJCF文件不存在: {mjcf_file}")

    with open(mjcf_file, 'r') as f:
        xml = f.read()

    # 构造 mocap 目标体 XML 片段
    mocap_xml = '\n    <!-- IK测试: 可拖拽目标球体 -->\n'
    for finger, color in zip(FINGER_NAMES, TARGET_COLORS):
        mocap_xml += (
            f'    <body name="target_{finger}" mocap="true" pos="0 0 0.15">\n'
            f'      <geom type="sphere" size="0.008" rgba="{color}" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n'
        )

    # 插入到 </worldbody> 之前
    xml = xml.replace('</worldbody>', mocap_xml + '  </worldbody>')

    # 保存到同目录（保证 mesh 相对路径正确）
    temp_path = mjcf_dir / f'_test_ik_{hand_type}.xml'
    with open(temp_path, 'w') as f:
        f.write(xml)

    return str(temp_path)


# ============================================================
# HandIKTester: 交互式 IK 测试器
# ============================================================
class HandIKTester:

    def __init__(self, hand_type='left'):
        self.hand_type = hand_type
        self.hand_prefix = 'hand1' if hand_type == 'left' else 'hand2'
        self.side_prefix = hand_type   # 'left' / 'right'

        # 创建增强模型
        print(f"[INFO] 创建 {hand_type} 手测试模型...")
        xml_path = create_augmented_xml(hand_type)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.configuration = mink.Configuration(self.model)

        self._setup()

    # ----------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------
    def _setup(self):
        """设置关节、site、IK 任务"""
        # 指尖 site 名称
        self.fingertip_sites = [
            f'{self.side_prefix}_{f}_tip' for f in FINGER_NAMES
        ]

        # 获取 site ID
        self.site_ids = {}
        for s in self.fingertip_sites:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
            if sid >= 0:
                self.site_ids[s] = sid
                print(f"  [OK] site: {s} (id={sid})")

        # 获取 mocap body 的 mocap 索引（用于读写 data.mocap_pos）
        self.target_mocap_ids = {}
        for finger in FINGER_NAMES:
            name = f'target_{finger}'
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                mid = self.model.body_mocapid[bid]
                if mid >= 0:
                    self.target_mocap_ids[finger] = mid

        # 主动关节
        all_joints = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        active = [
            f'{self.hand_prefix}_joint_link_1_1',
            f'{self.hand_prefix}_joint_link_1_2',
            f'{self.hand_prefix}_joint_link_2_1',
            f'{self.hand_prefix}_joint_link_3_1',
            f'{self.hand_prefix}_joint_link_4_1',
            f'{self.hand_prefix}_joint_link_5_1',
        ]
        self.joint_names = [j for j in active if j in all_joints]

        # mimic 关节关系
        self.mimic_relations = [
            (f'{self.hand_prefix}_joint_link_1_2',
             f'{self.hand_prefix}_joint_link_1_3', 1.675, 0.0),
            (f'{self.hand_prefix}_joint_link_2_1',
             f'{self.hand_prefix}_joint_link_2_2', 1.0, 0.0),
            (f'{self.hand_prefix}_joint_link_3_1',
             f'{self.hand_prefix}_joint_link_3_2', 1.0, 0.0),
            (f'{self.hand_prefix}_joint_link_4_1',
             f'{self.hand_prefix}_joint_link_4_2', 1.0, 0.0),
            (f'{self.hand_prefix}_joint_link_5_1',
             f'{self.hand_prefix}_joint_link_5_2', 1.0, 0.0),
        ]

        # IK 任务: 每个指尖一个 FrameTask（仅位置约束）
        self.tasks = []
        self.hand_tasks = []
        for site_name in self.fingertip_sites:
            task = mink.FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.001,
                lm_damping=0.001,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)

        # 关节限制
        self.limits = [mink.ConfigurationLimit(self.model)]

        # 初始化配置（home keyframe）
        try:
            self.configuration.update_from_keyframe("home")
        except Exception:
            self.configuration.q = np.zeros(self.model.nq)

        # 运行一次 FK
        mujoco.mj_forward(self.model, self.configuration.data)

        # 把 mocap 目标球体放到当前指尖位置
        for i, finger in enumerate(FINGER_NAMES):
            site_name = self.fingertip_sites[i]
            if site_name in self.site_ids and finger in self.target_mocap_ids:
                pos = self.configuration.data.site(
                    self.site_ids[site_name]).xpos.copy()
                self.configuration.data.mocap_pos[
                    self.target_mocap_ids[finger]] = pos

        # 预热
        for _ in range(10):
            mujoco.mj_forward(self.model, self.configuration.data)

        print(f"[INFO] 初始化完成 (关节: {len(self.joint_names)}, "
              f"指尖: {len(self.site_ids)})")

    # ----------------------------------------------------------
    # Mimic 关节
    # ----------------------------------------------------------
    def _apply_mimic(self):
        for leader, follower, mult, offset in self.mimic_relations:
            try:
                lid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, leader)
                fid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, follower)
                self.configuration.data.qpos[fid] = (
                    offset + mult * self.configuration.data.qpos[lid])
            except Exception:
                pass

    # ----------------------------------------------------------
    # IK 单步
    # ----------------------------------------------------------
    def _ik_step(self):
        dt = 0.01   # 100Hz

        mujoco.mj_forward(self.model, self.configuration.data)

        # 为每个指尖设置目标（从 mocap 球体读取位置）
        for i, finger in enumerate(FINGER_NAMES):
            site_name = self.fingertip_sites[i]
            if finger in self.target_mocap_ids and site_name in self.site_ids:
                target_pos = self.configuration.data.mocap_pos[
                    self.target_mocap_ids[finger]].copy()
                site_id = self.site_ids[site_name]
                current_rot = self.configuration.data.site(
                    site_id).xmat.reshape(3, 3).copy()

                T = np.eye(4)
                T[:3, :3] = current_rot
                T[:3, 3] = target_pos
                self.hand_tasks[i].set_target(mink.SE3.from_matrix(T))

        # 求解
        vel = mink.solve_ik(
            configuration=self.configuration,
            tasks=self.tasks,
            dt=dt,
            solver='daqp',
            damping=1e-5,
            safety_break=False,
            limits=self.limits,
        )

        self.configuration.integrate_inplace(vel, dt)
        self._apply_mimic()

        return np.linalg.norm(vel)

    # ----------------------------------------------------------
    # 误差计算
    # ----------------------------------------------------------
    def _get_errors(self):
        mujoco.mj_forward(self.model, self.configuration.data)
        errors = {}
        for i, finger in enumerate(FINGER_NAMES):
            site_name = self.fingertip_sites[i]
            if site_name in self.site_ids and finger in self.target_mocap_ids:
                cur = self.configuration.data.site(
                    self.site_ids[site_name]).xpos
                tgt = self.configuration.data.mocap_pos[
                    self.target_mocap_ids[finger]]
                errors[finger] = np.linalg.norm(cur - tgt) * 1000  # mm
        return errors

    # ----------------------------------------------------------
    # 可视化辅助
    # ----------------------------------------------------------
    def _draw_frame(self, viewer, pos, rot, scale=0.015):
        """绘制坐标系（RGB 三轴）"""
        if not hasattr(viewer, 'user_scn'):
            return
        scn = viewer.user_scn
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        for ax in range(3):
            if scn.ngeom >= scn.maxgeom:
                break
            end = pos + scale * rot[:, ax]
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                scale * 0.05,
                np.array(pos, dtype=np.float32),
                np.array(end, dtype=np.float32),
            )
            scn.geoms[scn.ngeom].rgba[:] = colors[ax]
            scn.ngeom += 1

    def _draw_error_line(self, viewer, from_pos, to_pos, rgba):
        """绘制从指尖到目标的误差线"""
        if not hasattr(viewer, 'user_scn'):
            return
        scn = viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            0.001,
            np.array(from_pos, dtype=np.float32),
            np.array(to_pos, dtype=np.float32),
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1

    # ----------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------
    def run(self):
        print(f"\n{'='*60}")
        print(f"  Hand IK 交互测试 — {self.hand_type.upper()} Hand")
        print(f"{'='*60}")
        print()
        print("  操作:")
        print("    1. 双击彩色球体选中它")
        print("    2. Ctrl + 鼠标右键拖拽  → 在相机平面内移动")
        print("    3. Ctrl + 鼠标右键 + Shift → 前后移动")
        print("    4. 手指会实时跟踪目标球")
        print()
        print("  颜色: 红=拇指 绿=食指 蓝=中指 黄=无名指 紫=小指")
        print()
        print("  按 Ctrl+C 或关闭窗口退出")
        print()

        viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.configuration.data,
            show_left_ui=True,
            show_right_ui=True,
        )
        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

        print("[INFO] Viewer 已启动，拖拽球体测试 IK ...\n")

        # 误差线颜色（与目标球体同色系，半透明）
        line_colors = [
            [1, 0.4, 0.4, 0.6],
            [0.4, 1, 0.4, 0.6],
            [0.4, 0.4, 1, 0.6],
            [1, 1, 0.4, 0.6],
            [1, 0.4, 1, 0.6],
        ]

        last_print_time = 0

        try:
            while viewer.is_running():
                # 多步 IK（提高跟踪速度）
                for _ in range(5):
                    self._ik_step()

                mujoco.mj_forward(self.model, self.configuration.data)

                # 绘制可视化
                if hasattr(viewer, 'user_scn'):
                    viewer.user_scn.ngeom = 0

                    # 世界坐标系
                    self._draw_frame(
                        viewer, np.zeros(3), np.eye(3), scale=0.03)

                    # 各指尖坐标系 + 误差线
                    for i, (finger, site_name) in enumerate(
                        zip(FINGER_NAMES, self.fingertip_sites)
                    ):
                        if site_name in self.site_ids:
                            sid = self.site_ids[site_name]
                            p = self.configuration.data.site(sid).xpos.copy()
                            r = self.configuration.data.site(sid).xmat \
                                .reshape(3, 3).copy()
                            self._draw_frame(viewer, p, r, scale=0.01)

                            # 误差连线
                            if finger in self.target_mocap_ids:
                                tgt = self.configuration.data.mocap_pos[
                                    self.target_mocap_ids[finger]].copy()
                                self._draw_error_line(
                                    viewer, p, tgt, line_colors[i])

                viewer.sync()

                # 每秒打印误差
                t = time.time()
                if t - last_print_time > 1.0:
                    errors = self._get_errors()
                    parts = [f"{FINGER_CN[f]}:{e:.2f}mm"
                             for f, e in errors.items()]
                    print(f"\r  [误差] {' | '.join(parts)}",
                          end='    ', flush=True)
                    last_print_time = t

                time.sleep(0.02)   # ~50 Hz 显示

        except KeyboardInterrupt:
            print("\n\n[INFO] 退出...")
        finally:
            viewer.close()
            self._cleanup()
            print("[INFO] 完成")

    # ----------------------------------------------------------
    # 清理临时文件
    # ----------------------------------------------------------
    def _cleanup(self):
        hand_prefix = 'Left' if self.hand_type == 'left' else 'Right'
        script_dir = Path(__file__).parent
        mjcf_dir = (script_dir / 'InspiredHand_RuiYan' / '0611_v1.4'
                     / 'Version_3.0'
                     / f'RuiYan_Hand_{hand_prefix}_Mimic' / 'meshes')
        temp = mjcf_dir / f'_test_ik_{self.hand_type}.xml'
        if temp.exists():
            temp.unlink()
            print(f"[INFO] 已清理临时文件: {temp.name}")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Hand IK 交互测试 - 拖拽目标球测试IK精度')
    parser.add_argument(
        '--hand', type=str, default='left',
        choices=['left', 'right'],
        help='选择左手或右手 (默认: left)')
    args = parser.parse_args()

    tester = HandIKTester(hand_type=args.hand)
    tester.run()


if __name__ == '__main__':
    main()

