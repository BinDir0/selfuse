#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import threading
import time
import requests
import bisect
from enum import Enum
from collections import deque
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ROS 消息类型
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import String
from cv_bridge import CvBridge

# 引入 WebSocket 客户端逻辑
try:
    from .utils.websocket_client import WebsocketClientPolicy
except ImportError:
    from utils.websocket_client import WebsocketClientPolicy

# --- 系统状态机定义 ---
class SystemState(Enum):
    IDLE = 0        # 初始待机：等待用户输入指令
    READY = 1       # 指令就绪：等待按下脚踏板 1
    FIRST_OBS = 2   # 冷启动：等待传感器各 Topic 接收第一帧
    INFERENCE = 3   # 推理中：正在请求模型服务器（此时冻结观测更新）
    RUNNING = 4     # 执行中：Deploy 模式下持续下发动作块
    PAUSED = 5      # 暂停：Deploy 模式手动暂停
    STEP_WAIT = 6   # 等待按键：Debug 模式下等待按键触发下一步
    STEP_ONCE = 7   # 单步执行：Debug 模式下发一帧动作
    RESETTING = 8   # 系统重置中

# --- 坐标转换与 6D 旋转工具 ---
def matrix_from_pose_msg(pose):
    """geometry_msgs/Pose -> 4x4 Matrix"""
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T

def pose_from_matrix(T):
    """4x4 Matrix -> geometry_msgs/Pose"""
    msg = Pose()
    msg.position.x = float(T[0, 3])
    msg.position.y = float(T[1, 3])
    msg.position.z = float(T[2, 3])
    
    quat = R.from_matrix(T[:3, :3]).as_quat()
    msg.orientation.x = float(quat[0])
    msg.orientation.y = float(quat[1])
    msg.orientation.z = float(quat[2])
    msg.orientation.w = float(quat[3])
    return msg

def get_6d_rot_from_matrix(T):
    """提取旋转矩阵前两列作为 6D 旋转表示 (6维向量)"""
    rot_mat = T[:3, :3]
    col1 = rot_mat[:, 0]
    col2 = rot_mat[:, 1]
    return np.concatenate([col1, col2])

def matrix_from_6d_rot(trans, rot6d):
    """从 6D 旋转恢复 4x4 矩阵 (使用 Gram-Schmidt 正交化)"""
    v1 = rot6d[:3]
    v2 = rot6d[3:]
    
    # 归一化第一列
    e1 = v1 / (np.linalg.norm(v1) + 1e-6)
    # 正交化并归一化第二列
    e2 = v2 - np.dot(e1, v2) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-6)
    # 计算第三列
    e3 = np.cross(e1, e2)
    
    rot_mat = np.stack([e1, e2, e3], axis=1)
    
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # --- 1. 锁与同步机制 ---
        self.state_lock = threading.Lock()  
        self.infer_event = threading.Event()

        # --- 2. 声明与获取 ROS 参数 ---
        self.declare_parameter('arm_frequency', 30.0)      
        self.declare_parameter('hand_frequency', 30.0)     
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('audio_service_host', 'localhost')
        self.declare_parameter('audio_service_port', 8080)
        
        self.declare_parameter('data_frequency', 30.0)     # 基准对齐频率
        self.declare_parameter('state_horizon', 10)        # 状态序列长度
        self.declare_parameter('state_stride', 1)          # 状态采样步长
        self.declare_parameter('image_horizon', 1)         # 图像序列长度
        self.declare_parameter('image_stride', 1)          # 图像采样步长
        self.declare_parameter('action_execution_len', 6)  # 执行步数
        self.declare_parameter('buffer_size', 300)

        # 参数本地化
        self.data_freq = self.get_parameter('data_frequency').value
        self.arm_freq = self.get_parameter('arm_frequency').value
        self.hand_freq = self.get_parameter('hand_frequency').value
        self.calib_root = self.get_parameter('calibration_path').value
        self.cam_name = self.get_parameter('camera_name').value
        self.audio_host = self.get_parameter('audio_service_host').value
        self.audio_port = self.get_parameter('audio_service_port').value
        self.s_horizon = self.get_parameter('state_horizon').value
        self.s_stride = self.get_parameter('state_stride').value
        self.i_horizon = self.get_parameter('image_horizon').value
        self.i_stride = self.get_parameter('image_stride').value
        self.act_exec_len = self.get_parameter('action_execution_len').value
        self.max_buffer = self.get_parameter('buffer_size').value

        # --- 3. 虚拟时间轴管理变量 ---
        self.first_ts_ns = 0            # Episode 启动绝对基准
        self.total_inactive_ns = 0      # 累计非记录状态时长
        self.inactive_start_ns = None   # 记录“停表”起始点

        # --- 4. 标定参数加载 ---
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)

        # --- 5. 状态机与缓冲区 ---
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instruction = ""
        
        # 动作执行队列 (deque 是线程安全的)
        self.arm_queue = deque(maxlen=200)
        self.hand_queue = deque(maxlen=200)
        
        # 传感器历史缓冲区 (virtual_timestamp_ns, data)
        self.buf_rgb = deque(maxlen=self.max_buffer)
        self.buf_depth = deque(maxlen=self.max_buffer)
        self.buf_l_wrist = deque(maxlen=self.max_buffer)
        self.buf_r_wrist = deque(maxlen=self.max_buffer)
        self.buf_l_kps = deque(maxlen=self.max_buffer)
        self.buf_r_kps = deque(maxlen=self.max_buffer)
        
        self.cv_bridge = CvBridge()
        self.audio_session = requests.Session()

        # --- 6. ROS 通信接口 ---
        self.pub_system_mode = self.create_publisher(String, '/system/mode', 10)
        self.pub_action_poses = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 1)
        self.pub_action_hand_l = self.create_publisher(PoseArray, '/action/left_hand/keypoints', 1)
        self.pub_action_hand_r = self.create_publisher(PoseArray, '/action/right_hand/keypoints', 1)

        self.create_subscription(Image, f'/camera/{self.cam_name}/rgb', self.rgb_cb, 1)
        self.create_subscription(Image, f'/camera/{self.cam_name}/depth', self.depth_cb, 1)
        self.create_subscription(PoseStamped, '/state/left_arm/wrist_pose', self.l_pose_cb, 1)
        self.create_subscription(PoseStamped, '/state/right_arm/wrist_pose', self.r_pose_cb, 1)
        self.create_subscription(PoseArray, '/state/left_hand/keypoints', self.l_kp_cb, 1)
        self.create_subscription(PoseArray, '/state/right_hand/keypoints', self.r_kp_cb, 1)

        # --- 7. 推理客户端初始化 ---
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
            self.get_logger().info("✅ WebSocket 推理服务器已连接")
        except Exception as e:
            self.get_logger().error(f"❌ 无法连接服务器: {e}")

        # --- 8. 启动线程与定时器 ---
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()
        
        threading.Thread(target=self.user_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

        self.arm_cmd_timer = self.create_timer(1.0 / self.arm_freq, self.arm_command_timer_cb)
        self.hand_cmd_timer = self.create_timer(1.0 / self.hand_freq, self.hand_command_timer_cb)

        self.get_logger().info("🚀 Model Interface 节点已就绪（虚拟时轴对齐架构）")

    # --- 核心：状态与时间管理逻辑 ---
    def _is_recording_state(self, state):
        """
        判断当前状态是否需要将传感器数据写入缓冲区。
        按要求：FIRST_OBS, RUNNING, STEP_ONCE 需要记录。
        INFERENCE, STEP_WAIT, PAUSED, IDLE 等属于非活跃状态，需要累计 Gap。
        """
        return state in [SystemState.FIRST_OBS, SystemState.RUNNING, SystemState.STEP_ONCE]

    def _switch_state(self, new_state):
        """统一管理状态切换：计算并补偿非记录状态产生的耗时"""
        with self.state_lock:
            old_state = self.state
            if old_state == new_state:
                return

            now_ns = self.get_clock().now().nanoseconds
            
            # 记录 -> 停止记录：开始计时非活跃时长（开启停表）
            if self._is_recording_state(old_state) and not self._is_recording_state(new_state):
                self.inactive_start_ns = now_ns
            
            # 停止记录 -> 恢复记录：结算非活跃时长并累加（按下停表）
            if not self._is_recording_state(old_state) and self._is_recording_state(new_state):
                if self.inactive_start_ns is not None:
                    self.total_inactive_ns += (now_ns - self.inactive_start_ns)
                    self.inactive_start_ns = None

            self.state = new_state
            self.get_logger().info(f"状态切换: {old_state.name} -> {new_state.name}")

    def _get_msg_ns(self, header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def _update_buffer(self, buf, header, data):
        """高性能无锁写入缓冲区"""
        # 原子读取当前状态，如果不符合记录条件则直接抛弃
        if not self._is_recording_state(self.state):
            return

        raw_ns = self._get_msg_ns(header)
        
        # 核心逻辑：虚拟连续时间戳 = 原始时间 - 任务启动基准 - 累积不活跃时长
        virtual_ts = raw_ns - self.first_ts_ns - self.total_inactive_ns
        
        buf.append((virtual_ts, data))
        
        # 如果是冷启动，检查数据是否到齐
        if self.state == SystemState.FIRST_OBS:
            self._check_first_obs_readiness()

    def _check_first_obs_readiness(self):
        """检查冷启动时各 Topic 是否都收到了至少一帧"""
        bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, 
                self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        if all(len(b) > 0 for b in bufs):
            self._switch_state(SystemState.INFERENCE)
            self.infer_event.set()

    # --- 传感器回调 ---
    def rgb_cb(self, m):
        self._update_buffer(self.buf_rgb, m.header, self.cv_bridge.imgmsg_to_cv2(m, 'rgb8'))

    def depth_cb(self, m):
        self._update_buffer(self.buf_depth, m.header, self.cv_bridge.imgmsg_to_cv2(m, 'passthrough'))

    def l_pose_cb(self, m):
        self._update_buffer(self.buf_l_wrist, m.header, m.pose)

    def r_pose_cb(self, m):
        self._update_buffer(self.buf_r_wrist, m.header, m.pose)

    def l_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses[:5]])
        if len(pts) < 5:
            pts = np.pad(pts, ((0, 5 - len(pts)), (0, 0)))
        self._update_buffer(self.buf_l_kps, m.header, pts.flatten())

    def r_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses[:5]])
        if len(pts) < 5:
            pts = np.pad(pts, ((0, 5 - len(pts)), (0, 0)))
        self._update_buffer(self.buf_r_kps, m.header, pts.flatten())

    # --- 重采样与对齐工具 ---
    def _find_nearest(self, buf, target_ns):
        if not buf: return None
        times = [x[0] for x in buf]
        idx = bisect.bisect_left(times, target_ns)
        if idx == 0: return buf[0][1]
        if idx == len(times): return buf[-1][1]
        return buf[idx-1][1] if (target_ns - times[idx-1]) < (times[idx] - target_ns) else buf[idx][1]

    def _is_in_range(self, buf, target_ts):
        """允许在缓冲区边缘向外扩展一个 dt 的范围内采样最近邻"""
        if not buf: return False
        dt_ns = int(1e9 / self.data_freq)
        return (buf[0][0] - dt_ns) <= target_ts <= (buf[-1][0] + dt_ns)

    # --- 推理逻辑：准备 Payload ---
    def prepare_inference_payload(self):
        all_bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, 
                    self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        
        # 强制性断言：进入推理状态前数据必须是齐的
        assert all(len(b) > 0 for b in all_bufs), "采样缓冲区数据不全，无法执行推理！"

        # 确定共同采样基准：最新帧中的最早虚拟时间戳
        t_ref = min(b[-1][0] for b in all_bufs)
        grid_ns = int(1e9 / self.data_freq)

        # 1. Image 历史采样
        rgb_seq, depth_seq = [], []
        for h in range(self.i_horizon):
            target_t = t_ref - (h * self.i_stride * grid_ns)
            # 有多少取多少
            if not self._is_in_range(self.buf_rgb, target_t): break
            rgb_seq.append(self._find_nearest(self.buf_rgb, target_t))
            depth_seq.append(self._find_nearest(self.buf_depth, target_t))
        
        # 翻转顺序：早的在前
        rgb_in = np.stack(rgb_seq)[::-1]
        depth_in = np.stack(depth_seq)[::-1]
        if depth_in.ndim == 3: depth_in = np.expand_dims(depth_in, axis=-1)

        # 2. State 历史采样 (48维)
        states_list = []
        for h in range(self.s_horizon):
            target_t = t_ref - (h * self.s_stride * grid_ns)
            state_bufs = [self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
            if not all(self._is_in_range(b, target_t) for b in state_bufs): break

            # 采样数据并转换 Base -> Camera 系
            tl_m = self.T_base2cam_l @ matrix_from_pose_msg(self._find_nearest(self.buf_l_wrist, target_t))
            tr_m = self.T_base2cam_r @ matrix_from_pose_msg(self._find_nearest(self.buf_r_wrist, target_t))
            
            lk = self._find_nearest(self.buf_l_kps, target_t)
            rk = self._find_nearest(self.buf_r_kps, target_t)

            # 拼装向量: Trans(3+3), Rot6D(6+6), KPs(15+15) = 48
            vec = np.concatenate([
                tl_m[:3, 3], tr_m[:3, 3],
                get_6d_rot_from_matrix(tl_m), get_6d_rot_from_matrix(tr_m),
                lk, rk
            ])
            states_list.append(vec)

        states_in = np.array(states_list)[::-1].astype(np.float32)

        with self.state_lock: instr = self.current_instruction

        return {
            "image": rgb_in,
            "depth_image": depth_in,
            "camera_intrinsics": self.K_mat,
            "instruction": instr,
            "states": states_in
        }

    # --- 推理 Worker：生产者逻辑 ---
    def inference_worker(self):
        while rclpy.ok():
            # 严格控制：仅在 INFERENCE 状态下执行请求
            if self.state != SystemState.INFERENCE:
                self.infer_event.wait(timeout=0.1)
                self.infer_event.clear()
                continue

            payload = self.prepare_inference_payload()
            try:
                # 阻塞式远程请求
                response = self.policy_client.infer(payload)
                pred_actions = response["pred_actions"]
                
                # 处理执行切片
                steps = min(self.act_exec_len, pred_actions.shape[0])
                for i in range(steps):
                    v = pred_actions[i]
                    step_data = {
                        'l': matrix_from_6d_rot(v[0:3], v[6:12]),
                        'r': matrix_from_6d_rot(v[3:6], v[12:18]),
                        'lk': v[18:33], 'rk': v[33:48]
                    }
                    self.arm_queue.append(step_data)
                    self.hand_queue.append(step_data)

                # 推理结束，流转到执行或等待状态（自动补偿 Gap 时间）
                self._switch_state(SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT)

            except Exception as e:
                self.get_logger().error(f"推理故障: {e}")
                time.sleep(0.1)

    # --- 动作播放定时器：消费者逻辑 ---
    def arm_command_timer_cb(self):
        st = self.state
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
            return

        if not self.arm_queue:
            # Chunk 执行完毕，切换回 INFERENCE 以获取下一块
            if st == SystemState.RUNNING:
                self._switch_state(SystemState.INFERENCE)
                self.infer_event.set()
            return
        
        data = self.arm_queue.popleft()
        is_chunk_finished = (len(self.arm_queue) == 0)

        # 发布 臂部 PoseArray (Camera -> Base)
        try:
            pa = PoseArray()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = "base_link"
            pa.poses.append(pose_from_matrix(self.T_cam2base_l @ data['l']))
            pa.poses.append(pose_from_matrix(self.T_cam2base_r @ data['r']))
            self.pub_action_poses.publish(pa)
        except:
            pass

        # Debug 模式单步结束处理：只有 Chunk 完结才请求下一次推理
        if st == SystemState.STEP_ONCE:
            if is_chunk_finished:
                self._switch_state(SystemState.INFERENCE)
            else:
                self._switch_state(SystemState.STEP_WAIT)
            self.infer_event.set()

    def hand_command_timer_cb(self):
        if self.state not in [SystemState.RUNNING, SystemState.STEP_ONCE] or not self.hand_queue:
            return
        data = self.hand_queue.popleft()

        def _pub_pa(topic, kps, frame):
            pa = PoseArray()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = frame
            for p3 in kps.reshape(-1, 3):
                p = Pose()
                p.position.x, p.position.y, p.position.z = map(float, p3)
                pa.poses.append(p)
            topic.publish(pa)

        try:
            _pub_pa(self.pub_action_hand_l, data['lk'], "left_wrist_link")
            _pub_pa(self.pub_action_hand_r, data['rk'], "right_wrist_link")
        except:
            pass

    # --- 交互线程与工具 ---
    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                print("\n" + "="*40)
                instr = input("[Input] 指令: ").strip()
                mode = input("[Input] 模式 (deploy/debug): ").strip().lower()
                if instr:
                    with self.state_lock:
                        self.current_instruction, self.mode = instr, ('debug' if 'debug' in mode else 'deploy')
                        self.state = SystemState.READY
                        # 初始化任务变量
                        self.first_ts_ns, self.total_inactive_ns, self.inactive_start_ns = 0, 0, None
            time.sleep(0.1)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        
        if k == '1' and self.state == SystemState.READY:
            self.play_sound("start")
            self.pub_system_mode.publish(String(data="inference"))
            # 按下时刻确认为时间原点
            self.first_ts_ns = self.get_clock().now().nanoseconds
            self.total_inactive_ns, self.inactive_start_ns = 0, None
            # 清理缓冲区
            self.buf_rgb.clear(); self.buf_depth.clear(); self.buf_l_wrist.clear()
            self.buf_r_wrist.clear(); self.buf_l_kps.clear(); self.buf_r_kps.clear()
            self._switch_state(SystemState.FIRST_OBS)

        elif (k == '2' or key == keyboard.Key.space):
            # Deploy 暂停逻辑 或 Debug 步进触发
            if self.mode == 'deploy' and self.state in [SystemState.RUNNING, SystemState.INFERENCE, SystemState.PAUSED]:
                if self.state != SystemState.PAUSED:
                    self._pre_pause = self.state
                    self._switch_state(SystemState.PAUSED)
                    self.play_sound("pause")
                else:
                    self._switch_state(getattr(self, '_pre_pause', SystemState.INFERENCE))
                    self.play_sound("continue")
                    self.infer_event.set()
            elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                self._switch_state(SystemState.STEP_ONCE)
                self.play_sound("continue")

        elif k == '3' and self.state != SystemState.IDLE:
            self._switch_state(SystemState.RESETTING)
            self.play_sound("stop_and_reset")
            self.pub_system_mode.publish(String(data="reset"))
            self.arm_queue.clear(); self.hand_queue.clear()
            time.sleep(3.0)
            self._switch_state(SystemState.IDLE)

    def find_and_load_calibration(self, cam, arm):
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        path = os.path.join(self.calib_root, sorted(matches)[-1], 'calibration_results', 'result.npz')
        d = np.load(path)
        return d['T_cam2base'], d['camera_matrix']

    def play_sound(self, name):
        url = f"http://{self.audio_host}:{self.audio_port}/play/{name}"
        threading.Thread(target=lambda: self.audio_session.post(url, timeout=0.5), daemon=True).start()

def main(args=None):
    rclpy.init(args=args)
    node = ModelInterfaceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()