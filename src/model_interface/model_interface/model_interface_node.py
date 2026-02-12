#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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

# ROS 消息
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    from .utils.websocket_client import WebsocketClientPolicy
except ImportError:
    from utils.websocket_client import WebsocketClientPolicy

class SystemState(Enum):
    IDLE = 0
    READY = 1
    FIRST_OBS = 2   # 收集初始数据 (Write)
    INFERENCE = 3   # 推理 (Read Only, Write Stopped)
    RUNNING = 4     # 执行 (Write)
    PAUSED = 5      # 暂停 (No Write)
    STEP_WAIT = 6   # Debug 等待 (No Write)
    STEP_ONCE = 7   # Debug 执行 (Write)
    RESETTING = 8   # 重置

# --- 数学工具 ---
def matrix_from_pose_msg(pose):
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T

def pose_from_matrix(T):
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = map(float, T[:3, 3])
    quat = R.from_matrix(T[:3, :3]).as_quat()
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = map(float, quat)
    return msg

def get_6d_rot(T):
    return np.concatenate([T[:3, 0], T[:3, 1]])

def matrix_from_6d_rot(trans, rot6d):
    v1, v2 = rot6d[:3], rot6d[3:]
    e1 = v1 / (np.linalg.norm(v1) + 1e-6)
    e2 = v2 - np.dot(e1, v2) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-6)
    e3 = np.cross(e1, e2)
    T = np.eye(4)
    T[:3, :3] = np.stack([e1, e2, e3], axis=1)
    T[:3, 3] = trans
    return T

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # --- 1. 锁和信号 ---
        self.action_timer_lock = threading.Lock()
        self.rgb_cb_lock = threading.Lock()
        self.depth_cb_lock = threading.Lock()
        self.l_pose_cb_lock = threading.Lock()
        self.r_pose_cb_lock = threading.Lock()
        self.l_kp_cb_lock = threading.Lock()
        self.r_kp_cb_lock = threading.Lock()
        self.infer_event = threading.Event()

        # --- 2. 参数加载 ---
        self.declare_parameter('control_frequency', 30.0)
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('audio_service_host', 'localhost')
        self.declare_parameter('audio_service_port', 8080)
        
        self.declare_parameter('data_frequency', 30.0)
        self.declare_parameter('state_horizon', 10)
        self.declare_parameter('state_stride', 1)
        self.declare_parameter('image_horizon', 1)
        self.declare_parameter('image_stride', 1)
        self.declare_parameter('action_execution_len', 6)
        self.declare_parameter('buffer_size', 300)

        # 参数提取
        self.ctrl_freq = self.get_parameter('control_frequency').value
        self.calib_root = self.get_parameter('calibration_path').value
        self.cam_name = self.get_parameter('camera_name').value
        self.audio_host = self.get_parameter('audio_service_host').value
        self.audio_port = self.get_parameter('audio_service_port').value
        self.data_freq = self.get_parameter('data_frequency').value
        self.s_hor = self.get_parameter('state_horizon').value
        self.s_str = self.get_parameter('state_stride').value
        self.i_hor = self.get_parameter('image_horizon').value
        self.i_str = self.get_parameter('image_stride').value
        self.act_len = self.get_parameter('action_execution_len').value
        self.max_buf = self.get_parameter('buffer_size').value

        # --- 3. 虚拟时间轴 ---
        self.first_ts_ns = 0
        self.total_inactive_ns = 0
        self.inactive_start_ns = None

        # --- 4. 标定 ---
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)

        # --- 5. 数据结构 ---
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instr = ""
        
        self.action_queue = deque(maxlen=200)
        
        # 传感器缓冲区：deque 在 CPython 中 append 是原子操作，无需加锁
        self.buf_rgb = deque(maxlen=self.max_buf)
        self.buf_depth = deque(maxlen=self.max_buf)
        self.buf_l_wrist = deque(maxlen=self.max_buf)
        self.buf_r_wrist = deque(maxlen=self.max_buf)
        self.buf_l_kps = deque(maxlen=self.max_buf)
        self.buf_r_kps = deque(maxlen=self.max_buf)
        
        self.cv_bridge = CvBridge()
        self.audio_session = requests.Session()

        # --- 6. 通信接口 (分配 Callback Group) ---
        self.pub_system_mode = self.create_publisher(String, '/system/mode', 10)
        self.pub_action_poses = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 1)
        self.pub_action_hand_l = self.create_publisher(PoseArray, '/action/left_hand/keypoints', 1)
        self.pub_action_hand_r = self.create_publisher(PoseArray, '/action/right_hand/keypoints', 1)

        self.create_subscription(Image, f'/camera/{self.cam_name}/rgb', self.rgb_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(Image, f'/camera/{self.cam_name}/depth', self.depth_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseStamped, '/state/left_arm/wrist_pose', self.l_pose_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseStamped, '/state/right_arm/wrist_pose', self.r_pose_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseArray, '/state/left_hand/keypoints', self.l_kp_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseArray, '/state/right_hand/keypoints', self.r_kp_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())

        # --- 7. 启动 ---
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
            self.get_logger().info("✅ WebSocket 连接成功")
        except Exception as e:
            self.get_logger().error(f"❌ WebSocket 连接失败: {e}")

        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()
        
        threading.Thread(target=self.user_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

        # 控制定时器
        self.control_timer = self.create_timer(1.0 / self.ctrl_freq, self.control_timer_cb, callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info(f"🚀 Model Interface 启动 (Multi-threaded & Lock-Optimized)")

    # --- 状态与时间逻辑 ---
    def _is_active_recording(self, state):
        """原子读取: 仅在这些状态下记录数据"""
        return state in [SystemState.FIRST_OBS, SystemState.RUNNING, SystemState.STEP_ONCE]

    def _switch_state(self, new_state):
        old_state = self.state
        if old_state == new_state: return

        now_ns = self.get_clock().now().nanoseconds
        
        # 记录 -> 停止：开始计时 Gap
        if self._is_active_recording(old_state) and not self._is_active_recording(new_state):
            self.inactive_start_ns = now_ns
        
        # 停止 -> 记录：结算 Gap
        if not self._is_active_recording(old_state) and self._is_active_recording(new_state):
            if self.inactive_start_ns is not None:
                self.total_inactive_ns += (now_ns - self.inactive_start_ns)
                self.inactive_start_ns = None

        self.state = new_state
        self.get_logger().info(f"State: {old_state.name} -> {new_state.name}")

    def _get_msg_ns(self, header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def _update_buffer(self, buf, header, data):
        """无锁写入，依赖 deque 的线程安全性和状态隔离"""
        # 1. 原子读取状态
        if not self._is_active_recording(self.state):
            return

        # 2. 计算虚拟时间
        raw_ns = self._get_msg_ns(header)
        virtual_ts = raw_ns - self.first_ts_ns - self.total_inactive_ns
        
        # 3. 原子写入
        buf.append((virtual_ts, data))
        
        # 4. 冷启动检查 (简单长度判断无需加锁，只有状态切换需要)
        if self.state == SystemState.FIRST_OBS:
            self._check_first_obs_complete()

    def _check_first_obs_complete(self):
        # 检查是否所有缓冲区都有数据
        if all(len(b) > 0 for b in [self.buf_rgb, self.buf_depth, self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]):
            if self.state == SystemState.FIRST_OBS:
                self._switch_state(SystemState.INFERENCE)
                self.infer_event.set()

    # --- 传感器回调 (完全无锁，依赖 Executor 并行) ---
    def rgb_cb(self, m): 
        with self.rgb_cb_lock:
            self._update_buffer(self.buf_rgb, m.header, self.cv_bridge.imgmsg_to_cv2(m, 'rgb8'))

    def depth_cb(self, m): 
        with self.depth_cb_lock:
            self._update_buffer(self.buf_depth, m.header, self.cv_bridge.imgmsg_to_cv2(m, 'passthrough'))

    def l_pose_cb(self, m): 
        with self.l_pose_cb_lock:
            self._update_buffer(self.buf_l_wrist, m.header, m.pose)

    def r_pose_cb(self, m): 
        with self.r_pose_cb_lock:
            self._update_buffer(self.buf_r_wrist, m.header, m.pose)

    def l_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses])
        with self.l_kp_cb_lock:
            self._update_buffer(self.buf_l_kps, m.header, pts.flatten())

    def r_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses])
        with self.r_kp_cb_lock:
            self._update_buffer(self.buf_r_kps, m.header, pts.flatten())

    # --- 采样与推理 ---
    def _find_nearest(self, sorted_buf, target_ns):
        # 注意：这里输入的是 list 副本，不是 deque
        if not sorted_buf: return None
        times = [x[0] for x in sorted_buf]
        idx = bisect.bisect_left(times, target_ns)
        
        if idx == 0: return sorted_buf[0][1]
        if idx == len(times): return sorted_buf[-1][1]
        
        if (target_ns - times[idx-1]) < (times[idx] - target_ns):
            return sorted_buf[idx-1][1]
        else:
            return sorted_buf[idx][1]

    def _is_in_range(self, sorted_buf, target_ts):
        if not sorted_buf: return False
        dt_ns = int(1e9 / self.data_freq)
        return (sorted_buf[0][0] - dt_ns) <= target_ts <= (sorted_buf[-1][0] + dt_ns)

    def prepare_inference_payload(self):
        # 1. 创建快照 & 排序
        with 
        all_snaps = [
            sorted(list(self.buf_rgb), key=lambda x: x[0]),
            sorted(list(self.buf_depth), key=lambda x: x[0]),
            sorted(list(self.buf_l_wrist), key=lambda x: x[0]),
            sorted(list(self.buf_r_wrist), key=lambda x: x[0]),
            sorted(list(self.buf_l_kps), key=lambda x: x[0]),
            sorted(list(self.buf_r_kps), key=lambda x: x[0])
        ]
        
        assert all(len(s) > 0 for s in all_snaps), "缓冲区数据不全"

        snap_rgb, snap_depth, snap_lw, snap_rw, snap_lk, snap_rk = all_snaps

        # 2. 对齐采样
        t_ref = min(s[-1][0] for s in all_snaps)
        grid_ns = int(1e9 / self.data_freq)

        # Image
        rgb_seq, depth_seq = [], []
        for h in range(self.i_hor):
            t = t_ref - (h * self.i_str * grid_ns)
            if not self._is_in_range(snap_rgb, t): break
            rgb_seq.append(self._find_nearest(snap_rgb, t))
            depth_seq.append(self._find_nearest(snap_depth, t))
        
        rgb_in = np.stack(rgb_seq)[::-1]
        depth_in = np.stack(depth_seq)[::-1]
        if depth_in.ndim == 3: depth_in = np.expand_dims(depth_in, axis=-1)

        # State
        states_list = []
        for h in range(self.s_hor):
            t = t_ref - (h * self.s_str * grid_ns)
            state_snaps = [snap_lw, snap_rw, snap_lk, snap_rk]
            if not all(self._is_in_range(b, t) for b in state_snaps): break

            tl = self.T_base2cam_l @ matrix_from_pose_msg(self._find_nearest(snap_lw, t))
            tr = self.T_base2cam_r @ matrix_from_pose_msg(self._find_nearest(snap_rw, t))
            lk = self._find_nearest(snap_lk, t)
            rk = self._find_nearest(snap_rk, t)

            vec = np.concatenate([
                tl[:3, 3], tr[:3, 3],
                get_6d_rot(tl), get_6d_rot(tr),
                lk, rk
            ])
            states_list.append(vec)

        states_in = np.array(states_list)[::-1].astype(np.float32)

        instr = self.current_instr

        return {
            "image": rgb_in, "depth_image": depth_in, "camera_intrinsics": self.K_mat,
            "instruction": instr, "states": states_in
        }

    # --- 推理 Worker ---
    def inference_worker(self):
        while rclpy.ok():
            if self.state != SystemState.INFERENCE:
                self.infer_event.wait(timeout=0.1)
                self.infer_event.clear()
                continue

            try:
                payload = self.prepare_inference_payload()
            except AssertionError:
                time.sleep(0.02); continue

            try:
                res = self.policy_client.infer(payload)
                pred = res["pred_actions"]
                steps = min(self.act_len, pred.shape[0])

                with self.queue_lock:
                    for i in range(steps):
                        v = pred[i]
                        self.action_queue.append({
                            'l': matrix_from_6d_rot(v[0:3], v[6:12]),
                            'r': matrix_from_6d_rot(v[3:6], v[12:18]),
                            'lk': v[18:33], 'rk': v[33:48]
                        })

                # 推理结束，切回执行/等待状态
                self._switch_state(SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT)
            except Exception as e:
                self.get_logger().error(f"Infer Error: {e}")
                time.sleep(0.1)

    # --- 控制定时器 ---
    def control_timer_cb(self):
        st = self.state
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

        if not self.action_queue:
            # 队列空，触发推理
            self._switch_state(SystemState.INFERENCE)
            self.infer_event.set()
            return
        
        with self.queue_lock:
            data = self.action_queue.popleft()
        
        now = self.get_clock().now().to_msg()
        try:
            # 1. Arm
            pa = PoseArray(); pa.header.stamp, pa.header.frame_id = now, "base_link"
            pa.poses.append(pose_from_matrix(self.T_cam2base_l @ data['l']))
            pa.poses.append(pose_from_matrix(self.T_cam2base_r @ data['r']))
            self.pub_action_poses.publish(pa)

            # 2. Hand
            def _pub(topic, kps, frame):
                p_arr = PoseArray(); p_arr.header.stamp, p_arr.header.frame_id = now, frame
                for pt in kps.reshape(-1, 3):
                    p = Pose(); p.position.x, p.position.y, p.position.z = map(float, pt)
                    p_arr.poses.append(p)
                topic.publish(p_arr)
            
            _pub(self.pub_action_hand_l, data['lk'], "left_wrist_link")
            _pub(self.pub_action_hand_r, data['rk'], "right_wrist_link")
        except: pass

        if st == SystemState.STEP_ONCE:
            self._switch_state(SystemState.STEP_WAIT)

    # --- 交互 ---
    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                print("\n" + "="*30)
                instr = input("[Input] 指令: ").strip()
                mode = input("[Input] 模式: ").strip().lower()
                if instr:
                    with self.state_lock:
                        self.current_instr, self.mode, self.state = instr, ('debug' if 'debug' in mode else 'deploy'), SystemState.READY
                        self.first_ts_ns, self.total_inactive_ns, self.inactive_start_ns = 0, 0, None
            time.sleep(0.1)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        if k == '1' and self.state == SystemState.READY:
            self.play_sound("start"); self.pub_system_mode.publish(String(data="inference"))
            self.first_ts_ns = self.get_clock().now().nanoseconds
            self.total_inactive_ns = 0; self.inactive_start_ns = None
            # 清空无需锁，因为此时状态不是 Recording
            self.buf_rgb.clear(); self.buf_depth.clear(); self.buf_l_wrist.clear()
            self.buf_r_wrist.clear(); self.buf_l_kps.clear(); self.buf_r_kps.clear()
            self.action_queue.clear()
            self._switch_state(SystemState.FIRST_OBS)
        elif (k == '2' or key == keyboard.Key.space):
            if self.mode == 'deploy' and self.state in [SystemState.RUNNING, SystemState.PAUSED]:
                if self.state != SystemState.PAUSED:
                    self._switch_state(SystemState.PAUSED); self.play_sound("pause")
                else:
                    self._switch_state(SystemState.RUNNING); self.play_sound("continue")
            elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                self._switch_state(SystemState.STEP_ONCE); self.play_sound("continue")
        elif k == '3' and self.state != SystemState.IDLE:
            self._switch_state(SystemState.RESETTING); self.play_sound("stop_and_reset"); self.pub_system_mode.publish(String(data="reset"))
            time.sleep(3.0); self._switch_state(SystemState.IDLE)

    def find_and_load_calibration(self, cam, arm):
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        d = np.load(os.path.join(self.calib_root, sorted(matches)[-1], 'calibration_results', 'result.npz'))
        return d['T_cam2base'], d['camera_matrix']

    def play_sound(self, name):
        url = f"http://{self.audio_host}:{self.audio_port}/play/{name}"
        threading.Thread(target=lambda: self.audio_session.post(url, timeout=0.5), daemon=True).start()

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = ModelInterfaceNode()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()