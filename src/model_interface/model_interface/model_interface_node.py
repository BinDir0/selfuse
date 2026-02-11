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
    FIRST_OBS = 2   # 收集初始观测
    INFERENCE = 3   # 冻结观测，进行计算
    RUNNING = 4     # 执行动作序列
    PAUSED = 5      # 暂停
    STEP_WAIT = 6   # Debug 模式等待
    STEP_ONCE = 7   # 执行单步
    RESETTING = 8   # 系统重置

# --- 转换工具 (无锁纯数学运算) ---
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
    """旋转矩阵前二列拼接 (6D 表达)"""
    return np.concatenate([T[:3, 0], T[:3, 1]])

def matrix_from_6d_rot(trans, rot6d):
    """Gram-Schmidt 正交化恢复矩阵"""
    v1, v2 = rot6d[:3], rot6d[3:]
    e1 = v1 / np.linalg.norm(v1)
    e2 = v2 - np.dot(e1, v2) * e1
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    T = np.eye(4)
    T[:3, :3] = np.stack([e1, e2, e3], axis=1)
    T[:3, 3] = trans
    return T

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # --- 1. 唯一必要的锁与事件 ---
        self.state_lock = threading.Lock()  
        self.infer_event = threading.Event()

        # --- 2. 参数获取 ---
        self.declare_parameter('arm_frequency', 30.0)
        self.declare_parameter('hand_frequency', 30.0)
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

        # 参数本地化
        self.arm_freq = self.get_parameter('arm_frequency').value
        self.hand_freq = self.get_parameter('hand_frequency').value
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

        # --- 3. 标定与网络 ---
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)

        self.cv_bridge = CvBridge()
        self.audio_session = requests.Session()
        
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
        except Exception as e:
            self.get_logger().error(f"WebSocket 链接失败: {e}")

        # --- 4. 数据结构 (deque 是原生线程安全的) ---
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instr = ""
        
        self.arm_queue = deque(maxlen=200)
        self.hand_queue = deque(maxlen=200)
        
        # 观测缓冲区 (timestamp_ns, data)
        self.buf_rgb = deque(maxlen=self.max_buf)
        self.buf_depth = deque(maxlen=self.max_buf)
        self.buf_l_wrist = deque(maxlen=self.max_buf)
        self.buf_r_wrist = deque(maxlen=self.max_buf)
        self.buf_l_kps = deque(maxlen=self.max_buf)
        self.buf_r_kps = deque(maxlen=self.max_buf)

        # --- 5. ROS 发布订阅 ---
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

        # --- 6. 线程控制 ---
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()
        
        threading.Thread(target=self.user_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

        self.arm_timer = self.create_timer(1.0 / self.arm_freq, self.arm_command_timer_cb)
        self.hand_timer = self.create_timer(1.0 / self.hand_freq, self.hand_command_timer_cb)

        self.get_logger().info("🚀 Model Interface 节点已启动（高性能架构）")

    # --- 回调逻辑 (极简，无锁) ---
    def _is_obs_allowed(self):
        """原子读取状态，无锁拦截"""
        return self.state in [SystemState.FIRST_OBS, SystemState.RUNNING, SystemState.STEP_ONCE]

    def _check_cold_start(self):
        """检查各通道是否齐备，仅在 FIRST_OBS 时被调用"""
        bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        if all(len(b) > 0 for b in bufs):
            with self.state_lock:
                if self.state == SystemState.FIRST_OBS:
                    self.state = SystemState.INFERENCE
                    self.infer_event.set()

    def _get_ns(self, header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def rgb_cb(self, m):
        if not self._is_obs_allowed(): return
        self.buf_rgb.append((self._get_ns(m.header), self.cv_bridge.imgmsg_to_cv2(m, 'rgb8')))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start()

    def depth_cb(self, m):
        if not self._is_obs_allowed(): return
        self.buf_depth.append((self._get_ns(m.header), self.cv_bridge.imgmsg_to_cv2(m, 'passthrough')))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start()

    def l_pose_cb(self, m):
        if not self._is_obs_allowed(): return
        self.buf_l_wrist.append((self._get_ns(m.header), m.pose))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start()

    def r_pose_cb(self, m):
        if not self._is_obs_allowed(): return
        self.buf_r_wrist.append((self._get_ns(m.header), m.pose))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start()

    def l_kp_cb(self, m):
        if not self._is_obs_allowed(): return
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses[:5]])
        if len(pts) < 5: pts = np.pad(pts, ((0, 5 - len(pts)), (0, 0)))
        self.buf_l_kps.append((self._get_ns(m.header), pts.flatten()))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start()

    def r_kp_cb(self, m):
        if not self._is_obs_allowed(): return
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses[:5]])
        if len(pts) < 5: pts = np.pad(pts, ((0, 5 - len(pts)), (0, 0)))
        self.buf_r_kps.append((self._get_ns(m.header), pts.flatten()))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start()

    # --- 重采样工具 ---
    def _find_nearest(self, buf, target_ns):
        if not buf: return None
        times = [x[0] for x in buf]
        idx = bisect.bisect_left(times, target_ns)
        if idx == 0: return buf[0][1]
        if idx == len(times): return buf[-1][1]
        return buf[idx-1][1] if (target_ns - times[idx-1]) < (times[idx] - target_ns) else buf[idx][1]

    def _in_range(self, buf, t):
        return buf[0][0] <= t <= buf[-1][0]

    # --- 准备 Payload (INFERENCE 状态下执行，数据已静止，无须加锁) ---
    def prepare_inference_payload(self):
        all_bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        if any(len(b) == 0 for b in all_bufs): return None

        t_ref = min(b[-1][0] for b in all_bufs)
        dt = int(1e9 / self.data_freq)

        # 采样图像
        rgb_seq, depth_seq = [], []
        for h in range(self.i_hor):
            t = t_ref - (h * self.i_str * dt)
            if not self._in_range(self.buf_rgb, t): break
            rgb_seq.append(self._find_nearest(self.buf_rgb, t))
            depth_seq.append(self._find_nearest(self.buf_depth, t))
        
        rgb_in = np.stack(rgb_seq)[::-1]
        depth_in = np.stack(depth_seq)[::-1]
        if depth_in.ndim == 3: depth_in = np.expand_dims(depth_in, axis=-1)

        # 采样状态
        states = []
        for h in range(self.s_hor):
            t = t_ref - (h * self.s_str * dt)
            if not all(self._in_range(b, t) for b in [self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]): break

            tl = self.T_base2cam_l @ matrix_from_pose_msg(self._find_nearest(self.buf_l_wrist, t))
            tr = self.T_base2cam_r @ matrix_from_pose_msg(self._find_nearest(self.buf_r_wrist, t))
            lk = self._find_nearest(self.buf_l_kps, t)
            rk = self._find_nearest(self.buf_r_kps, t)

            # 拼接 48 维向量
            vec = np.concatenate([tl[:3, 3], tr[:3, 3], get_6d_rot(tl), get_6d_rot(tr), lk, rk])
            states.append(vec)

        states_in = np.array(states)[::-1].astype(np.float32)

        return {
            "image": rgb_in, "depth_image": depth_in, "camera_intrinsics": self.K_mat,
            "instruction": self.current_instr, "states": states_in
        }

    # --- 推理线程 (生产者) ---
    def inference_worker(self):
        while rclpy.ok():
            if self.state != SystemState.INFERENCE:
                self.infer_event.wait(timeout=0.1)
                self.infer_event.clear()
                continue

            payload = self.prepare_inference_payload()
            if payload is None: continue

            try:
                # 阻塞式网络推理 (不持锁)
                res = self.policy_client.infer(payload)
                actions = res["pred_actions"]
                steps = min(self.act_len, actions.shape[0])

                # 填充队列 (deque 线程安全)
                for i in range(steps):
                    v = actions[i]
                    # 按照逻辑，将拆解后的位姿存入队列
                    step_data = {
                        'l': matrix_from_6d_rot(v[0:3], v[6:12]),
                        'r': matrix_from_6d_rot(v[3:6], v[12:18]),
                        'lk': v[18:33], 'rk': v[33:48]
                    }
                    self.arm_queue.append(step_data)
                    self.hand_queue.append(step_data)

                # 修改状态，允许定时器工作
                with self.state_lock:
                    if self.mode == 'deploy':
                        self.state = SystemState.RUNNING
                    else:
                        self.state = SystemState.STEP_WAIT
            except Exception as e:
                self.get_logger().error(f"Inference worker failed: {e}")
                time.sleep(0.1)

    # --- 命令发送定时器 (消费者) ---
    def arm_command_timer_cb(self):
        st = self.state # 原子读
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

        if not self.arm_queue:
            if st == SystemState.RUNNING:
                with self.state_lock: self.state = SystemState.INFERENCE
                self.infer_event.set()
            return
        
        data = self.arm_queue.popleft()
        is_done = (len(self.arm_queue) == 0)

        # 发布
        try:
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), "base_link"
            pa.poses.append(pose_from_matrix(self.T_cam2base_l @ data['l']))
            pa.poses.append(pose_from_matrix(self.T_cam2base_r @ data['r']))
            self.pub_action_poses.publish(pa)
            # 暂存给手部定时器
            self._current_step_cache = data
        except: pass

        if st == SystemState.STEP_ONCE:
            with self.state_lock:
                self.state = SystemState.INFERENCE if is_done else SystemState.STEP_WAIT
            self.infer_event.set()

    def hand_command_timer_cb(self):
        if self.state not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return
        if not self.hand_queue: return
        
        data = self.hand_queue.popleft() # 独立消费 hand_queue

        def _pub_hand(topic, kps, frame):
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), frame
            for p3 in kps.reshape(-1, 3):
                p = Pose(); p.position.x, p.position.y, p.position.z = map(float, p3)
                pa.poses.append(p)
            topic.publish(pa)

        try:
            _pub_hand(self.pub_action_hand_l, data['lk'], "left_wrist_link")
            _pub_hand(self.pub_action_hand_r, data['rk'], "right_wrist_link")
        except: pass

    # --- 交互逻辑 ---
    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                print("\n" + "="*40)
                instr = input("[Input] 指令: ").strip()
                mode = input("[Input] 模式 (deploy/debug): ").strip().lower()
                if instr:
                    with self.state_lock:
                        self.current_instr, self.mode, self.state = instr, ('debug' if 'debug' in mode else 'deploy'), SystemState.READY
            time.sleep(0.1)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        if k == '1' and self.state == SystemState.READY:
            with self.state_lock:
                self.play_sound("start")
                self.pub_system_mode.publish(String(data="inference"))
                # 清空不需要锁，因为此时 is_obs_allowed 为 False
                self.buf_rgb.clear(); self.buf_depth.clear(); self.buf_l_wrist.clear(); self.buf_r_wrist.clear(); self.buf_l_kps.clear(); self.buf_r_kps.clear()
                self.state = SystemState.FIRST_OBS
        elif (k == '2' or key == keyboard.Key.space):
            with self.state_lock:
                if self.mode == 'deploy' and self.state in [SystemState.RUNNING, SystemState.INFERENCE, SystemState.PAUSED]:
                    if self.state != SystemState.PAUSED:
                        self._pre_pause = self.state; self.state = SystemState.PAUSED; self.play_sound("pause")
                    else:
                        self.state = getattr(self, '_pre_pause', SystemState.INFERENCE); self.play_sound("continue")
                        self.infer_event.set()
                elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                    self.state = SystemState.STEP_ONCE; self.play_sound("continue")
        elif k == '3' and self.state != SystemState.IDLE:
            with self.state_lock:
                self.state = SystemState.RESETTING; self.play_sound("stop_and_reset"); self.pub_system_mode.publish(String(data="reset"))
                self.arm_queue.clear(); self.hand_queue.clear()
            time.sleep(3.0)
            with self.state_lock: self.state = SystemState.IDLE

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