#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import os
import threading
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import bisect
import cv2
from enum import Enum
from collections import deque
try:
    from pynput import keyboard
except Exception:
    keyboard = None
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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
    FIRST_OBS = 2
    INFERENCE = 3
    RUNNING = 4
    PAUSED = 5
    STEP_WAIT = 6
    STEP_ONCE = 7
    RESETTING = 8

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
        self.rgb_cb_lock = threading.Lock()
        self.breast_cb_lock = threading.Lock()
        self.depth_cb_lock = threading.Lock()
        self.l_pose_cb_lock = threading.Lock()
        self.r_pose_cb_lock = threading.Lock()
        self.l_kp_cb_lock = threading.Lock()
        self.r_kp_cb_lock = threading.Lock()
        self.action_timer_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.infer_event = threading.Event()

        # --- 2. 参数加载 ---
        self.declare_parameter('control_frequency', 30.0)
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('breast_camera_name', 'breast')
        self.declare_parameter('use_breast', True)
        self.declare_parameter('ui_service_host', 'localhost')
        self.declare_parameter('ui_service_port', 8080)
        self.declare_parameter('data_frequency', 30.0)
        self.declare_parameter('state_horizon', 10)
        self.declare_parameter('state_stride', 1)
        self.declare_parameter('image_horizon', 1)
        self.declare_parameter('image_stride', 1)
        self.declare_parameter('action_execution_len', 6)
        self.declare_parameter('buffer_size', 300)
        self.declare_parameter('debug_code', False)
        self.declare_parameter('do_resize', False)
        self.declare_parameter('require_depth', True)
        self.declare_parameter('auto_start', False)
        self.declare_parameter('auto_start_delay_sec', 1.0)
        self.declare_parameter('enable_keyboard', True)
        self.declare_parameter('use_mock_calibration', False)

        # 参数提取
        self.ctrl_freq = self.get_parameter('control_frequency').value
        self.calib_root = self.get_parameter('calibration_path').value
        self.cam_name = self.get_parameter('camera_name').value
        self.breast_cam_name = self.get_parameter('breast_camera_name').value
        self.use_breast = self.get_parameter('use_breast').value
        self.ui_host = self.get_parameter('ui_service_host').value
        self.ui_port = self.get_parameter('ui_service_port').value
        self.data_freq = self.get_parameter('data_frequency').value
        self.s_hor = self.get_parameter('state_horizon').value
        self.s_str = self.get_parameter('state_stride').value
        self.i_hor = self.get_parameter('image_horizon').value
        self.i_str = self.get_parameter('image_stride').value
        self.act_len = self.get_parameter('action_execution_len').value
        self.max_buf = self.get_parameter('buffer_size').value
        self.debug_code = self.get_parameter('debug_code').value
        self.do_resize = self.get_parameter('do_resize').value
        self.require_depth = self.get_parameter('require_depth').value
        self.auto_start = self.get_parameter('auto_start').value
        self.auto_start_delay_sec = float(self.get_parameter('auto_start_delay_sec').value)
        self.enable_keyboard = self.get_parameter('enable_keyboard').value
        self.use_mock_calibration = self.get_parameter('use_mock_calibration').value

        self.get_logger().info(f"📋 参数配置: 控制频率={self.ctrl_freq}Hz, 数据频率={self.data_freq}Hz, "
                              f"状态窗口={self.s_hor}, 图像窗口={self.i_hor}, 动作长度={self.act_len}, "
                              f"缓冲区大小={self.max_buf}, do_resize={self.do_resize}, require_depth={self.require_depth}, "
                              f"auto_start={self.auto_start}, enable_keyboard={self.enable_keyboard}, "
                              f"use_mock_calibration={self.use_mock_calibration}")

        # --- 3. 虚拟时间轴 ---
        self.first_ts_ns = 0
        self.total_inactive_ns = 0
        self.inactive_start_ns = None
        self.dt_ns = int(1e9 / self.data_freq)
        self.first_inference = True

        # --- 4. 标定 ---
        self.get_logger().info(f"📐 加载标定数据: 相机={self.cam_name}, 路径={self.calib_root}")
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)
        self.get_logger().info(f"✅ 标定数据加载完成: 内参矩阵形状={self.K_mat.shape}")

        self.T_wrist2tcp_l = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0.0345],
            [0, 0, 0, 1]
        ])
        self.T_wrist2tcp_r = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [-1, 0, 0, 0.0345],
            [0, 0, 0, 1]
        ])
        self.T_tcp2wrist_l = np.linalg.inv(self.T_wrist2tcp_l)
        self.T_tcp2wrist_r = np.linalg.inv(self.T_wrist2tcp_r)
        self.T_wrist2handbase_l = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0.01665],
            [0, 0, 0, 1]
        ])
        self.T_wrist2handbase_r = np.array([
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [-1, 0, 0, 0.01665],
            [0, 0, 0, 1]
        ])
        self.T_handbase2wrist_l = np.linalg.inv(self.T_wrist2handbase_l)
        self.T_handbase2wrist_r = np.linalg.inv(self.T_wrist2handbase_r)

        # --- 5. 数据结构 ---
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instr = ""
        
        self.action_queue = deque(maxlen=200)
        
        # 传感器缓冲区：deque 在 CPython 中 append 是原子操作，无需加锁
        self.buf_rgb = deque(maxlen=self.max_buf)
        self.buf_breast = deque(maxlen=self.max_buf)
        self.buf_depth = deque(maxlen=self.max_buf)
        self.buf_l_wrist = deque(maxlen=self.max_buf)
        self.buf_r_wrist = deque(maxlen=self.max_buf)
        self.buf_l_kps = deque(maxlen=self.max_buf)
        self.buf_r_kps = deque(maxlen=self.max_buf)
        
        self.cv_bridge = CvBridge()
        
        self.ui_session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        )
        self.ui_session.mount('http://', adapter)
        self.ui_session.mount('https://', adapter)

        # --- 6. 通信接口 (分配 Callback Group) ---
        self.pub_system_mode = self.create_publisher(String, '/system/mode', 10)
        self.pub_action_poses = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 1)
        self.pub_action_hand_l = self.create_publisher(PoseArray, '/action/left_hand/keypoints', 1)
        self.pub_action_hand_r = self.create_publisher(PoseArray, '/action/right_hand/keypoints', 1)

        self.create_subscription(Image, f'/camera/{self.cam_name}/rgb', self.rgb_cb, qos_profile_sensor_data, callback_group=MutuallyExclusiveCallbackGroup())
        if self.use_breast:
            self.create_subscription(Image, f'/camera/{self.breast_cam_name}/rgb', self.breast_cb, qos_profile_sensor_data, callback_group=MutuallyExclusiveCallbackGroup())
        if self.require_depth:
            self.create_subscription(Image, f'/camera/{self.cam_name}/depth', self.depth_cb, qos_profile_sensor_data, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseStamped, '/state/left_arm/wrist_pose', self.l_pose_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseStamped, '/state/right_arm/wrist_pose', self.r_pose_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseArray, '/state/left_hand/keypoints', self.l_kp_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseArray, '/state/right_hand/keypoints', self.r_kp_cb, 1, callback_group=MutuallyExclusiveCallbackGroup())
        depth_topic = f'/camera/{self.cam_name}/depth' if self.require_depth else 'disabled'
        breast_topic = f'/camera/{self.breast_cam_name}/rgb' if self.use_breast else 'disabled'
        self.get_logger().info(f"📡 订阅话题: RGB={f'/camera/{self.cam_name}/rgb'}, Breast={breast_topic}, Depth={depth_topic}, "
                              f"左腕={'/state/left_arm/wrist_pose'}, 右腕={'/state/right_arm/wrist_pose'}")

        # --- 7. 启动 ---
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
            self.get_logger().info("✅ WebSocket 连接成功")
        except Exception as e:
            self.get_logger().error(f"❌ WebSocket 连接失败: {e}")

        self.kbd_listener = None
        if self.enable_keyboard and keyboard is not None:
            self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
            self.kbd_listener.start()
        elif self.enable_keyboard:
            self.get_logger().warn("⌨️  键盘监听初始化失败，当前环境可能没有可用 X server")
        else:
            self.get_logger().info("⌨️  键盘监听已关闭，适用于虚拟/CI smoke test")
        
        threading.Thread(target=self.remote_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()
        self.get_logger().info("🔄 后台线程已启动: 用户输入循环、推理工作线程")

        # 控制定时器
        self.control_timer = self.create_timer(1.0 / self.ctrl_freq, self.control_timer_cb, callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info(f"🚀 Model Interface 启动 (Multi-threaded & Lock-Optimized)")

    def calc_wrist_to_cam(self, T_tcp2base, side):
        if side == 'left':
            T_wrist2tcp = self.T_wrist2tcp_l
            T_base2cam = self.T_base2cam_l
        elif side == 'right':
            T_wrist2tcp = self.T_wrist2tcp_r
            T_base2cam = self.T_base2cam_r
        T_wrist2cam = T_base2cam @ T_tcp2base @ T_wrist2tcp
        return T_wrist2cam

    def calc_keypoints_to_wrist(self, keypoints, side):
        assert len(keypoints) == 5, "关键点数量错误"
        keypoints_transformed = []
        for keypoint in keypoints:
            keypoint_homo = np.ones((4, 1))
            keypoint_homo[:3, 0] = keypoint
            if side == 'left':
                T_handbase2wrist = self.T_handbase2wrist_l
            elif side == 'right':
                T_handbase2wrist = self.T_handbase2wrist_r
            keypoint_homo = T_handbase2wrist @ keypoint_homo
            keypoints_transformed.append(keypoint_homo[:3, 0])
        return np.array(keypoints_transformed)

    def calc_tcp_to_arm_base(self, T_wrist2cam, side):
        if side == 'left':
            T_tcp2wrist = self.T_tcp2wrist_l
            T_cam2base = self.T_cam2base_l
        elif side == 'right':
            T_tcp2wrist = self.T_tcp2wrist_r
            T_cam2base = self.T_cam2base_r
        T_tcp2base = T_cam2base @ T_wrist2cam @ T_tcp2wrist
        return T_tcp2base

    def calc_keypoints_to_hand_base(self, keypoints, side):
        assert len(keypoints) == 5, "关键点数量错误"
        keypoints_transformed = []
        for keypoint in keypoints:
            keypoint_homo = np.ones((4, 1))
            keypoint_homo[:3, 0] = keypoint
            if side == 'left':
                T_wrist2handbase = self.T_wrist2handbase_l
            elif side == 'right':
                T_wrist2handbase = self.T_wrist2handbase_r
            keypoint_homo = T_wrist2handbase @ keypoint_homo
            keypoints_transformed.append(keypoint_homo[:3, 0])
        return np.array(keypoints_transformed)

    # --- 状态与时间逻辑 ---
    def _is_active_recording(self, state):
        """原子读取: 仅在这些状态下记录数据"""
        return state in [SystemState.FIRST_OBS, SystemState.RUNNING, SystemState.STEP_ONCE]

    def _execute_switch_state(self, new_state):
        old_state = self.state

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
        self.get_logger().info(f"🔄 状态转换: {old_state.name} -> {new_state.name} "
                              f"(总非活跃时间={self.total_inactive_ns/1e9:.3f}s)")

    def _switch_state(self, new_state):
        old_state = self.state
        if old_state == new_state: return
        if new_state == SystemState.INFERENCE:
            assert not self.action_queue, "推理状态时，动作队列应该没有数据"
            with self.rgb_cb_lock, self.breast_cb_lock, self.depth_cb_lock, self.l_pose_cb_lock, self.r_pose_cb_lock, self.l_kp_cb_lock, self.r_kp_cb_lock:
                self._execute_switch_state(new_state)
        elif new_state == SystemState.RESETTING:
            with self.rgb_cb_lock, self.breast_cb_lock, self.depth_cb_lock, self.l_pose_cb_lock, self.r_pose_cb_lock, self.l_kp_cb_lock, self.r_kp_cb_lock, self.action_timer_lock, self.inference_lock:
                self._execute_switch_state(new_state)
            queue_size = len(self.action_queue)
            buf_sizes = [len(self.buf_rgb), len(self.buf_breast), len(self.buf_depth), len(self.buf_l_wrist),
                        len(self.buf_r_wrist), len(self.buf_l_kps), len(self.buf_r_kps)]
            self.action_queue.clear()
            self.buf_rgb.clear(); self.buf_breast.clear(); self.buf_depth.clear(); self.buf_l_wrist.clear()
            self.buf_r_wrist.clear(); self.buf_l_kps.clear(); self.buf_r_kps.clear()
            self.first_inference = True
            self.get_logger().info(f"🧹 重置完成: 清空动作队列({queue_size}个动作), 清空缓冲区({buf_sizes})")
        elif new_state == SystemState.FIRST_OBS:
            assert not self.action_queue, "第一次观测状态时，动作队列应该没有数据"
            assert not self.buf_rgb, "第一次观测状态时，RGB缓冲区应该没有数据"
            assert not (self.use_breast and self.buf_breast), "第一次观测状态时，胸部相机缓冲区应该没有数据"
            assert not self.buf_depth, "第一次观测状态时，深度缓冲区应该没有数据"
            assert not self.buf_l_wrist, "第一次观测状态时，左腕部缓冲区应该没有数据"
            assert not self.buf_r_wrist, "第一次观测状态时，右腕部缓冲区应该没有数据"
            assert not self.buf_l_kps, "第一次观测状态时，左手关键点缓冲区应该没有数据"
            assert not self.buf_r_kps, "第一次观测状态时，右手关键点缓冲区应该没有数据"
            self._execute_switch_state(new_state)
        else:
            self._execute_switch_state(new_state)

    def _get_msg_ns(self, header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def _update_buffer(self, buf, header, data, buf_name=""):
        """无锁写入，依赖 deque 的线程安全性和状态隔离"""
        # 1. 原子读取状态
        if not self._is_active_recording(self.state):
            return

        # 2. 计算虚拟时间
        raw_ns = self._get_msg_ns(header)
        virtual_ts = raw_ns - self.first_ts_ns - self.total_inactive_ns

        if len(buf) > 0 and buf[-1][0] >= virtual_ts:
            self.get_logger().warn(f"⚠️  时间戳错误: 当前={buf[-1][0]/1e9:.3f}s, 新={virtual_ts/1e9:.3f}s")
            return
        
        # 3. 原子写入
        buf.append((virtual_ts, data))
        
        # 记录首次数据接收
        if len(buf) == 1 and buf_name:
            self.get_logger().info(f"📥 首次接收 {buf_name} 数据: 虚拟时间={virtual_ts/1e9:.3f}s")
        
        # 4. 冷启动检查 (简单长度判断无需加锁，只有状态切换需要)
        if self.state == SystemState.FIRST_OBS:
            self._check_first_obs_complete()

    def _check_first_obs_complete(self):
        # 检查是否所有缓冲区都有数据
        buf_sizes = {
            'RGB': len(self.buf_rgb),
            '左腕': len(self.buf_l_wrist),
            '右腕': len(self.buf_r_wrist),
            '左手关键点': len(self.buf_l_kps),
            '右手关键点': len(self.buf_r_kps)
        }
        required_buffers = [self.buf_rgb, self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        if self.use_breast:
            buf_sizes['Breast'] = len(self.buf_breast)
            required_buffers.append(self.buf_breast)
        if self.require_depth:
            buf_sizes['Depth'] = len(self.buf_depth)
            required_buffers.append(self.buf_depth)
        if all(len(b) > 5 for b in required_buffers):
            self.get_logger().info(f"✅ 首次观测完成: 缓冲区大小={buf_sizes}")
            threading.Thread(target=self._do_switch_to_inference, daemon=True).start()
        else:
            missing = [k for k, v in buf_sizes.items() if v == 0]
            if len(missing) <= 2:  # 只记录接近完成的情况，避免日志过多
                self.get_logger().info(f"⏳ 等待数据: 缺失={missing}, 当前={buf_sizes}")

    def _do_switch_to_inference(self):
        """在独立线程中执行状态转换，避免死锁"""
        self._switch_state(SystemState.INFERENCE)
        self.infer_event.set()

    def _start_first_observation(self, source):
        if self.state != SystemState.READY:
            self.get_logger().warn(f"⚠️  {source}: 当前状态不是 READY，跳过启动 (state={self.state.name})")
            return
        self.get_logger().info(f"▶️  {source}: 开始首次观测 (指令='{self.current_instr}')")
        self.play_sound("start"); self.pub_system_mode.publish(String(data="inference"))
        self.first_ts_ns = self.get_clock().now().nanoseconds
        self.total_inactive_ns = 0; self.inactive_start_ns = None
        self._switch_state(SystemState.FIRST_OBS)

    def _auto_start_after_delay(self):
        time.sleep(max(0.0, self.auto_start_delay_sec))
        self._start_first_observation("auto_start")

    # --- 传感器回调 (完全无锁，依赖 Executor 并行) ---
    def rgb_cb(self, m): 
        data = self.cv_bridge.imgmsg_to_cv2(m, 'rgb8')
        data = self._resize_image(data, is_depth=False)
        with self.rgb_cb_lock:
            self._update_buffer(self.buf_rgb, m.header, data, "RGB")

    def breast_cb(self, m):
        data = self.cv_bridge.imgmsg_to_cv2(m, 'rgb8')
        data = self._resize_image(data, is_depth=False)
        with self.breast_cb_lock:
            self._update_buffer(self.buf_breast, m.header, data, "Breast RGB")

    def depth_cb(self, m):
        data = self.cv_bridge.imgmsg_to_cv2(m, 'passthrough')
        data = self._resize_image(data, is_depth=True)
        with self.depth_cb_lock:
            self._update_buffer(self.buf_depth, m.header, data, "Depth")

    def l_pose_cb(self, m): 
        with self.l_pose_cb_lock:
            self._update_buffer(self.buf_l_wrist, m.header, m.pose, "左腕部姿态")

    def r_pose_cb(self, m): 
        with self.r_pose_cb_lock:
            self._update_buffer(self.buf_r_wrist, m.header, m.pose, "右腕部姿态")

    def l_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses])
        with self.l_kp_cb_lock:
            self._update_buffer(self.buf_l_kps, m.header, pts, "左手关键点")

    def r_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses])
        with self.r_kp_cb_lock:
            self._update_buffer(self.buf_r_kps, m.header, pts, "右手关键点")

    # --- 采样与推理 ---
    def _find_nearest(self, sorted_buf, target_ns):
        idx = bisect.bisect_left(sorted_buf, target_ns, key=lambda x: x[0])
        
        if idx == 0: return sorted_buf[0][1]
        if idx == len(sorted_buf): return sorted_buf[-1][1]
        
        if (target_ns - sorted_buf[idx-1][0]) < (sorted_buf[idx][0] - target_ns):
            return sorted_buf[idx-1][1]
        else:
            return sorted_buf[idx][1]

    def _is_in_range(self, sorted_buf, target_ts):
        return (sorted_buf[0][0] - self.dt_ns) <= target_ts <= (sorted_buf[-1][0] + self.dt_ns)

    def _resize_image(self, img, width=224, height=224, is_depth=False):
        if not self.do_resize:
            return img

        if is_depth:
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return resized_img

    def _zero_depth_like(self, rgb_seq):
        """OpenPI pi0.5 EgoHands does not use depth; keep payload schema stable."""
        seq_len, height, width = rgb_seq.shape[:3]
        return np.zeros((seq_len, height, width, 1), dtype=np.uint16)

    def _orthogonalize_6d_svd(self, rot_6d):
        """ 使用SVD将非正交矩阵投影回SO(3) """
        b1 = rot_6d[:3]
        b2 = rot_6d[3:6]
        b3 = np.cross(b1, b2)
        
        # 构造近似旋转矩阵
        R_approx = np.stack([b1, b2, b3], axis=1)
        
        # SVD 投影
        U, S, Vh = np.linalg.svd(R_approx)
        R_real = U @ Vh
        
        # 确保是旋转矩阵而非反射矩阵 (det=1)
        if np.linalg.det(R_real) < 0:
            U[:, -1] *= -1
            R_real = U @ Vh
            
        # 转回 6D
        return np.concatenate([R_real[:3, 0], R_real[:3, 1]])

    def _orthogonalize_6d(self, rot_6d):
        """
        对6D旋转向量进行施密特正交化，确保其符合旋转矩阵特性
        rot_6d: (6,) -> [b1_x, b1_y, b1_z, b2_x, b2_y, b2_z]
        """
        b1 = rot_6d[:3]
        b2 = rot_6d[3:6]
        
        # 归一化第一个基向量
        eps = 1e-6
        v1 = b1 / (np.linalg.norm(b1) + eps)
        # 正交化第二个基向量
        v2 = b2 - np.dot(v1, b2) * v1
        v2 = v2 / (np.linalg.norm(v2) + eps)
        
        return np.concatenate([v1, v2])

    def smooth_action_chunk(self, pred_actions):
        """
        对 (32, 48) 的 action chunk 进行平滑处理
        """
        # pred_actions 形状为 (32, 48)
        seq_len, dim = pred_actions.shape
        x_old = np.linspace(0, 1, seq_len)
        
        # 1. 上采样：从 32 步插值到 64 步，增加 SG 滤波器的采样密度
        upsample_len = 64
        x_new = np.linspace(0, 1, upsample_len)
        # 使用线性插值防止在边界出现过冲
        interp_func = interp1d(x_old, pred_actions, axis=0, kind='linear')
        upsampled_actions = interp_func(x_new)

        smoothed_actions = np.zeros_like(upsampled_actions)

        # 2. 分组平滑
        # A. 平移部分 (维度 0-6): 左臂(0-3), 右臂(3-6)
        # 窗口长度需为奇数。这里选 11 对应插值后的序列，平滑效果明显
        for i in range(0, 6):
            smoothed_actions[:, i] = savgol_filter(upsampled_actions[:, i], window_length=11, polyorder=3)

        # B. 旋转部分 (维度 6-18): 左臂(6-12), 右臂(12-18)
        for i in range(6, 18):
            smoothed_actions[:, i] = savgol_filter(upsampled_actions[:, i], window_length=9, polyorder=2)
        
        # C. 灵巧手关键点 (维度 18-48): 左手(18-33), 右手(33-48)
        # 手指动作通常比较细微，窗口选小一点(7)，保留灵活性
        for i in range(18, 48):
            smoothed_actions[:, i] = savgol_filter(upsampled_actions[:, i], window_length=7, polyorder=2)

        # 3. 旋转正交化修正 (必须在平滑后做)
        # SG 滤波会破坏 6D 向量的正交性，需要逐帧修复
        for t in range(upsample_len):
            smoothed_actions[t, 6:12] = self._orthogonalize_6d(smoothed_actions[t, 6:12])
            smoothed_actions[t, 12:18] = self._orthogonalize_6d(smoothed_actions[t, 12:18])

        # 4. 下采样还原：回到 32 步输出给后续逻辑
        final_actions = smoothed_actions[::2]
        
        return final_actions.astype(np.float32)

    def prepare_inference_payload(self):
        # 1. 创建快照 & 排序
        snap_rgb = list(self.buf_rgb)
        snap_breast = list(self.buf_breast)
        snap_depth = list(self.buf_depth)
        snap_lw = list(self.buf_l_wrist)
        snap_rw = list(self.buf_r_wrist)
        snap_lk = list(self.buf_l_kps)
        snap_rk = list(self.buf_r_kps)
        all_snaps = [snap_rgb, snap_lw, snap_rw, snap_lk, snap_rk]
        if self.use_breast:
            all_snaps.insert(1, snap_breast)
        if self.require_depth:
            all_snaps.insert(1, snap_depth)

        assert all(len(s) > 0 for s in all_snaps), "缓冲区数据不全"
        
        if self.debug_code:
            buf_sizes = [len(s) for s in all_snaps]
            self.get_logger().info(f"📊 准备推理数据: 缓冲区大小={buf_sizes}")

        # check the timestamp of all_snaps
        if self.debug_code:
            self._print_buffer_info("RGB", snap_rgb)
            if self.require_depth:
                self._print_buffer_info("Depth", snap_depth)
            self._print_buffer_info("Left Wrist", snap_lw)
            self._print_buffer_info("Right Wrist", snap_rw)
            self._print_buffer_info("Left Kps", snap_lk)
            self._print_buffer_info("Right Kps", snap_rk)

        if self.first_inference:
            rgb_seq = [snap_rgb[-1][1]]
            rgb_in = np.stack(rgb_seq)
            if self.use_breast:
                breast_in = np.stack([snap_breast[-1][1]])
            if self.require_depth:
                depth_in = np.stack([snap_depth[-1][1]])
                if depth_in.ndim == 3: depth_in = np.expand_dims(depth_in, axis=-1)
            else:
                depth_in = self._zero_depth_like(rgb_in)
            tl = self.calc_wrist_to_cam(matrix_from_pose_msg(snap_lw[-1][1]), 'left')
            tr = self.calc_wrist_to_cam(matrix_from_pose_msg(snap_rw[-1][1]), 'right')
            lk = self.calc_keypoints_to_wrist(snap_lk[-1][1], 'left')
            rk = self.calc_keypoints_to_wrist(snap_rk[-1][1], 'right')
            vec = np.concatenate([
                tl[:3, 3], tr[:3, 3],
                get_6d_rot(tl), get_6d_rot(tr),
                lk.flatten(), rk.flatten()])
            states_in = vec[None, :].astype(np.float32)
            def clear_repeated_data(buf, retained_data):
                buf.clear()
                buf.append(retained_data)
            clear_repeated_data(self.buf_rgb, snap_rgb[-1])
            if self.use_breast:
                clear_repeated_data(self.buf_breast, snap_breast[-1])
            if self.require_depth:
                clear_repeated_data(self.buf_depth, snap_depth[-1])
            clear_repeated_data(self.buf_l_wrist, snap_lw[-1])
            clear_repeated_data(self.buf_r_wrist, snap_rw[-1])
            clear_repeated_data(self.buf_l_kps, snap_lk[-1])
            clear_repeated_data(self.buf_r_kps, snap_rk[-1])
            self.first_inference = False
        else:
            # 2. 对齐采样
            t_ref = min(s[-1][0] for s in all_snaps)
            if self.debug_code:
                self.get_logger().info(f"⏱️  时间对齐: 参考时间={t_ref/1e9:.3f}s, 网格间隔={self.dt_ns/1e6:.1f}ms")

            # Image
            rgb_seq, breast_seq, depth_seq = [], [], []
            for h in range(self.i_hor):
                t = t_ref - (h * self.i_str * self.dt_ns)
                if not self._is_in_range(snap_rgb, t): break
                if self.use_breast and not self._is_in_range(snap_breast, t): break
                if self.require_depth and not self._is_in_range(snap_depth, t): break
                rgb_seq.append(self._find_nearest(snap_rgb, t))
                if self.use_breast:
                    breast_seq.append(self._find_nearest(snap_breast, t))
                if self.require_depth:
                    depth_seq.append(self._find_nearest(snap_depth, t))

            rgb_in = np.stack(rgb_seq)[::-1]
            if self.use_breast:
                breast_in = np.stack(breast_seq)[::-1]
            if self.require_depth:
                depth_in = np.stack(depth_seq)[::-1]
                if depth_in.ndim == 3: depth_in = np.expand_dims(depth_in, axis=-1)
            else:
                depth_in = self._zero_depth_like(rgb_in)

            # State
            states_list = []
            for h in range(self.s_hor):
                t = t_ref - (h * self.s_str * self.dt_ns)
                state_snaps = [snap_lw, snap_rw, snap_lk, snap_rk]
                if not all(self._is_in_range(b, t) for b in state_snaps): break

                tl = self.calc_wrist_to_cam(matrix_from_pose_msg(self._find_nearest(snap_lw, t)), 'left')
                tr = self.calc_wrist_to_cam(matrix_from_pose_msg(self._find_nearest(snap_rw, t)), 'right')
                lk = self.calc_keypoints_to_wrist(self._find_nearest(snap_lk, t), 'left')
                rk = self.calc_keypoints_to_wrist(self._find_nearest(snap_rk, t), 'right')

                vec = np.concatenate([
                    tl[:3, 3], tr[:3, 3],
                    get_6d_rot(tl), get_6d_rot(tr),
                    lk.flatten(), rk.flatten()
                ])
                states_list.append(vec)

            states_in = np.array(states_list)[::-1].astype(np.float32)

        instr = self.current_instr
        
        if self.debug_code:
            breast_shape = breast_in.shape if self.use_breast else None
            self.get_logger().info(f"📦 推理数据准备完成: RGB形状={rgb_in.shape}, Breast形状={breast_shape}, Depth形状={depth_in.shape}, 状态形状={states_in.shape}")

        payload = {
            "image": rgb_in, "depth_image": depth_in, "camera_intrinsics": self.K_mat,
            "instruction": instr, "states": states_in
        }
        if self.use_breast:
            payload["breast_image"] = breast_in
        return payload

    # --- 推理 Worker ---
    def inference_worker(self):
        while rclpy.ok():
            with self.inference_lock:
                if self.state != SystemState.INFERENCE:
                    self.infer_event.wait(timeout=0.1)
                    self.infer_event.clear()
                    continue

                if self.debug_code:
                    self.get_logger().info(f"🧠 开始推理: 指令='{self.current_instr}', 模式={self.mode}")
                start_time = time.time()
                
                try:
                    payload = self.prepare_inference_payload()
                except AssertionError as e:
                    self.get_logger().warn(f"⚠️  数据准备失败: {e}, 等待更多数据...")
                    time.sleep(0.02); continue
                self.get_logger().info(f"🧠 推理数据准备完成: 耗时={time.time() - start_time:.3f}s")

                try:
                    res = self.policy_client.infer(payload)
                    pred = res.get("pred_actions", res.get("actions"))
                    if pred is None:
                        raise KeyError(f"模型返回缺少 pred_actions/actions，keys={list(res.keys())}")
                    pred = np.asarray(pred, dtype=np.float32)
                    if pred.ndim != 2 or pred.shape[1] != 48:
                        raise ValueError(f"预测动作形状错误: expected (T, 48), got {pred.shape}")
                    # self.get_logger().info("🪄 正在对预测轨迹进行平滑处理...")
                    # pred = self.smooth_action_chunk(pred)
                    steps = min(self.act_len, pred.shape[0])
                    
                    inference_time = time.time() - start_time
                    self.get_logger().info(f"✅ 推理完成: 耗时={inference_time:.3f}s, 预测动作数={pred.shape[0]}, 执行步数={steps}")

                    for i in range(steps):
                        v = pred[i]
                        self.action_queue.append({
                            'l': self.calc_tcp_to_arm_base(matrix_from_6d_rot(v[0:3], v[6:12]), 'left'),
                            'r': self.calc_tcp_to_arm_base(matrix_from_6d_rot(v[3:6], v[12:18]), 'right'),
                            'lk': self.calc_keypoints_to_hand_base(v[18:33].reshape(-1, 3), 'left'),
                            'rk': self.calc_keypoints_to_hand_base(v[33:48].reshape(-1, 3), 'right')
                        })

                    self.get_logger().info(f"📥 动作队列更新: 当前队列长度={len(self.action_queue)}")
                    
                    # 推理结束，切回执行/等待状态
                    self._switch_state(SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT)
                except Exception as e:
                    self.get_logger().error(f"❌ 推理错误: {e}", exc_info=True)
                    time.sleep(0.1)

    # --- 控制定时器 ---
    def control_timer_cb(self):
        with self.action_timer_lock:
            st = self.state
            if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

            if not self.action_queue:
                # 队列空，触发推理
                self.get_logger().info(f"⚠️  动作队列为空，触发推理 (当前状态={st.name})")
                self._switch_state(SystemState.INFERENCE)
                self.infer_event.set()
                return
            
            data = self.action_queue.popleft()
            
            now = self.get_clock().now().to_msg()
            try:
                # 1. Arm
                pa = PoseArray(); pa.header.stamp, pa.header.frame_id = now, "base_link"
                pa.poses.append(pose_from_matrix(data['l']))
                pa.poses.append(pose_from_matrix(data['r']))
                self.pub_action_poses.publish(pa)

                # 2. Hand
                def _pub(topic, kps, frame):
                    p_arr = PoseArray(); p_arr.header.stamp, p_arr.header.frame_id = now, frame
                    for pt in kps:
                        p = Pose(); p.position.x, p.position.y, p.position.z = map(float, pt)
                        p_arr.poses.append(p)
                    topic.publish(p_arr)
                
                _pub(self.pub_action_hand_l, data['lk'], "left_wrist_link")
                _pub(self.pub_action_hand_r, data['rk'], "right_wrist_link")
            except Exception as e:
                self.get_logger().warn(f"⚠️  动作发布失败: {e}")

            if st == SystemState.STEP_ONCE:
                self.get_logger().info(f"⏸️  单步执行完成，切换到等待状态")
                self._switch_state(SystemState.STEP_WAIT)

    def remote_input_loop(self):
        """通过 HTTP GET 宿主机服务获取阻塞式的指令输入"""
        self.get_logger().info(f"🌐 远程输入循环启动: 等待连接 {self.ui_host}:{self.ui_port}")
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                try:
                    url = f"http://{self.ui_host}:{self.ui_port}/get_input"
                    self.get_logger().info(f"📡 正在请求远程输入: {url} (无超时限制，等待用户输入...)")
                    resp = self.ui_session.get(url, timeout=None).json()
                    
                    self.current_instr = resp["instruction"]
                    self.mode = resp["mode"]
                    self._switch_state(SystemState.READY)
                    
                    self.get_logger().info(f"✅ 指令已接收: '{self.current_instr}' | 模式: {self.mode}")
                    if self.auto_start:
                        threading.Thread(target=self._auto_start_after_delay, daemon=True).start()
                except requests.exceptions.ConnectionError as e:
                    self.get_logger().warn(f"⚠️  无法连接到宿主机服务: {e}")
                    time.sleep(2.0)
                except Exception as e:
                    self.get_logger().warn(f"⚠️  获取远程输入时出错: {e}")
                    time.sleep(1.0)
            else:
                time.sleep(0.5)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        if k == '1' and self.state == SystemState.READY:
            self._start_first_observation("用户按下 '1'")
        elif (k == '2' or (keyboard is not None and key == keyboard.Key.space)):
            if self.mode == 'deploy' and self.state in [SystemState.RUNNING, SystemState.PAUSED]:
                if self.state != SystemState.PAUSED:
                    self.get_logger().info(f"⏸️  用户按下 '2'/'Space': 暂停执行")
                    self._switch_state(SystemState.PAUSED); self.play_sound("pause")
                else:
                    self.get_logger().info(f"▶️  用户按下 '2'/'Space': 继续执行")
                    self._switch_state(SystemState.RUNNING); self.play_sound("continue")
            elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                self.get_logger().info(f"⏭️  用户按下 '2'/'Space': 单步执行")
                self._switch_state(SystemState.STEP_ONCE); self.play_sound("continue")
        elif k == '3' and self.state != SystemState.IDLE:
            self.get_logger().info(f"🛑 用户按下 '3': 重置系统")
            self._switch_state(SystemState.RESETTING); self.play_sound("stop_and_reset"); self.pub_system_mode.publish(String(data="reset"))
            time.sleep(3.0); self._switch_state(SystemState.IDLE)

    def find_and_load_calibration(self, cam, arm):
        if self.use_mock_calibration:
            T_cam2base = np.eye(4, dtype=np.float32)
            T_cam2base[:3, 3] = np.array([0.0, 0.18 if arm == 'left' else -0.18, 0.0], dtype=np.float32)
            camera_matrix = np.array(
                [[600.0, 0.0, 112.0], [0.0, 600.0, 112.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.get_logger().warn(f"🧪 使用 mock 标定: 相机={cam}, 手臂={arm}")
            return T_cam2base, camera_matrix
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        if not matches:
            self.get_logger().error(f"❌ 未找到标定数据: 相机={cam}, 手臂={arm}, 路径={self.calib_root}")
            raise FileNotFoundError(f"标定数据未找到: {cam}/{arm}")
        calib_path = os.path.join(self.calib_root, sorted(matches)[-1], 'calibration_results', 'result.npz')
        self.get_logger().info(f"📂 加载标定文件: {calib_path}")
        d = np.load(calib_path)
        return d['T_cam2base'], d['camera_matrix']

    def play_sound(self, name):
        url = f"http://{self.ui_host}:{self.ui_port}/play/{name}"
        threading.Thread(target=lambda: self.ui_session.post(url, timeout=0.5), daemon=True).start()

    def _print_buffer_info(self, buf_name, buf):
        self.get_logger().info(f"Buffer Name: {buf_name}")
        self.get_logger().info(f"Buffer Size: {len(buf)}")
        self.get_logger().info(f"Buffer Timestamp: {', '.join([f'{x[0] / 1e6: .2f}ms' for x in buf])}")


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
