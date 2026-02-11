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

# --- 系统状态机枚举 ---
class SystemState(Enum):
    IDLE = 0        # 待机：等待终端输入指令
    READY = 1       # 就绪：等待踏板启动
    FIRST_OBS = 2   # 冷启动：等待传感器各 Topic 接收第一帧
    INFERENCE = 3   # 推理：正在准备数据并请求模型（此时冻结观测更新）
    RUNNING = 4     # 执行：Deploy 模式下持续下发动作块
    PAUSED = 5      # 暂停：Deploy 模式手动暂停
    STEP_WAIT = 6   # 等待单步：Debug 模式下等待按键触发
    STEP_ONCE = 7   # 单步执行：Debug 模式下下发一帧动作
    RESETTING = 8   # 归位：系统重置中

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
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = map(float, quat)
    return msg

def get_6d_rot_from_matrix(T):
    """旋转矩阵前二列拼接 (6维向量)"""
    rot_mat = T[:3, :3]
    return np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])

def matrix_from_6d_rot(trans, rot6d):
    """从 6D 旋转恢复 4x4 矩阵 (Gram-Schmidt)"""
    v1 = rot6d[:3]
    v2 = rot6d[3:]
    e1 = v1 / np.linalg.norm(v1)
    e2 = v2 - np.dot(e1, v2) * e1
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    rot_mat = np.stack([e1, e2, e3], axis=1)
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # --- 1. 细粒度锁 ---
        self.state_lock = threading.Lock()  
        self.queue_lock = threading.Lock()  
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.wrist_lock = threading.Lock()
        self.kp_lock = threading.Lock()

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

        # 参数读取
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

        # --- 3. 标定参数 ---
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)

        # --- 4. 状态机与缓冲区 ---
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instruction = ""
        
        self.arm_queue = deque(maxlen=200)
        self.hand_queue = deque(maxlen=200)
        
        # Buffer 存储 (timestamp_ns, data)
        self.buf_rgb = deque(maxlen=self.max_buffer)
        self.buf_depth = deque(maxlen=self.max_buffer)
        self.buf_l_wrist = deque(maxlen=self.max_buffer)
        self.buf_r_wrist = deque(maxlen=self.max_buffer)
        self.buf_l_kps = deque(maxlen=self.max_buffer)
        self.buf_r_kps = deque(maxlen=self.max_buffer)
        
        self.cv_bridge = CvBridge()

        # --- 5. ROS 通信 ---
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

        # --- 6. 模型连接 ---
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
            self.get_logger().info("✅ WebSocket 推理服务器已连接")
        except Exception as e:
            self.get_logger().error(f"❌ 无法连接服务器: {e}")

        # --- 7. 线程启动 ---
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()
        threading.Thread(target=self.user_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

        self.arm_cmd_timer = self.create_timer(1.0 / self.arm_freq, self.arm_command_timer_cb)
        self.hand_cmd_timer = self.create_timer(1.0 / self.hand_freq, self.hand_command_timer_cb)

        self.get_logger().info("🚀 Model Interface 节点已就绪")

    # --- 辅助方法 ---
    def _get_ts_ns(self, header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def _is_active(self):
        """
        只有在 FIRST_OBS(冷启动)、RUNNING(持续执行) 或 STEP_ONCE(单步执行) 时，
        才允许观测数据进入缓冲区。
        INFERENCE 状态下观测会被冻结，防止重复静止数据充斥时序窗口。
        """
        with self.state_lock:
            return self.state in [SystemState.FIRST_OBS, SystemState.RUNNING, SystemState.STEP_ONCE]

    def _check_cold_start_complete(self):
        """检查冷启动数据是否齐备"""
        with self.rgb_lock, self.depth_lock, self.wrist_lock, self.kp_lock:
            is_ready = all([
                len(self.buf_rgb) > 0, len(self.buf_depth) > 0,
                len(self.buf_l_wrist) > 0, len(self.buf_r_wrist) > 0,
                len(self.buf_l_kps) > 0, len(self.buf_r_kps) > 0
            ])
            if is_ready:
                with self.state_lock:
                    if self.state == SystemState.FIRST_OBS:
                        self.state = SystemState.INFERENCE
                        self.get_logger().info("冷启动观测已完成，自动进入 INFERENCE 状态")

    def _find_nearest(self, buffer, target_ns):
        if not buffer: return None
        times = [item[0] for item in buffer]
        idx = bisect.bisect_left(times, target_ns)
        if idx == 0: return buffer[0][1]
        if idx == len(times): return buffer[-1][1]
        if (target_ns - times[idx-1]) < (times[idx] - target_ns):
            return buffer[idx-1][1]
        return buffer[idx][1]

    def _is_time_in_buffer(self, buffer, target_ns):
        if not buffer: return False
        return buffer[0][0] <= target_ns <= buffer[-1][0]

    # --- 传感器回调 ---
    def rgb_cb(self, msg):
        if not self._is_active(): return
        data = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
        with self.rgb_lock: self.buf_rgb.append((self._get_ts_ns(msg.header), data))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start_complete()

    def depth_cb(self, msg):
        if not self._is_active(): return
        data = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
        with self.depth_lock: self.buf_depth.append((self._get_ts_ns(msg.header), data))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start_complete()

    def l_pose_cb(self, msg):
        if not self._is_active(): return
        with self.wrist_lock: self.buf_l_wrist.append((self._get_ts_ns(msg.header), msg.pose))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start_complete()

    def r_pose_cb(self, msg):
        if not self._is_active(): return
        with self.wrist_lock: self.buf_r_wrist.append((self._get_ts_ns(msg.header), msg.pose))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start_complete()

    def l_kp_cb(self, msg):
        if not self._is_active(): return
        pts = msg.poses[:5]
        arr = np.array([[p.position.x, p.position.y, p.position.z] for p in pts])
        if len(arr) < 5: arr = np.pad(arr, ((0, 5 - len(arr)), (0, 0)))
        with self.kp_lock: self.buf_l_kps.append((self._get_ts_ns(msg.header), arr.flatten()))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start_complete()

    def r_kp_cb(self, msg):
        if not self._is_active(): return
        pts = msg.poses[:5]
        arr = np.array([[p.position.x, p.position.y, p.position.z] for p in pts])
        if len(arr) < 5: arr = np.pad(arr, ((0, 5 - len(arr)), (0, 0)))
        with self.kp_lock: self.buf_r_kps.append((self._get_ts_ns(msg.header), arr.flatten()))
        if self.state == SystemState.FIRST_OBS: self._check_cold_start_complete()

    # --- 准备 Payload (非对称采样) ---
    def prepare_inference_payload(self):
        with self.rgb_lock, self.depth_lock, self.wrist_lock, self.kp_lock:
            all_bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, 
                       self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
            if any(len(b) == 0 for b in all_bufs): return None

            t_ref = min(b[-1][0] for b in all_bufs)
            grid_ns = int(1e9 / self.data_freq)

            # 1. 采样 Image 序列 (早在前，晚在后)
            rgb_seq, depth_seq = [], []
            for h in range(self.i_horizon):
                target_t = t_ref - (h * self.i_stride * grid_ns)
                if not self._is_time_in_buffer(self.buf_rgb, target_t): break
                rgb_seq.append(self._find_nearest(self.buf_rgb, target_t))
                depth_seq.append(self._find_nearest(self.buf_depth, target_t))
            
            rgb_input = np.stack(rgb_seq)[::-1]
            depth_input = np.stack(depth_seq)[::-1]
            if depth_input.ndim == 3: depth_input = np.expand_dims(depth_input, axis=-1)

            # 2. 采样 State 序列 (48维向量)
            states_list = []
            for h in range(self.s_horizon):
                target_t = t_ref - (h * self.s_stride * grid_ns)
                state_bufs = [self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
                if not all(self._is_time_in_buffer(b, target_t) for b in state_bufs): break

                T_cl = self.T_base2cam_l @ matrix_from_pose_msg(self._find_nearest(self.buf_l_wrist, target_t))
                T_cr = self.T_base2cam_r @ matrix_from_pose_msg(self._find_nearest(self.buf_r_wrist, target_t))
                lk = self._find_nearest(self.buf_l_kps, target_t)
                rk = self._find_nearest(self.buf_r_kps, target_t)

                # 组装 48 维向量
                vec = np.concatenate([
                    T_cl[:3, 3], T_cr[:3, 3],
                    get_6d_rot_from_matrix(T_cl), get_6d_rot_from_matrix(T_cr),
                    lk, rk
                ])
                states_list.append(vec)

            states_input = np.array(states_list)[::-1].astype(np.float32)

        with self.state_lock: instr = self.current_instruction

        return {
            "image": rgb_input,
            "depth_image": depth_input,
            "camera_intrinsics": self.K_mat,
            "instruction": instr,
            "states": states_input
        }

    # --- 推理 Worker 线程：仅在 INFERENCE 状态下工作 ---
    def inference_worker(self):
        while rclpy.ok():
            with self.state_lock:
                st = self.state
                if st != SystemState.INFERENCE:
                    continue

            payload = self.prepare_inference_payload()
            if payload is None:
                time.sleep(0.02); continue

            try:
                # 执行网络推理
                response = self.policy_client.infer(payload)
                pred_actions = response["pred_actions"]
                exec_steps = min(self.act_exec_len, pred_actions.shape[0])
                
                with self.queue_lock:
                    for i in range(exec_steps):
                        v = pred_actions[i]
                        step = {
                            'left': {
                                'wrist_pose': matrix_from_6d_rot(v[0:3], v[6:12]),
                                'keypoints': v[18:33]
                            },
                            'right': {
                                'wrist_pose': matrix_from_6d_rot(v[3:6], v[12:18]),
                                'keypoints': v[33:48]
                            }
                        }
                        self.arm_queue.append(step)
                        self.hand_queue.append(step)

                # 推理结束，决定下一步去向
                with self.state_lock:
                    if self.mode == 'deploy':
                        self.state = SystemState.RUNNING
                        self.get_logger().info("推理成功，开始连续执行动作序列")
                    else:
                        self.state = SystemState.STEP_WAIT
                        self.get_logger().info("推理成功，等待按键触发单步动作")

            except Exception as e:
                self.get_logger().error(f"推理失败: {e}")
                time.sleep(0.1)
            time.sleep(0.01)

    # --- 指令下发定时器：消费者逻辑 ---
    def arm_command_timer_cb(self):
        with self.state_lock:
            st = self.state
        
        # 只在运行或单步状态下消费队列
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

        with self.queue_lock:
            if not self.arm_queue:
                # 队列消费完毕，切回推理状态以请求新数据
                with self.state_lock:
                    if self.state == SystemState.RUNNING:
                        self.state = SystemState.INFERENCE
                return
            data = self.arm_queue.popleft()
            # 注意：Debug 模式下，消耗完一帧后自动切回等待状态
            is_last_step = (len(self.arm_queue) == 0)

        # 发布 臂部 PoseArray
        try:
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), "base_link"
            pa.poses.append(pose_from_matrix(self.T_cam2base_l @ data['left']['wrist_pose']))
            pa.poses.append(pose_from_matrix(self.T_cam2base_r @ data['right']['wrist_pose']))
            self.pub_action_poses.publish(pa)
        except: pass

        # Debug 模式单步结束处理
        if st == SystemState.STEP_ONCE:
            with self.state_lock:
                if is_last_step:
                    self.state = SystemState.INFERENCE
                else:
                    self.state = SystemState.STEP_WAIT

    def hand_command_timer_cb(self):
        with self.state_lock:
            st = self.state
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

        with self.queue_lock:
            if not self.hand_queue: return
            data = self.hand_queue.popleft()

        def _to_pa(kps, frame):
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), frame
            for pt in np.array(kps).reshape(-1, 3):
                p = Pose(); p.position.x, p.position.y, p.position.z = map(float, pt)
                pa.poses.append(p)
            return pa
            
        try:
            if 'left' in data and data['left'].get('keypoints') is not None:
                self.pub_action_hand_l.publish(_to_pa(data['left']['keypoints'], "left_wrist_link"))
            if 'right' in data and data['right'].get('keypoints') is not None:
                self.pub_action_hand_r.publish(_to_pa(data['right']['keypoints'], "right_wrist_link"))
        except: pass

    # --- 交互线程 ---
    def user_input_loop(self):
        while rclpy.ok():
            with self.state_lock: idle = (self.state == SystemState.IDLE)
            if idle:
                print("\n" + "="*40)
                instr = input("[Input] 指令: ").strip()
                mode_in = input("[Input] 模式 (deploy/debug): ").strip().lower()
                if not instr: continue
                with self.state_lock:
                    self.current_instruction, self.mode, self.state = instr, ('debug' if mode_in == 'debug' else 'deploy'), SystemState.READY
                print(f"✅ 就绪。按脚踏板 1 开始。")
            time.sleep(0.2)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        if k == '1':
            with self.state_lock:
                if self.state == SystemState.READY:
                    self.play_sound("start")
                    self.pub_system_mode.publish(String(data="inference"))
                    # 重置缓冲区，进入冷启动等待
                    with self.rgb_lock, self.depth_lock, self.wrist_lock, self.kp_lock:
                        self.buf_rgb.clear(); self.buf_depth.clear(); self.buf_l_wrist.clear(); self.buf_r_wrist.clear(); self.buf_l_kps.clear(); self.buf_r_kps.clear()
                    self.state = SystemState.FIRST_OBS

        elif k == '2' or key == keyboard.Key.space:
            with self.state_lock:
                if self.mode == 'deploy' and self.state in [SystemState.RUNNING, SystemState.PAUSED]:
                    is_pausing = (self.state != SystemState.PAUSED)
                    if is_pausing:
                        self.state = SystemState.PAUSED
                        self.play_sound("pause")
                    else:
                        self.state = SystemState.RUNNING
                        self.play_sound("continue")
                elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                    self.state = SystemState.STEP_ONCE
                    self.play_sound("continue")

        elif k == '3':
            with self.state_lock:
                if self.state != SystemState.IDLE:
                    self.state = SystemState.RESETTING
                    self.play_sound("stop_and_reset"); self.pub_system_mode.publish(String(data="reset"))
            time.sleep(3.0)
            with self.state_lock: self.state = SystemState.IDLE

    def find_and_load_calibration(self, cam, arm):
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        assert len(matches) == 1
        path = os.path.join(self.calib_root, matches[0], 'calibration_results', 'result.npz')
        data = np.load(path)
        return data['T_cam2base'], data['camera_matrix']

    def play_sound(self, name):
        url = f"http://{self.audio_host}:{self.audio_port}/play/{name}"
        threading.Thread(target=lambda: requests.post(url, timeout=0.5) if True else None, daemon=True).start()

def main(args=None):
    rclpy.init(args=args)
    node = ModelInterfaceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()