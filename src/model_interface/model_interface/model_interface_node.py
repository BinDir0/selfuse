#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import threading
import time
import requests
from enum import Enum
from collections import deque
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ROS Messages
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import String
from cv_bridge import CvBridge

# 引入 WebSocket 客户端逻辑
try:
    from .utils.websocket_client import WebsocketClientPolicy
except ImportError:
    from utils.websocket_client import WebsocketClientPolicy

# --- 系统状态枚举 ---
class SystemState(Enum):
    IDLE = 0        # 等待终端输入指令
    READY = 1       # 指令就绪，等待踏板1开始
    RUNNING = 2     # Deploy：连续运行
    PAUSED = 3      # Deploy：暂停
    STEP_WAIT = 4   # Debug：等待下一步
    STEP_ONCE = 5   # Debug：执行单步
    RESETTING = 6   # 归位中

# --- 坐标转换工具 ---
def matrix_from_pose_msg(pose):
    """geometry_msgs/Pose -> 4x4 matrix"""
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T

def pose_from_matrix(T):
    """4x4 matrix -> geometry_msgs/Pose"""
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = T[0, 3], T[1, 3], T[2, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
    return msg

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # 1. 声明锁 (细粒度控制)
        self.obs_lock = threading.Lock()    # 保护传感器观测数据
        self.state_lock = threading.Lock()  # 保护 FSM 状态、指令
        self.queue_lock = threading.Lock()  # 保护动作队列

        # 2. 获取参数
        self.declare_parameter('arm_frequency', 30.0)
        self.declare_parameter('hand_frequency', 30.0)
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('audio_service_host', 'localhost')
        self.declare_parameter('audio_service_port', 8080)

        self.arm_freq = self.get_parameter('arm_frequency').value
        self.hand_freq = self.get_parameter('hand_frequency').value
        self.calib_root = self.get_parameter('calibration_path').value
        self.cam_name = self.get_parameter('camera_name').value
        self.audio_host = self.get_parameter('audio_service_host').value
        self.audio_port = self.get_parameter('audio_service_port').value

        # 3. 加载标定与内参
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)

        # 4. 状态与缓冲区
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instruction = ""
        self.arm_queue = deque(maxlen=200)
        self.hand_queue = deque(maxlen=200)
        
        self.latest_obs = {
            'image': None, 'depth_image': None,
            'l_wrist_pose': None, 'r_wrist_pose': None,
            'l_kps': None, 'r_kps': None
        }
        self.cv_bridge = CvBridge()

        # 5. 发布与订阅
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

        # 6. 推理客户端
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
            self.get_logger().info("✅ WebSocket Connected.")
        except Exception as e:
            self.get_logger().error(f"❌ WebSocket Failed: {e}")

        # 7. 启动子线程与定时器
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()

        threading.Thread(target=self.user_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

        # 专业命名：Command Timers
        self.arm_cmd_timer = self.create_timer(1.0 / self.arm_freq, self.arm_command_timer_cb)
        self.hand_cmd_timer = self.create_timer(1.0 / self.hand_freq, self.hand_command_timer_cb)

        self.get_logger().info(f"🚀 Model Interface Node Started.")

    # --- 辅助：音频播放 ---
    def play_sound(self, audio_name):
        threading.Thread(target=self._send_audio_request, args=(audio_name,), daemon=True).start()

    def _send_audio_request(self, name):
        url = f"http://{self.audio_host}:{self.audio_port}/play/{name}"
        try:
            requests.post(url, timeout=0.5)
        except: pass

    # --- 辅助：标定加载 ---
    def find_and_load_calibration(self, cam, arm):
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        assert len(matches) == 1, f"Calibration match error for {cam}-{arm}"
        path = os.path.join(self.calib_root, matches[0], 'calibration_results', 'result.npz')
        data = np.load(path)
        return data['T_cam2base'], data['camera_matrix']

    # --- 交互线程：使用 state_lock ---
    def user_input_loop(self):
        while rclpy.ok():
            with self.state_lock:
                is_idle = (self.state == SystemState.IDLE)
            if is_idle:
                print("\n" + "="*40)
                instr = input("[Input] Instruction: ").strip()
                mode_in = input("[Input] Mode (deploy/debug) [deploy]: ").strip().lower()
                if not instr:
                    continue
                with self.state_lock:
                    self.current_instruction = instr
                    self.mode = 'debug' if mode_in == 'debug' else 'deploy'
                    self.state = SystemState.READY
                print(f"✅ Ready. Press '1' to Start.")
            time.sleep(0.2)

    def on_key_press(self, key):
        try:
            k = key.char
        except:
            k = None
        if k == '1':
            with self.state_lock:
                if self.state == SystemState.READY:
                    self.play_sound("start")
                    self.pub_system_mode.publish(String(data="inference"))
                    self.state = SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT
        elif k == '2' or key == keyboard.Key.space:
            with self.state_lock:
                if self.mode == 'deploy':
                    if self.state in [SystemState.RUNNING, SystemState.PAUSED]:
                        is_pausing = (self.state == SystemState.RUNNING)
                        self.state = SystemState.PAUSED if is_pausing else SystemState.RUNNING
                        self.play_sound("pause" if is_pausing else "continue")
                elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                    self.state = SystemState.STEP_ONCE
                    self.play_sound("continue")
        elif k == '3':
            with self.state_lock:
                if self.state != SystemState.IDLE:
                    self.state = SystemState.RESETTING
                    self.play_sound("stop_and_reset")
                    self.pub_system_mode.publish(String(data="reset"))
            with self.queue_lock:
                self.arm_queue.clear()
                self.hand_queue.clear()
            time.sleep(3.0)
            with self.state_lock:
                self.state = SystemState.IDLE

    # --- 推理生产者：多锁流水线 ---
    def inference_worker(self):
        while rclpy.ok():
            # 1. 检查状态 (state_lock)
            with self.state_lock:
                st = self.state
                instr = self.current_instruction
                if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
                    time.sleep(0.1); continue

            # 2. 获取快照 (obs_lock)
            with self.obs_lock:
                obs = self.latest_obs.copy()
            
            if obs['image'] is None or obs['depth_image'] is None or obs['l_wrist_pose'] is None:
                time.sleep(0.01); continue

            try:
                # 3. 准备 Payload (不占锁)
                T_cl = self.T_base2cam_l @ matrix_from_pose_msg(obs['l_wrist_pose'].pose)
                T_cr = self.T_base2cam_r @ matrix_from_pose_msg(obs['r_wrist_pose'].pose)
                
                payload = {
                    "image": obs['image'], "depth_image": obs['depth_image'], "camera_matrix": self.K_mat,
                    "instruction": instr,
                    "proprio": {
                        "left": {"wrist_pose": T_cl, "keypoints": obs['l_kps']},
                        "right": {"wrist_pose": T_cr, "keypoints": obs['r_kps']}
                    }
                }

                # 4. 阻塞推理 (耗时操作，不占任何锁)
                result = self.policy_client.infer(payload)

                # 5. 存入队列 (queue_lock)
                with self.queue_lock:
                    self.arm_queue.append(result)
                    self.hand_queue.append(result)
                
                # 6. 处理单步逻辑 (state_lock)
                if st == SystemState.STEP_ONCE:
                    with self.state_lock: self.state = SystemState.STEP_WAIT

            except Exception as e:
                self.get_logger().error(f"Inference Loop Error: {e}"); time.sleep(0.1)

    # --- 播放定时器：只占必要的锁 ---
    def arm_command_timer_cb(self):
        with self.state_lock:
            st = self.state
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

        with self.queue_lock:
            if not self.arm_queue: return
            data = self.arm_queue.popleft()
        
        try:
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), "base_link"
            if 'left' in data: pa.poses.append(pose_from_matrix(self.T_cam2base_l @ data['left']['wrist_pose']))
            if 'right' in data: pa.poses.append(pose_from_matrix(self.T_cam2base_r @ data['right']['wrist_pose']))
            self.pub_action_poses.publish(pa)
        except: pass

    def hand_command_timer_cb(self):
        with self.state_lock:
            st = self.state
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]: return

        with self.queue_lock:
            if not self.hand_queue: return
            data = self.hand_queue.popleft()
        
        def to_pa(kps, frame):
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), frame
            for pt in np.array(kps).reshape(-1, 3):
                p = Pose(); p.position.x, p.position.y, p.position.z = map(float, pt)
                pa.poses.append(p)
            return pa

        try:
            if 'left' in data and 'keypoints' in data['left']:
                self.pub_action_hand_l.publish(to_pa(data['left']['keypoints'], "left_wrist_link"))
            if 'right' in data and 'keypoints' in data['right']:
                self.pub_action_hand_r.publish(to_pa(data['right']['keypoints'], "right_wrist_link"))
        except: pass

    # --- 传感器回调：只占 obs_lock ---
    def rgb_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
        with self.obs_lock:
            self.latest_obs['image'] = img

    def depth_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
        with self.obs_lock:
            self.latest_obs['depth_image'] = img

    def l_pose_cb(self, msg):
        with self.obs_lock:
            self.latest_obs['l_wrist_pose'] = msg

    def r_pose_cb(self, msg):
        with self.obs_lock:
            self.latest_obs['r_wrist_pose'] = msg

    def l_kp_cb(self, msg):
        arr = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
        with self.obs_lock:
            self.latest_obs['l_kps'] = arr

    def r_kp_cb(self, msg):
        arr = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
        with self.obs_lock:
            self.latest_obs['r_kps'] = arr


def main(args=None):
    rclpy.init(args=args)
    node = ModelInterfaceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()