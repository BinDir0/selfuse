#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import threading
import subprocess
import time
from enum import Enum
from collections import deque
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ROS Messages
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
    RUNNING = 2
    PAUSED = 3
    STEP_WAIT = 4
    STEP_ONCE = 5
    RESETTING = 6

# --- 坐标转换工具 ---
def matrix_from_pose_msg(pose):
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rmat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = t
    return T

def pose_from_matrix(T):
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = T[0, 3], T[1, 3], T[2, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
    return msg

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # 1. 参数获取
        self.declare_parameter('arm_frequency', 100.0)
        self.declare_parameter('hand_frequency', 80.0)
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('assets_folder', '')

        self.arm_freq = self.get_parameter('arm_frequency').value
        self.hand_freq = self.get_parameter('hand_frequency').value
        self.calib_root = self.get_parameter('calibration_path').value
        self.cam_name = self.get_parameter('camera_name').value
        self.assets_dir = self.get_parameter('assets_folder').value

        # 2. 加载标定与内参
        self.T_cam2base_left, self.K_left = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_right, self.K_right = self.find_and_load_calibration(self.cam_name, 'right')
        
        self.T_base2cam_left = np.linalg.inv(self.T_cam2base_left)
        self.T_base2cam_right = np.linalg.inv(self.T_cam2base_right)
        
        # 既然两只手都对同一个相机标定，内参应该是一样的。我们取其中一个作为当前相机的全局内参。
        self.camera_matrix = self.K_left

        # 3. 状态与队列
        self.lock = threading.Lock()
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instruction = ""
        self.arm_queue = deque(maxlen=200)
        self.hand_queue = deque(maxlen=200)
        
        self.latest_obs = {
            'image': None,
            'depth_image': None,
            'left_wrist_pose': None,
            'right_wrist_pose': None,
            'left_keypoints': None,
            'right_keypoints': None
        }
        self.cv_bridge = CvBridge()

        # 4. 发布者
        self.pub_system_mode = self.create_publisher(String, '/system/mode', 10)
        self.pub_action_poses = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 1)
        self.pub_action_hand_l = self.create_publisher(PoseArray, '/action/left_hand/keypoints', 1)
        self.pub_action_hand_r = self.create_publisher(PoseArray, '/action/right_hand/keypoints', 1)

        # 5. 订阅者
        self.create_subscription(Image, f'/camera/{self.cam_name}/rgb', self.img_cb, 1)
        self.create_subscription(Image, f'/camera/{self.cam_name}/depth', self.depth_cb, 1)
        self.create_subscription(PoseStamped, '/state/left_arm/wrist_pose', self.left_pose_cb, 1)
        self.create_subscription(PoseStamped, '/state/right_arm/wrist_pose', self.right_pose_cb, 1)
        self.create_subscription(PoseArray, '/state/left_hand/keypoints', self.left_kp_cb, 1)
        self.create_subscription(PoseArray, '/state/right_hand/keypoints', self.right_kp_cb, 1)

        # 6. 推理客户端
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
        except Exception as e:
            self.get_logger().error(f"Server Connection Failed: {e}")

        # 7. 线程与定时器
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()

        self.input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread.start()

        self.infer_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.infer_thread.start()

        self.arm_timer = self.create_timer(1.0 / self.arm_freq, self.arm_command_timer)
        self.hand_timer = self.create_timer(1.0 / self.hand_freq, self.hand_command_timer)

    # --- 辅助方法 ---
    def play_sound(self, name):
        path = os.path.join(self.assets_dir, name)
        if os.path.exists(path):
            subprocess.Popen(['mpg123', '-q', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def find_and_load_calibration(self, cam, arm):
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        assert len(matches) == 1, f"Found {len(matches)} calibration results for {cam} camera and {arm} arm."
        path = os.path.join(self.calib_root, matches[0], 'calibration_results', 'result.npz')
        
        data = np.load(path)
        T_cam2base = data['T_cam2base']
        camera_matrix = data['camera_matrix'] # 提取 3x3 矩阵
        
        self.get_logger().info(f"Loaded calibration and intrinsics from {matches[0]}")
        return T_cam2base, camera_matrix

    # --- 交互逻辑 ---
    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                print("\n" + "="*40)
                instr = input("[Input] Enter Instruction: ").strip()
                mode_in = input("[Input] Mode (deploy/debug) [deploy]: ").strip().lower()
                if not instr: continue
                with self.lock:
                    self.current_instruction, self.mode, self.state = instr, ('debug' if mode_in == 'debug' else 'deploy'), SystemState.READY
            time.sleep(0.2)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        if k == '1':
            with self.lock:
                if self.state == SystemState.READY:
                    self.play_sound("start.mp3")
                    self.pub_system_mode.publish(String(data="inference"))
                    self.state = SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT
        elif k == '2' or key == keyboard.Key.space:
            with self.lock:
                if self.mode == 'deploy':
                    if self.state in [SystemState.RUNNING, SystemState.PAUSED]:
                        self.state = SystemState.PAUSED if self.state == SystemState.RUNNING else SystemState.RUNNING
                        self.play_sound("pause.mp3" if self.state == SystemState.PAUSED else "continue.mp3")
                elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                    self.state, _ = SystemState.STEP_ONCE, self.play_sound("continue.mp3")
        elif k == '3':
            with self.lock:
                if self.state != SystemState.IDLE:
                    self.state = SystemState.RESETTING
                    self.play_sound("stop_and_reset.mp3")
                    self.pub_system_mode.publish(String(data="reset"))
                    self.arm_queue.clear(); self.hand_queue.clear()
            time.sleep(3.0) 
            with self.lock: self.state = SystemState.IDLE

    # --- 推理生产者 ---
    def inference_worker(self):
        while rclpy.ok():
            with self.lock:
                st = self.state
                if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
                    time.sleep(0.1); continue
                obs = self.latest_obs.copy()
                instr = self.current_instruction

            if obs['image'] is None or obs['depth_image'] is None or obs['left_wrist_pose'] is None:
                time.sleep(0.01); continue

            try:
                T_bl = matrix_from_pose_msg(obs['left_wrist_pose'].pose)
                T_br = matrix_from_pose_msg(obs['right_wrist_pose'].pose)
                
                # 构建 Payload，包含相机内参
                payload = {
                    "image": obs['image'],
                    "depth_image": obs['depth_image'],
                    "camera_matrix": self.camera_matrix,
                    "instruction": instr,
                    "proprio": {
                        "left": {"wrist_pose": self.T_base2cam_left @ T_bl, "keypoints": obs['left_keypoints']},
                        "right": {"wrist_pose": self.T_base2cam_right @ T_br, "keypoints": obs['right_keypoints']}
                    }
                }
                
                result = self.policy_client.infer(payload)
                
                with self.lock:
                    self.arm_queue.append(result)
                    self.hand_queue.append(result)
                if st == SystemState.STEP_ONCE:
                    with self.lock: self.state = SystemState.STEP_WAIT
            except Exception as e:
                self.get_logger().error(f"Infer Error: {e}"); time.sleep(0.1)

    # --- 定时器回调 (消费者) ---
    def arm_command_timer(self):
        if self.state not in [SystemState.RUNNING, SystemState.STEP_ONCE] or not self.arm_queue: return
        with self.lock: data = self.arm_queue.popleft()
        try:
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = self.get_clock().now().to_msg(), "base_link"
            if 'left' in data: pa.poses.append(pose_from_matrix(self.T_cam2base_left @ data['left']['wrist_pose']))
            if 'right' in data: pa.poses.append(pose_from_matrix(self.T_cam2base_right @ data['right']['wrist_pose']))
            self.pub_action_poses.publish(pa)
        except: pass

    def hand_command_timer(self):
        if self.state not in [SystemState.RUNNING, SystemState.STEP_ONCE] or not self.hand_queue: return
        with self.lock: data = self.hand_queue.popleft()
        
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

    # --- 传感器回调 ---
    def img_cb(self, msg):
        with self.lock: self.latest_obs['image'] = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
    def depth_cb(self, msg):
        with self.lock: self.latest_obs['depth_image'] = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
    def left_pose_cb(self, msg):
        with self.lock: self.latest_obs['left_wrist_pose'] = msg
    def right_pose_cb(self, msg):
        with self.lock: self.latest_obs['right_wrist_pose'] = msg

    def left_kp_cb(self, msg):
        with self.lock:
            self.latest_obs['left_keypoints'] = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
    
    def right_kp_cb(self, msg):
        with self.lock:
            self.latest_obs['right_keypoints'] = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])

def main(args=None):
    rclpy.init(args=args)
    node = ModelInterfaceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()