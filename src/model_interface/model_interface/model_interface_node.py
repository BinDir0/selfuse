import rclpy
from rclpy.node import Node
import numpy as np
import os
import threading
import subprocess
import time
from enum import Enum
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ROS Messages
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge

# Utils
from .utils.websocket_client import WebsocketClientPolicy

# --- 状态定义 ---
class SystemState(Enum):
    IDLE = 0        # 等待用户输入指令
    READY = 1       # 指令已就绪，等待脚踏板1启动
    RUNNING = 2     # (Deploy) 正在连续推理
    PAUSED = 3      # (Deploy) 暂停中
    STEP_WAIT = 4   # (Debug) 等待下一步触发
    STEP_ONCE = 5   # (Debug) 执行单步推理
    RESETTING = 6   # 归位中

def matrix_from_pose_msg(pose_msg):
    t = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    q = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
    r = R.from_quat(q)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T

def pose_from_matrix(T):
    msg = Pose()
    msg.position.x = T[0, 3]
    msg.position.y = T[1, 3]
    msg.position.z = T[2, 3]
    r = R.from_matrix(T[:3, :3])
    q = r.as_quat()
    msg.orientation.x = q[0]
    msg.orientation.y = q[1]
    msg.orientation.z = q[2]
    msg.orientation.w = q[3]
    return msg

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')
        
        # 1. Parameters
        self.declare_parameter('frequency', 100.0)
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('assets_folder', '/root/workspace/a2d-tele/assets')

        self.freq = self.get_parameter('frequency').value
        self.host = self.get_parameter('model_server_host').value
        self.port = self.get_parameter('model_server_port').value
        self.calib_root_path = self.get_parameter('calibration_path').value
        self.cam_name = self.get_parameter('camera_name').value
        self.assets_folder = self.get_parameter('assets_folder').value

        # 2. Calibration
        self.T_cam2base_left = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_right = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_left = np.linalg.inv(self.T_cam2base_left)
        self.T_base2cam_right = np.linalg.inv(self.T_cam2base_right)

        # 3. State Buffer
        self.lock = threading.Lock()
        self.state = SystemState.IDLE
        self.mode = 'deploy' 
        self.current_instruction = ""
        
        self.latest_obs = {
            'image': None,
            'instruction': "",
            'left_keypoints': None,
            'right_keypoints': None,
            'left_wrist_pose': None,
            'right_wrist_pose': None
        }
        self.cv_bridge = CvBridge()

        # 4. WebSocket Client
        try:
            self.policy_client = WebsocketClientPolicy(host=self.host, port=self.port)
            self.get_logger().info("✅ Model Server Connected.")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to connect to model server: {e}")

        # 5. Subscribers
        self.create_subscription(Image, f'/camera/{self.cam_name}/rgb', self.img_cb, 1)
        self.create_subscription(Float32MultiArray, '/state/left_hand/keypoints', self.left_kp_cb, 1)
        self.create_subscription(Float32MultiArray, '/state/right_hand/keypoints', self.right_kp_cb, 1)
        self.create_subscription(PoseStamped, '/state/left_arm/wrist_pose', self.left_pose_cb, 1)
        self.create_subscription(PoseStamped, '/state/right_arm/wrist_pose', self.right_pose_cb, 1)

        # 6. Publishers
        self.pub_action_poses = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 1)
        self.pub_action_left_kp = self.create_publisher(Float32MultiArray, '/action/left_hand/keypoints', 1)
        self.pub_action_right_kp = self.create_publisher(Float32MultiArray, '/action/right_hand/keypoints', 1)

        # 7. Keyboard & Input Threads
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
        
        self.input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread.start()

        # 8. Main Loop
        self.timer = self.create_timer(1.0 / self.freq, self.inference_loop)
        
        self.print_instructions()

    def print_instructions(self):
        print("\n" + "="*50)
        print("🤖 VLA Model Interface Control")
        print("PEDAL 1 (Key '1'): Start Task")
        print("PEDAL 2 (Key '2' / Space): Pause/Resume (Deploy) or Step (Debug)")
        print("PEDAL 3 (Key '3'): Reset / Home")
        print("="*50 + "\n")

    def play_sound(self, filename):
        """Play sound using mpg123 in a non-blocking way"""
        path = os.path.join(self.assets_folder, filename)
        if not os.path.exists(path):
            # self.get_logger().warn(f"Audio file not found: {path}")
            return
        try:
            subprocess.Popen(['mpg123', '-q', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.get_logger().warn(f"Failed to play audio: {e}")

    def find_and_load_calibration(self, cam_name, arm_name):
        if not os.path.exists(self.calib_root_path):
            return np.eye(4)
        subdirs = [d for d in os.listdir(self.calib_root_path) if os.path.isdir(os.path.join(self.calib_root_path, d))]
        matched_dirs = [d for d in subdirs if cam_name in d and arm_name in d]
        if not matched_dirs:
            return np.eye(4)
        matched_dirs.sort(reverse=True)
        target_dir = matched_dirs[0]
        npz_path = os.path.join(self.calib_root_path, target_dir, 'calibration_results', 'result.npz')
        try:
            data = np.load(npz_path)
            if 'T_cam2base' in data:
                return data['T_cam2base']
        except Exception:
            pass
        return np.eye(4)

    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                try:
                    print("\n[Input] Enter Instruction (or 'exit'): ", end='', flush=True)
                    instr = input()
                    if instr.strip().lower() == 'exit':
                        rclpy.shutdown()
                        return
                    
                    print("[Input] Enter Mode ('deploy' [default] or 'debug'): ", end='', flush=True)
                    mode_str = input().strip().lower()
                    
                    with self.lock:
                        self.current_instruction = instr
                        self.mode = 'debug' if mode_str == 'debug' else 'deploy'
                        self.state = SystemState.READY
                    
                    print(f"✅ Configuration Set. Mode: {self.mode.upper()}. Waiting for PEDAL 1 to start...")
                except EOFError:
                    pass
            else:
                time.sleep(0.5)

    def on_key_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = None

        # --- PEDAL 1: START ---
        if key_char == '1':
            with self.lock:
                if self.state == SystemState.READY:
                    self.play_sound("start.mp3") # <--- START
                    if self.mode == 'deploy':
                        self.state = SystemState.RUNNING
                        self.get_logger().info("🚀 STARTED (Deploy Mode)")
                    else:
                        self.state = SystemState.STEP_WAIT
                        self.get_logger().info("🐞 STARTED (Debug Mode) - Waiting for Step Trigger")

        # --- PEDAL 2: PAUSE / RESUME / STEP ---
        elif key_char == '2' or key == keyboard.Key.space:
            with self.lock:
                if self.mode == 'deploy':
                    if self.state == SystemState.RUNNING:
                        self.state = SystemState.PAUSED
                        self.play_sound("pause.mp3") # <--- PAUSE
                        self.get_logger().info("⏸️ PAUSED")
                    elif self.state == SystemState.PAUSED:
                        self.state = SystemState.RUNNING
                        self.play_sound("continue.mp3") # <--- CONTINUE
                        self.get_logger().info("▶️ RESUMED")
                
                elif self.mode == 'debug':
                    if self.state == SystemState.STEP_WAIT:
                        self.state = SystemState.STEP_ONCE
                        self.play_sound("continue.mp3") # <--- STEP (Use continue sound)
                        self.get_logger().info("🦶 STEP EXECUTION")

        # --- PEDAL 3: STOP & RESET ---
        elif key_char == '3':
            with self.lock:
                # 只要不在 IDLE 状态，都可以触发复位
                if self.state != SystemState.IDLE:
                    self.state = SystemState.RESETTING
                    self.play_sound("stop_and_reset.mp3") # <--- STOP & RESET
                    self.get_logger().info("🔄 RESETTING/HOMING...")
                    
                    self.perform_reset_logic()
                    
                    self.state = SystemState.IDLE
                    self.get_logger().info("✅ System IDLE. Please enter new instruction.")

    def perform_reset_logic(self):
        # Implement homing logic here
        pass

    # --- Callbacks ---
    def img_cb(self, msg):
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self.lock: self.latest_obs['image'] = cv_img
        except Exception: pass
    
    def left_kp_cb(self, msg):
        with self.lock: self.latest_obs['left_keypoints'] = np.array(msg.data)
    def right_kp_cb(self, msg):
        with self.lock: self.latest_obs['right_keypoints'] = np.array(msg.data)
    def left_pose_cb(self, msg):
        with self.lock: self.latest_obs['left_wrist_pose'] = msg
    def right_pose_cb(self, msg):
        with self.lock: self.latest_obs['right_wrist_pose'] = msg

    # --- Main Loop ---
    def inference_loop(self):
        current_state = self.state

        should_infer = False
        if current_state == SystemState.RUNNING:
            should_infer = True
        elif current_state == SystemState.STEP_ONCE:
            should_infer = True
        
        if not should_infer:
            return

        with self.lock:
            obs_data = self.latest_obs.copy()
            instruction = self.current_instruction

        if obs_data['image'] is None or obs_data['left_wrist_pose'] is None:
            return

        # Transform Input
        try:
            T_base_wrist_left = matrix_from_pose_msg(obs_data['left_wrist_pose'].pose)
            T_cam_wrist_left = self.T_base2cam_left @ T_base_wrist_left

            T_base_wrist_right = matrix_from_pose_msg(obs_data['right_wrist_pose'].pose)
            T_cam_wrist_right = self.T_base2cam_right @ T_base_wrist_right
        except Exception:
            return

        payload = {
            "image": obs_data['image'],
            "instruction": instruction,
            "proprio": {
                "left": {
                    "wrist_pose": T_cam_wrist_left,
                    "keypoints": obs_data['left_keypoints']
                },
                "right": {
                    "wrist_pose": T_cam_wrist_right,
                    "keypoints": obs_data['right_keypoints']
                }
            }
        }

        try:
            result = self.policy_client.infer(payload)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return

        # Transform Output
        try:
            # 准备 PoseArray
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = self.get_clock().now().to_msg()
            pose_array_msg.header.frame_id = "base_link" 
            
            # --- Left Arm ---
            if 'left' in result:
                action_T_cam_left = result['left']['wrist_pose']
                action_T_base_left = self.T_cam2base_left @ action_T_cam_left
                
                pose_left = pose_from_matrix(action_T_base_left)
                pose_array_msg.poses.append(pose_left)

                if 'keypoints' in result['left']:
                    kp_msg = Float32MultiArray()
                    kp_msg.data = result['left']['keypoints'].flatten().tolist()
                    self.pub_action_left_kp.publish(kp_msg)
            else:
                pose_array_msg.poses.append(Pose())

            # --- Right Arm ---
            if 'right' in result:
                action_T_cam_right = result['right']['wrist_pose']
                action_T_base_right = self.T_cam2base_right @ action_T_cam_right
                
                pose_right = pose_from_matrix(action_T_base_right)
                pose_array_msg.poses.append(pose_right)

                if 'keypoints' in result['right']:
                    kp_msg = Float32MultiArray()
                    kp_msg.data = result['right']['keypoints'].flatten().tolist()
                    self.pub_action_right_kp.publish(kp_msg)
            else:
                pose_array_msg.poses.append(Pose())

            # Publish Poses
            self.pub_action_poses.publish(pose_array_msg)

        except Exception as e:
            self.get_logger().error(f"Post-processing error: {e}")

        # State Transition for Debug
        if current_state == SystemState.STEP_ONCE:
            with self.lock:
                self.state = SystemState.STEP_WAIT
                self.get_logger().info("🦶 Step Complete. Waiting...")

def main(args=None):
    rclpy.init(args=args)
    node = ModelInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()