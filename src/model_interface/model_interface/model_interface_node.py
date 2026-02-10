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
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge

# 引入自定义的 WebSocket 客户端
try:
    from .utils.websocket_client import WebsocketClientPolicy
except ImportError:
    from utils.websocket_client import WebsocketClientPolicy

# --- 系统状态枚举 ---
class SystemState(Enum):
    IDLE = 0        # 等待终端输入指令
    READY = 1       # 指令就绪，等待脚踏板1开始
    RUNNING = 2     # Deploy模式：连续运行
    PAUSED = 3      # Deploy模式：暂停
    STEP_WAIT = 4   # Debug模式：等待触发下一步
    STEP_ONCE = 5   # Debug模式：正在执行单步
    RESETTING = 6   # 归位中（由IK Node执行，此处仅做状态维持）

# --- 坐标转换工具 ---
def matrix_from_pose_msg(pose):
    """geometry_msgs/Pose -> 4x4 matrix"""
    t = [pose.position.x, pose.position.y, pose.position.z]
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rmat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rmat
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

        # 1. 获取参数
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

        # 2. 加载标定文件 (Base <-> Cam)
        self.T_cam2base_left = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_right = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_left = np.linalg.inv(self.T_cam2base_left)
        self.T_base2cam_right = np.linalg.inv(self.T_cam2base_right)

        # 3. 内部状态与动作队列
        self.lock = threading.Lock()
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instruction = ""
        
        # 动作块缓存队列
        self.arm_queue = deque(maxlen=200)
        self.hand_queue = deque(maxlen=200)
        
        self.latest_obs = {
            'image': None,
            'left_wrist_pose': None,
            'right_wrist_pose': None,
            'left_keypoints': None,
            'right_keypoints': None
        }
        self.cv_bridge = CvBridge()

        # 4. 发布者与订阅者
        # 全局模式话题：控制 IK Node 的行为
        self.pub_system_mode = self.create_publisher(String, '/system/mode', 10)
        
        # 推理输出：发给 IK Nodes
        self.pub_action_poses = self.create_publisher(PoseArray, '/action/both_arms/wrist_poses', 1)
        self.pub_action_hand_l = self.create_publisher(Float32MultiArray, '/action/left_hand/keypoints', 1)
        self.pub_action_hand_r = self.create_publisher(Float32MultiArray, '/action/right_hand/keypoints', 1)

        # 传感器订阅
        self.create_subscription(Image, f'/camera/{self.cam_name}/rgb', self.img_cb, 1)
        self.create_subscription(PoseStamped, '/state/left_arm/wrist_pose', self.left_pose_cb, 1)
        self.create_subscription(PoseStamped, '/state/right_arm/wrist_pose', self.right_pose_cb, 1)
        self.create_subscription(Float32MultiArray, '/state/left_hand/keypoints', self.left_kp_cb, 1)
        self.create_subscription(Float32MultiArray, '/state/right_hand/keypoints', self.right_kp_cb, 1)

        # 5. 推理客户端初始化
        try:
            self.policy_client = WebsocketClientPolicy(
                host=self.get_parameter('model_server_host').value,
                port=self.get_parameter('model_server_port').value
            )
            self.get_logger().info("✅ WebSocket Inference Server Connected.")
        except Exception as e:
            self.get_logger().error(f"❌ Server Connection Failed: {e}")

        # 6. 线程与定时器
        # 全局键盘/脚踏板监听
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()

        # 终端输入线程
        self.input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread.start()

        # 推理生产者线程
        self.infer_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.infer_thread.start()

        # 异步动作播放定时器 (基于用户指定频率)
        self.arm_timer = self.create_timer(1.0 / self.arm_freq, self.arm_playback_loop)
        self.hand_timer = self.create_timer(1.0 / self.hand_freq, self.hand_playback_loop)

        self.get_logger().info(f"🚀 Model Interface Ready. Arm:{self.arm_freq}Hz, Hand:{self.hand_freq}Hz")

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
        return data['T_cam2base']

    # --- 交互逻辑 ---
    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                print("\n" + "="*40)
                instr = input("[Input] Instruction: ").strip()
                mode_in = input("[Input] Mode (deploy/debug) [deploy]: ").strip().lower()
                if not instr: continue
                with self.lock:
                    self.current_instruction = instr
                    self.mode = 'debug' if mode_in == 'debug' else 'deploy'
                    self.state = SystemState.READY
                print(f"✅ Loaded. Mode: {self.mode.upper()}. Press '1' to Start.")
            time.sleep(0.2)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None

        if k == '1': # 开始任务
            with self.lock:
                if self.state == SystemState.READY:
                    self.play_sound("start.mp3")
                    self.pub_system_mode.publish(String(data="inference"))
                    self.state = SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT

        elif k == '2' or key == keyboard.Key.space: # 暂停 / 单步继续
            with self.lock:
                if self.mode == 'deploy':
                    if self.state == SystemState.RUNNING:
                        self.state = SystemState.PAUSED
                        self.play_sound("pause.mp3")
                    elif self.state == SystemState.PAUSED:
                        self.state = SystemState.RUNNING
                        self.play_sound("continue.mp3")
                elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                    self.state = SystemState.STEP_ONCE
                    self.play_sound("continue.mp3")

        elif k == '3': # 系统归位
            with self.lock:
                if self.state != SystemState.IDLE:
                    self.state = SystemState.RESETTING
                    self.play_sound("stop_and_reset.mp3")
                    # 仅发布 reset，让 IK Nodes 接管归位动作
                    self.pub_system_mode.publish(String(data="reset"))
                    self.arm_queue.clear()
                    self.hand_queue.clear()
            
            # 等待足够的时间让机械臂完成物理归位
            time.sleep(3.0) 
            
            with self.lock:
                self.state = SystemState.IDLE
                self.get_logger().info("✅ Reset Done. Back to IDLE.")

    # --- 生产者：模型推理 ---
    def inference_worker(self):
        while rclpy.ok():
            with self.lock:
                st = self.state
                if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
                    time.sleep(0.1); continue
                obs = self.latest_obs.copy()
                instr = self.current_instruction

            if obs['image'] is None or obs['left_wrist_pose'] is None:
                time.sleep(0.01); continue

            try:
                # 坐标变换：观测到的 Base 系下位姿 -> 相机系下位姿（模型输入要求）
                T_bl = matrix_from_pose_msg(obs['left_wrist_pose'].pose)
                T_cl = self.T_base2cam_left @ T_bl
                T_br = matrix_from_pose_msg(obs['right_wrist_pose'].pose)
                T_cr = self.T_base2cam_right @ T_br

                payload = {
                    "image": obs['image'],
                    "instruction": instr,
                    "proprio": {
                        "left": {"wrist_pose": T_cl, "keypoints": obs['left_keypoints']},
                        "right": {"wrist_pose": T_cr, "keypoints": obs['right_keypoints']}
                    }
                }

                # 阻塞推理
                result = self.policy_client.infer(payload)

                with self.lock:
                    # 存入播放缓冲区，供两个频率不同的 Timer 消费
                    self.arm_queue.append(result)
                    self.hand_queue.append(result)
                
                if st == SystemState.STEP_ONCE:
                    with self.lock: self.state = SystemState.STEP_WAIT

            except Exception as e:
                self.get_logger().error(f"Inference Loop Error: {e}")
                time.sleep(0.1)

    # --- 消费者：臂部播放 (100Hz) ---
    def arm_playback_loop(self):
        # 仅在推理或单步时工作
        if self.state not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
            return

        if not self.arm_queue:
            return

        with self.lock:
            data = self.arm_queue.popleft()

        try:
            pa = PoseArray()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = "base_link" # 输出相对于 Base 系
            
            # 模型输出在相机系下，转回 Base 系
            if 'left' in data:
                T_cl_action = data['left']['wrist_pose']
                T_bl_action = self.T_cam2base_left @ T_cl_action
                pa.poses.append(pose_from_matrix(T_bl_action))
            if 'right' in data:
                T_cr_action = data['right']['wrist_pose']
                T_br_action = self.T_cam2base_right @ T_cr_action
                pa.poses.append(pose_from_matrix(T_br_action))
            
            self.pub_action_poses.publish(pa)
        except:
            pass

    # --- 消费者：手部播放 (80Hz) ---
    def hand_playback_loop(self):
        if self.state not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
            return

        if not self.hand_queue:
            return

        with self.lock:
            data = self.hand_queue.popleft()

        try:
            # 手部数据通常是关键点或关节动作，直接透传给 Hand IK
            if 'left' in data and 'keypoints' in data['left']:
                msg = Float32MultiArray(data=data['left']['keypoints'].flatten().tolist())
                self.pub_action_hand_l.publish(msg)
            if 'right' in data and 'keypoints' in data['right']:
                msg = Float32MultiArray(data=data['right']['keypoints'].flatten().tolist())
                self.pub_action_hand_r.publish(msg)
        except:
            pass

    # --- 传感器订阅回调 ---
    def img_cb(self, msg):
        with self.lock: self.latest_obs['image'] = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
    def left_pose_cb(self, msg):
        with self.lock: self.latest_obs['left_wrist_pose'] = msg
    def right_pose_cb(self, msg):
        with self.lock: self.latest_obs['right_wrist_pose'] = msg
    def left_kp_cb(self, msg):
        with self.lock: self.latest_obs['left_keypoints'] = np.array(msg.data)
    def right_kp_cb(self, msg):
        with self.lock: self.latest_obs['right_keypoints'] = np.array(msg.data)

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

if __name__ == '__main__':
    main()