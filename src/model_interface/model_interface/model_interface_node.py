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

# --- 系统状态枚举 ---
class SystemState(Enum):
    IDLE = 0        # 待机：等待终端输入指令
    READY = 1       # 就绪：指令已输入
    FIRST_OBS = 2   # 冷启动：等待传感器各 Topic 接收第一帧
    INFERENCE = 3   # 推理：正在请求服务器（此时冻结观测更新）
    RUNNING = 4     # 执行：Deploy 模式连续下发动作块
    PAUSED = 5      # 暂停：手动暂停
    STEP_WAIT = 6   # 等待：Debug 模式等待按键触发（计入不活跃时间）
    STEP_ONCE = 7   # 步进：Debug 模式下发一帧动作
    RESETTING = 8   # 归位：系统重置中

# --- 坐标转换与 6D 旋转恢复工具 ---
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
    """提取旋转矩阵前两列拼接成 6D 旋转向量"""
    rot_mat = T[:3, :3]
    return np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])

def matrix_from_6d_rot(trans, rot6d):
    """从 6D 旋转恢复 4x4 矩阵 (Gram-Schmidt 正交化)"""
    v1 = rot6d[:3]
    v2 = rot6d[3:]
    
    e1 = v1 / (np.linalg.norm(v1) + 1e-6)
    e2 = v2 - np.dot(e1, v2) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-6)
    e3 = np.cross(e1, e2)
    
    rot_mat = np.stack([e1, e2, e3], axis=1)
    
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = trans
    return T

class ModelInterfaceNode(Node):
    def __init__(self):
        super().__init__('model_interface_node')

        # --- 1. 锁与事件机制 ---
        self.state_lock = threading.Lock()
        self.infer_event = threading.Event()

        # --- 2. 参数获取 ---
        self.declare_parameter('control_frequency', 30.0)      
        self.declare_parameter('model_server_host', '0.0.0.0')
        self.declare_parameter('model_server_port', 8000)
        self.declare_parameter('calibration_path', '')
        self.declare_parameter('camera_name', 'head')
        self.declare_parameter('audio_service_host', 'localhost')
        self.declare_parameter('audio_service_port', 8080)
        
        self.declare_parameter('data_frequency', 30.0)     # 对齐频率
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
        self.s_horizon = self.get_parameter('state_horizon').value
        self.s_stride = self.get_parameter('state_stride').value
        self.i_horizon = self.get_parameter('image_horizon').value
        self.i_stride = self.get_parameter('image_stride').value
        self.act_exec_len = self.get_parameter('action_execution_len').value
        self.max_buffer = self.get_parameter('buffer_size').value

        # --- 3. 虚拟时间轴管理变量 ---
        self.first_ts_ns = 0            #Episode 起始时间
        self.total_inactive_ns = 0      # 累计非记录状态时长
        self.inactive_start_ns = None   # 停表起始点

        # --- 4. 标定参数 ---
        self.T_cam2base_l, self.K_mat = self.find_and_load_calibration(self.cam_name, 'left')
        self.T_cam2base_r, _ = self.find_and_load_calibration(self.cam_name, 'right')
        self.T_base2cam_l = np.linalg.inv(self.T_cam2base_l)
        self.T_base2cam_r = np.linalg.inv(self.T_cam2base_r)

        # --- 5. 状态机与缓冲区 ---
        self.state = SystemState.IDLE
        self.mode = 'deploy'
        self.current_instruction = ""
        
        # 统一动作队列
        self.action_queue = deque(maxlen=200)
        
        # 传感器历史缓冲区 (virtual_timestamp_ns, data)
        self.buf_rgb = deque(maxlen=self.max_buffer)
        self.buf_depth = deque(maxlen=self.max_buffer)
        self.buf_l_wrist = deque(maxlen=self.max_buffer)
        self.buf_r_wrist = deque(maxlen=self.max_buffer)
        self.buf_l_kps = deque(maxlen=self.max_buffer)
        self.buf_r_kps = deque(maxlen=self.max_buffer)
        
        self.cv_bridge = CvBridge()
        self.audio_session = requests.Session()

        # --- 6. ROS 通信 ---
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
            self.get_logger().info("✅ 成功连接推理服务器")
        except Exception as e:
            self.get_logger().error(f"❌ 无法连接服务器: {e}")

        # --- 8. 启动子线程 ---
        self.kbd_listener = keyboard.Listener(on_press=self.on_key_press)
        self.kbd_listener.start()
        threading.Thread(target=self.user_input_loop, daemon=True).start()
        threading.Thread(target=self.inference_worker, daemon=True).start()

        # 统一 30Hz 控制定时器
        self.control_timer = self.create_timer(1.0 / self.ctrl_freq, self.control_timer_cb)

        self.get_logger().info(f"🚀 Model Interface 节点已启动。控制/采样频率: {self.ctrl_freq}Hz")

    # --- 逻辑控制：状态与时间补偿 ---
    def _is_recording_state(self, state):
        """判断是否需要记录观测数据"""
        return state in [SystemState.FIRST_OBS, SystemState.RUNNING, SystemState.STEP_ONCE]

    def _switch_state(self, new_state):
        """统一管理状态切换：计算并扣除非活跃时长 (包含 INFERENCE, STEP_WAIT, PAUSED)"""
        with self.state_lock:
            old_state = self.state
            if old_state == new_state: return

            now_ns = self.get_clock().now().nanoseconds
            
            # 记录 -> 停止：开启停表
            if self._is_recording_state(old_state) and not self._is_recording_state(new_state):
                self.inactive_start_ns = now_ns
            
            # 停止 -> 记录：结算并累加 Gap 时长
            if not self._is_recording_state(old_state) and self._is_recording_state(new_state):
                if self.inactive_start_ns is not None:
                    self.total_inactive_ns += (now_ns - self.inactive_start_ns)
                    self.inactive_start_ns = None

            self.state = new_state
            self.get_logger().info(f"状态流转: {old_state.name} -> {new_state.name}")

    def _get_msg_ts(self, header):
        return header.stamp.sec * 10**9 + header.stamp.nanosec

    def _update_buf(self, buf, header, data):
        # 仅在活跃状态下记录
        if not self._is_recording_state(self.state):
            return

        raw_ns = self._get_msg_ts(header)
        # 计算虚拟连续时间戳
        virtual_ts = raw_ns - self.first_ts_ns - self.total_inactive_ns
        buf.append((virtual_ts, data))
        
        # 冷启动检查
        if self.state == SystemState.FIRST_OBS:
            self._check_first_obs_complete()

    def _check_first_obs_complete(self):
        bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, 
                self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        if all(len(b) > 0 for b in bufs):
            self._switch_state(SystemState.INFERENCE)
            self.infer_event.set()

    # --- 传感器回调 ---
    def rgb_cb(self, m): self._update_buf(self.buf_rgb, m.header, self.cv_bridge.imgmsg_to_cv2(m, 'rgb8'))
    def depth_cb(self, m): self._update_buf(self.buf_depth, m.header, self.cv_bridge.imgmsg_to_cv2(m, 'passthrough'))
    def l_pose_cb(self, m): self._update_buf(self.buf_l_wrist, m.header, m.pose)
    def r_pose_cb(self, m): self._update_buf(self.buf_r_wrist, m.header, m.pose)
    def l_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses[:5]])
        if len(pts) < 5: pts = np.pad(pts, ((0, 5 - len(pts)), (0, 0)))
        self._update_buf(self.buf_l_kps, m.header, pts.flatten())
    def r_kp_cb(self, m):
        pts = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses[:5]])
        if len(pts) < 5: pts = np.pad(pts, ((0, 5 - len(pts)), (0, 0)))
        self._update_buf(self.buf_r_kps, m.header, pts.flatten())

    # --- 采样工具 ---
    def _find_nearest(self, buf, target_ns):
        if not buf: return None
        times = [x[0] for x in buf]
        idx = bisect.bisect_left(times, target_ns)
        if idx == 0: return buf[0][1]
        if idx == len(times): return buf[-1][1]
        return buf[idx-1][1] if (target_ns - times[idx-1]) < (times[idx] - target_ns) else buf[idx][1]

    def _is_in_range(self, buf, t):
        if not buf: return False
        dt_ns = int(1e9 / self.data_freq)
        # 允许前后扩展一个 dt 的容错采样
        return (buf[0][0] - dt_ns) <= t <= (buf[-1][0] + dt_ns)

    # --- 推理 Payload 准备 (在 INFERENCE 状态下执行，此时观测数据静止，安全读) ---
    def prepare_inference_payload(self):
        all_bufs = [self.buf_rgb, self.buf_depth, self.buf_l_wrist, 
                    self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
        
        # 强制性齐备断言
        assert all(len(b) > 0 for b in all_bufs), "数据缓冲区不全，无法推理"

        # 确定共同采样基准点
        t_ref = min(b[-1][0] for b in all_bufs)
        dt_ns = int(1e9 / self.data_freq)

        # 1. Image 采样
        rgb_seq, depth_seq = [], []
        for h in range(self.i_horizon):
            target_t = t_ref - (h * self.i_stride * dt_ns)
            if not self._is_in_range(self.buf_rgb, target_t): break
            rgb_seq.append(self._find_nearest(self.buf_rgb, target_t))
            depth_seq.append(self._find_nearest(self.buf_depth, target_t))
        
        rgb_in = np.stack(rgb_seq)[::-1]
        depth_in = np.stack(depth_seq)[::-1]
        if depth_in.ndim == 3: depth_in = np.expand_dims(depth_in, axis=-1)

        # 2. State 采样 (48维向量)
        states_list = []
        for h in range(self.s_horizon):
            target_t = t_ref - (h * self.s_stride * dt_ns)
            state_bufs = [self.buf_l_wrist, self.buf_r_wrist, self.buf_l_kps, self.buf_r_kps]
            if not all(self._is_in_range(b, target_t) for b in state_bufs): break

            # 采样并转换
            tl_m = self.T_base2cam_l @ matrix_from_pose_msg(self._find_nearest(self.buf_l_wrist, target_t))
            tr_m = self.T_base2cam_r @ matrix_from_pose_msg(self._find_nearest(self.buf_r_wrist, target_t))
            lk = self._find_nearest(self.buf_l_kps, target_t)
            rk = self._find_nearest(self.buf_r_kps, target_t)

            # 拼装顺序: [L_Trans(3), R_Trans(3), L_Rot6D(6), R_Rot6D(6), L_KPs(15), R_KPs(15)]
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

    # --- 推理生产者：仅在 INFERENCE 状态下请求模型 ---
    def inference_worker(self):
        while rclpy.ok():
            if self.state != SystemState.INFERENCE:
                self.infer_event.wait(timeout=0.1)
                self.infer_event.clear()
                continue

            payload = self.prepare_inference_payload()
            try:
                # 远程推理调用 (不占锁)
                response = self.policy_client.infer(payload)
                pred_actions = response["pred_actions"] # (Tp, 48)
                
                # 动作序列切分
                steps = min(self.act_exec_len, pred_actions.shape[0])
                for i in range(steps):
                    v = pred_actions[i]
                    # 48维解析并入队
                    self.action_queue.append({
                        'l': matrix_from_6d_rot(v[0:3], v[6:12]),
                        'r': matrix_from_6d_rot(v[3:6], v[12:18]),
                        'lk': v[18:33], 'rk': v[33:48]
                    })

                # 推理结束，切回执行/等待状态（计算时间轴 Gap）
                self._switch_state(SystemState.RUNNING if self.mode == 'deploy' else SystemState.STEP_WAIT)

            except Exception as e:
                self.get_logger().error(f"推理故障: {e}")
                time.sleep(0.1)

    # --- 动作下发定时器：消费者逻辑 ---
    def control_timer_cb(self):
        # 原子状态读
        st = self.state
        if st not in [SystemState.RUNNING, SystemState.STEP_ONCE]:
            return

        # 核心逻辑：队列为空再触发推理
        if not self.action_queue:
            # 仅在 RUNNING 或步进完成后的 STEP_WAIT 状态触发下一轮推理
            self._switch_state(SystemState.INFERENCE)
            self.infer_event.set()
            return
        
        data = self.action_queue.popleft()
        is_chunk_done = (len(self.action_queue) == 0)

        # 发布 臂 + 手
        try:
            now = self.get_clock().now().to_msg()
            
            # 1. 臂 PoseArray
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = now, "base_link"
            pa.poses.append(pose_from_matrix(self.T_cam2base_l @ data['l']))
            pa.poses.append(pose_from_matrix(self.T_cam2base_r @ data['r']))
            self.pub_action_poses.publish(pa)

            # 2. 手 PoseArray
            def _pub_kps(topic, kps, frame):
                p_arr = PoseArray()
                p_arr.header.stamp, p_arr.header.frame_id = now, frame
                for pt in kps.reshape(-1, 3):
                    p = Pose(); p.position.x, p.position.y, p.position.z = map(float, pt)
                    p_arr.poses.append(p)
                topic.publish(p_arr)

            _pub_kps(self.pub_action_hand_l, data['lk'], "left_wrist_link")
            _pub_kps(self.pub_action_hand_r, data['rk'], "right_wrist_link")

        except Exception as e:
            self.get_logger().error(f"下发定时器发布异常: {e}")

        # Debug 模式单步处理
        if st == SystemState.STEP_ONCE:
            # 如果动作块还有剩余，回到等待按键状态；如果块空了，切回 INFERENCE
            if is_chunk_done:
                self._switch_state(SystemState.INFERENCE)
            else:
                self._switch_state(SystemState.STEP_WAIT)
            self.infer_event.set()

    # --- 交互线程 ---
    def user_input_loop(self):
        while rclpy.ok():
            if self.state == SystemState.IDLE:
                print("\n" + "="*40)
                instr = input("[Input] 指令: ").strip()
                mode_in = input("[Input] 模式 (deploy/debug) [deploy]: ").strip().lower()
                if not instr: continue
                with self.state_lock:
                    self.current_instruction, self.mode = instr, ('debug' if mode_in == 'debug' else 'deploy')
                    self.state = SystemState.READY
                    # 彻底重置时间轴状态
                    self.first_ts_ns, self.total_inactive_ns, self.inactive_start_ns = 0, 0, None
                print(f"✅ Ready. Mode: {self.mode.upper()}. 按下脚踏板 1 启动。")
            time.sleep(0.1)

    def on_key_press(self, key):
        try: k = key.char
        except: k = None
        
        if k == '1' and self.state == SystemState.READY:
            self.play_sound("start")
            self.pub_system_mode.publish(String(data="inference"))
            # 按下那一刻作为 Episode 的时间原点
            self.first_ts_ns = self.get_clock().now().nanoseconds
            self.total_inactive_ns, self.inactive_start_ns = 0, None
            # 清理缓冲区
            self.buf_rgb.clear(); self.buf_depth.clear(); self.buf_l_wrist.clear()
            self.buf_r_wrist.clear(); self.buf_l_kps.clear(); self.buf_r_kps.clear()
            self._switch_state(SystemState.FIRST_OBS)

        elif (k == '2' or key == keyboard.Key.space):
            if self.mode == 'deploy' and self.state in [SystemState.RUNNING, SystemState.PAUSED]:
                if self.state == SystemState.RUNNING:
                    self._switch_state(SystemState.PAUSED); self.play_sound("pause")
                else:
                    self._switch_state(SystemState.RUNNING); self.play_sound("continue")
            elif self.mode == 'debug' and self.state == SystemState.STEP_WAIT:
                self._switch_state(SystemState.STEP_ONCE); self.play_sound("continue")

        elif k == '3' and self.state != SystemState.IDLE:
            self._switch_state(SystemState.RESETTING); self.play_sound("stop_and_reset"); self.pub_system_mode.publish(String(data="reset"))
            self.action_queue.clear()
            time.sleep(3.0)
            self._switch_state(SystemState.IDLE)

    def find_and_load_calibration(self, cam, arm):
        subdirs = [d for d in os.listdir(self.calib_root) if os.path.isdir(os.path.join(self.calib_root, d))]
        matches = [d for d in subdirs if cam in d and arm in d]
        assert len(matches) == 1
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