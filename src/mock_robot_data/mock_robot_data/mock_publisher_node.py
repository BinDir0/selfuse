#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from cv_bridge import CvBridge


class MockRobotDataNode(Node):
    def __init__(self):
        super().__init__('mock_robot_data_node')
        self.cv_bridge = CvBridge()

        # 参数配置
        self.declare_parameter('camera_name', 'head')
        self.cam_name = self.get_parameter('camera_name').value
        # pi0.5 EgoHands trained with include_breast=True, so the mock must
        # also publish a breast view or the client never completes first-obs.
        self.declare_parameter('breast_camera_name', 'breast')
        self.declare_parameter('use_breast', True)
        self.breast_cam_name = self.get_parameter('breast_camera_name').value
        self.use_breast = self.get_parameter('use_breast').value

        # --- 1. 定义 Callback Groups ---
        # 使用独立的互斥组确保高频定时器不互相干扰
        self.group_cam = MutuallyExclusiveCallbackGroup()
        self.group_hand = MutuallyExclusiveCallbackGroup()
        self.group_arm = MutuallyExclusiveCallbackGroup()

        # --- 2. 初始化发布者 ---
        # 相机 30Hz - 增加队列大小以避免阻塞
        self.pub_rgb = self.create_publisher(Image, f'/camera/{self.cam_name}/rgb', 10)
        self.pub_depth = self.create_publisher(Image, f'/camera/{self.cam_name}/depth', 10)
        if self.use_breast:
            self.pub_breast = self.create_publisher(Image, f'/camera/{self.breast_cam_name}/rgb', 10)
        
        # 手部 80Hz
        self.pub_hand_l = self.create_publisher(PoseArray, '/state/left_hand/keypoints', 10)
        self.pub_hand_r = self.create_publisher(PoseArray, '/state/right_hand/keypoints', 10)
        
        # 臂部 100Hz
        self.pub_arm_l = self.create_publisher(PoseStamped, '/state/left_arm/wrist_pose', 10)
        self.pub_arm_r = self.create_publisher(PoseStamped, '/state/right_arm/wrist_pose', 10)

        # --- 3. 预分配图像数组以提升性能 ---
        # 初始化时生成一次随机数据（或全0），之后不再更新，只更新时间戳
        # 这样可以避免每次回调都生成大量随机数的开销
        self.rgb_data = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        self.depth_data = np.random.randint(500, 1500, size=(480, 640, 1), dtype=np.uint16)
        
        # 预转换消息对象，只更新时间戳
        self.rgb_msg = self.cv_bridge.cv2_to_imgmsg(self.rgb_data, encoding='rgb8')
        self.rgb_msg.header.frame_id = f"{self.cam_name}_camera_optical_frame"
        
        self.depth_msg = self.cv_bridge.cv2_to_imgmsg(self.depth_data, encoding='16UC1')
        self.depth_msg.header.frame_id = f"{self.cam_name}_camera_optical_frame"

        if self.use_breast:
            self.breast_data = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
            self.breast_msg = self.cv_bridge.cv2_to_imgmsg(self.breast_data, encoding='rgb8')
            self.breast_msg.header.frame_id = f"{self.breast_cam_name}_camera_optical_frame"

        # --- 4. 创建定时器 ---
        # 相机定时器: 30Hz (约 33.3ms)
        self.create_timer(1.0/30.0, self.timer_cam_cb, callback_group=self.group_cam)
        
        # 手部定时器: 80Hz (12.5ms)
        self.create_timer(1.0/80.0, self.timer_hand_cb, callback_group=self.group_hand)
        
        # 臂部定时器: 100Hz (10ms)
        self.create_timer(1.0/100.0, self.timer_arm_cb, callback_group=self.group_arm)

        self.get_logger().info(
            f"Mock数据发生器已启动 | 相机: 30Hz (head"
            f"{' + breast' if self.use_breast else ''}) | 手部: 80Hz | 臂部: 100Hz")

    # --- 定时器回调逻辑 ---

    def timer_cam_cb(self):
        """模拟 30Hz 相机数据"""
        # 只更新时间戳，数据在初始化时已经生成，不再更新
        now = self.get_clock().now().to_msg()
        
        self.rgb_msg.header.stamp = now
        self.depth_msg.header.stamp = now

        # 发布消息（非阻塞）
        self.pub_rgb.publish(self.rgb_msg)
        self.pub_depth.publish(self.depth_msg)
        if self.use_breast:
            self.breast_msg.header.stamp = now
            self.pub_breast.publish(self.breast_msg)

    def timer_hand_cb(self):
        """模拟 80Hz 手部 5 指关键点数据"""
        now = self.get_clock().now().to_msg()
        
        def create_mock_kps(side):
            pa = PoseArray()
            pa.header.stamp = now
            pa.header.frame_id = f"{side}_wrist_link"
            for i in range(5): # 5个手指
                p = Pose()
                p.position.x = 0.05 + 0.01 * i
                p.position.y = 0.02 * np.sin(time.time() * 2) # 加入微小正弦波动
                p.position.z = 0.1
                pa.poses.append(p)
            return pa

        self.pub_hand_l.publish(create_mock_kps("left"))
        self.pub_hand_r.publish(create_mock_kps("right"))

    def timer_arm_cb(self):
        """模拟 100Hz 臂部末端位姿"""
        now = self.get_clock().now().to_msg()
        
        def create_mock_pose(side):
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "base_link"
            # 模拟机器人左右手的基准位置
            ps.pose.position.x = 0.4
            ps.pose.position.y = 0.2 if side == "left" else -0.2
            ps.pose.position.z = 0.3 + 0.05 * np.cos(time.time())
            # 单位四元数
            ps.pose.orientation.w = 1.0
            return ps

        self.pub_arm_l.publish(create_mock_pose("left"))
        self.pub_arm_r.publish(create_mock_pose("right"))

def main(args=None):
    rclpy.init(args=args)
    node = MockRobotDataNode()
    # 使用多线程执行器确保高频 Timer 不会被阻塞
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()