#!/usr/bin/env python3
"""
Camera Node - 用于Inference系统的相机节点
按照图中架构，以30Hz频率发布RGB-D图像到指定topic
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from camera.realsense_image_module import RealSenseImage


class CameraNode(Node):
    """
    相机节点实现
    从RealSense摄像头读取图像并发布到ROS话题
    
    按照架构图：
    - 频率: 30Hz
    - 输出: Image类型
    - Topic: /camera/{head,chest}/depth 和 /camera/{head,chest}/rgb
    """
    
    def __init__(self):
        super().__init__('camera_node')
        
        # 声明参数
        self.declare_parameter('camera_name', 'head')  # 相机名称: head 或 chest
        self.declare_parameter('serial_number', '')    # 相机序列号
        self.declare_parameter('frequency', 30.0)      # 发布频率，默认30Hz
        self.declare_parameter('width', 640)           # 图像宽度
        self.declare_parameter('height', 480)          # 图像高度
        
        # 获取参数值
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        serial_number = self.get_parameter('serial_number').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        
        # 如果序列号为空字符串，则设置为None
        self.serial_number = serial_number if serial_number else None
        
        # 创建发布者 - 按照图中的接口输出
        # Topic: /camera/{head,chest}/rgb
        self.rgb_publisher = self.create_publisher(
            Image, 
            f'/camera/{self.camera_name}/rgb', 
            10
        )
        
        # Topic: /camera/{head,chest}/depth
        self.depth_publisher = self.create_publisher(
            Image,
            f'/camera/{self.camera_name}/depth',
            10
        )
        
        # 图像转换桥
        self.bridge = CvBridge()
        
        # 初始化RealSense相机
        try:
            self.realsense_camera = RealSenseImage(
                SN_number=self.serial_number,
                width=self.width,
                height=self.height,
                fps=int(self.frequency)
            )
            self.get_logger().info(f'成功初始化RealSense相机')
            if self.serial_number:
                self.get_logger().info(f'使用序列号: {self.serial_number}')
            else:
                self.get_logger().info('使用默认RealSense设备')
        except Exception as e:
            self.get_logger().error(f"无法初始化RealSense相机: {str(e)}")
            raise
        
        # 创建定时器，使用参数指定的发布频率
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.capture_and_publish)
        
        # 统计信息
        self.frame_count = 0
        
        self.get_logger().info(f'Camera Node已启动')
        self.get_logger().info(f'相机名称: {self.camera_name}')
        self.get_logger().info(f'发布频率: {self.frequency} Hz')
        self.get_logger().info(f'图像尺寸: {self.width}x{self.height}')
        self.get_logger().info(f'发布话题:')
        self.get_logger().info(f'  - /camera/{self.camera_name}/rgb (Image)')
        self.get_logger().info(f'  - /camera/{self.camera_name}/depth (Image)')
    
    def capture_and_publish(self):
        """捕获图像并发布"""
        try:
            # 从RealSense相机同时获取RGB和深度图像
            rgb_frame, depth_frame = self.realsense_camera.capture_rgb_depth_frames()
            
            if rgb_frame is not None and depth_frame is not None:
                # 获取当前时间戳
                current_time = self.get_clock().now().to_msg()
                frame_id = f'{self.camera_name}_camera_frame'
                
                # 转换RGB图像为ROS消息 (RealSense输出RGB格式)
                ros_rgb = self.bridge.cv2_to_imgmsg(rgb_frame, encoding='rgb8')
                ros_rgb.header.stamp = current_time
                ros_rgb.header.frame_id = frame_id
                
                # 转换深度图像为ROS消息 (深度图为16位单通道)
                ros_depth = self.bridge.cv2_to_imgmsg(depth_frame, encoding='16UC1')
                ros_depth.header.stamp = current_time
                ros_depth.header.frame_id = frame_id
                
                # 发布图像
                self.rgb_publisher.publish(ros_rgb)
                self.depth_publisher.publish(ros_depth)
                
                # 更新统计信息
                self.frame_count += 1
                
                # 每100帧输出一次日志
                if self.frame_count % 100 == 0:
                    self.get_logger().info(
                        f'已发布 {self.frame_count} 帧图像 '
                        f'(RGB: {rgb_frame.shape}, Depth: {depth_frame.shape})'
                    )
            else:
                self.get_logger().warn('无法从RealSense相机读取图像或深度数据')
                
        except Exception as e:
            self.get_logger().error(f'图像捕获或转换错误: {str(e)}')
    
    def destroy_node(self):
        """节点销毁时释放相机资源"""
        self.get_logger().info(f'关闭Camera Node，共发布了 {self.frame_count} 帧图像')
        if hasattr(self, 'realsense_camera'):
            self.realsense_camera.close()
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'节点运行错误: {str(e)}')
        else:
            print(f'节点初始化错误: {str(e)}')
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

