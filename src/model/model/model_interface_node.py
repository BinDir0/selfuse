#!/usr/bin/env python3
"""
Model Interface Node - 模型接口节点 (WebSocket Client)
根据架构图：
- 频率: ? Hz（未在图中明确标注）
- 作为WebSocket client连接到远程Model server
- 输入（订阅）:
  * Image: /camera/{head,chest}/depth
  * Image: /camera/{head,chest}/rgb
  * JointState: /state/{left,right}_arm/joints
  * ? DataType: /state/{left,right}_hand/keypoints
- 输出（发布）:
  * PoseStamped: /action/both_arms/wrist_poses (arm actions - wrist poses in camera frame)
  * ? DataType: /action/{left,right}_hand/keypoints (hand actions - keypoints in wrist frame)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import asyncio
import websockets
import json
import numpy as np
from typing import Dict, Any, Optional
import threading


class ModelInterfaceNode(Node):
    """
    模型接口节点 - WebSocket客户端
    连接ROS系统和远程模型服务器
    """
    
    def __init__(self):
        super().__init__('model_interface_node')
        
        # 声明参数
        self.declare_parameter('model_server_url', 'ws://localhost:8765')
        self.declare_parameter('frequency', 10.0)  # 推理请求频率
        self.declare_parameter('instruction', 'pick up the cup')  # 默认指令
        
        # 获取参数
        self.model_server_url = self.get_parameter('model_server_url').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.instruction = self.get_parameter('instruction').get_parameter_value().string_value
        
        # CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # 数据缓存
        self.head_rgb = None
        self.head_depth = None
        self.chest_rgb = None
        self.chest_depth = None
        self.left_arm_state = None
        self.right_arm_state = None
        self.left_hand_keypoints = None
        self.right_hand_keypoints = None
        
        # 状态历史
        self.state_history = []
        self.max_history_length = 10
        
        # 订阅相机图像
        self.head_rgb_sub = self.create_subscription(
            Image, '/camera/head/rgb', self.head_rgb_callback, 10
        )
        self.head_depth_sub = self.create_subscription(
            Image, '/camera/head/depth', self.head_depth_callback, 10
        )
        self.chest_rgb_sub = self.create_subscription(
            Image, '/camera/chest/rgb', self.chest_rgb_callback, 10
        )
        self.chest_depth_sub = self.create_subscription(
            Image, '/camera/chest/depth', self.chest_depth_callback, 10
        )
        
        # 订阅机械臂状态
        self.left_arm_sub = self.create_subscription(
            JointState, '/state/left_arm/joints', self.left_arm_callback, 10
        )
        self.right_arm_sub = self.create_subscription(
            JointState, '/state/right_arm/joints', self.right_arm_callback, 10
        )
        
        # TODO: 订阅手部关键点状态
        # self.left_hand_sub = self.create_subscription(
        #     ???Type, '/state/left_hand/keypoints', self.left_hand_callback, 10
        # )
        # self.right_hand_sub = self.create_subscription(
        #     ???Type, '/state/right_hand/keypoints', self.right_hand_callback, 10
        # )
        
        # 发布动作
        self.arm_action_pub = self.create_publisher(
            PoseStamped, '/action/both_arms/wrist_poses', 10
        )
        
        # TODO: 发布手部动作
        # self.left_hand_action_pub = self.create_publisher(
        #     ???Type, '/action/left_hand/keypoints', 10
        # )
        # self.right_hand_action_pub = self.create_publisher(
        #     ???Type, '/action/right_hand/keypoints', 10
        # )
        
        # WebSocket连接
        self.websocket = None
        self.ws_connected = False
        
        # 创建定时器 - 定期发送推理请求
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.inference_callback)
        
        self.get_logger().info(f'Model Interface Node 已启动')
        self.get_logger().info(f'模型服务器URL: {self.model_server_url}')
        self.get_logger().info(f'推理频率: {self.frequency} Hz')
        self.get_logger().info(f'默认指令: {self.instruction}')
        
        # 在后台线程中运行WebSocket连接
        self.ws_thread = threading.Thread(target=self.run_websocket, daemon=True)
        self.ws_thread.start()
    
    # ==================== 相机回调 ====================
    
    def head_rgb_callback(self, msg):
        """Head相机RGB图像回调"""
        self.head_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
    def head_depth_callback(self, msg):
        """Head相机深度图像回调"""
        self.head_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
    
    def chest_rgb_callback(self, msg):
        """Chest相机RGB图像回调"""
        self.chest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
    def chest_depth_callback(self, msg):
        """Chest相机深度图像回调"""
        self.chest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
    
    # ==================== 状态回调 ====================
    
    def left_arm_callback(self, msg):
        """左臂状态回调"""
        self.left_arm_state = {
            'position': list(msg.position),
            'velocity': list(msg.velocity) if msg.velocity else [],
            'effort': list(msg.effort) if msg.effort else []
        }
    
    def right_arm_callback(self, msg):
        """右臂状态回调"""
        self.right_arm_state = {
            'position': list(msg.position),
            'velocity': list(msg.velocity) if msg.velocity else [],
            'effort': list(msg.effort) if msg.effort else []
        }
    
    def left_hand_callback(self, msg):
        """左手关键点回调"""
        # TODO: 实现手部关键点处理
        pass
    
    def right_hand_callback(self, msg):
        """右手关键点回调"""
        # TODO: 实现手部关键点处理
        pass
    
    # ==================== 推理逻辑 ====================
    
    def inference_callback(self):
        """定时推理回调"""
        if not self.ws_connected:
            self.get_logger().warn('WebSocket未连接，跳过推理', throttle_duration_sec=5.0)
            return
        
        # 检查是否有必要的数据
        if self.head_rgb is None:
            self.get_logger().warn('等待相机数据...', throttle_duration_sec=5.0)
            return
        
        # 准备推理请求数据
        request_data = self.prepare_inference_request()
        
        # 异步发送推理请求
        asyncio.run_coroutine_threadsafe(
            self.send_inference_request(request_data),
            self.ws_loop
        )
    
    def prepare_inference_request(self) -> Dict[str, Any]:
        """准备推理请求数据"""
        # 构建请求数据
        request = {
            'camera_images': {
                'head_rgb': self._encode_image(self.head_rgb) if self.head_rgb is not None else None,
                'head_depth': self._encode_image(self.head_depth) if self.head_depth is not None else None,
                'chest_rgb': self._encode_image(self.chest_rgb) if self.chest_rgb is not None else None,
                'chest_depth': self._encode_image(self.chest_depth) if self.chest_depth is not None else None,
            },
            'instruction': self.instruction,
            'camera_meta_data': {
                # TODO: 添加相机内参等元数据
                'width': 640,
                'height': 480
            },
            'state_history': self.state_history[-self.max_history_length:],
            'current_state': {
                'left_arm': self.left_arm_state,
                'right_arm': self.right_arm_state,
                'left_hand': self.left_hand_keypoints,
                'right_hand': self.right_hand_keypoints
            },
            'timestamp': self.get_clock().now().nanoseconds
        }
        
        return request
    
    def _encode_image(self, image: np.ndarray) -> Dict[str, Any]:
        """编码图像为可传输格式"""
        # TODO: 可以使用base64编码或者直接发送numpy数组
        # 这里返回shape信息，实际实现时需要编码图像数据
        return {
            'shape': list(image.shape),
            'dtype': str(image.dtype),
            # 'data': base64.b64encode(image.tobytes()).decode('utf-8')
        }
    
    async def send_inference_request(self, request_data: Dict[str, Any]):
        """发送推理请求到模型服务器"""
        if not self.websocket:
            return
        
        try:
            # 发送请求
            await self.websocket.send(json.dumps(request_data))
            
            # 接收响应
            response_str = await self.websocket.recv()
            response = json.loads(response_str)
            
            # 处理响应
            self.process_inference_response(response)
            
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().error('WebSocket连接已关闭')
            self.ws_connected = False
        except Exception as e:
            self.get_logger().error(f'推理请求失败: {str(e)}')
    
    def process_inference_response(self, response: Dict[str, Any]):
        """处理模型返回的动作"""
        try:
            # 提取动作
            action_vlm = response.get('action_chunk_vlm', {})
            action_fm = response.get('action_chunk_fm', {})
            
            # TODO: 根据策略选择使用VLM还是FM的动作
            # 这里使用VLM的动作作为示例
            action = action_vlm
            
            # 发布机械臂动作
            if 'left_arm' in action and 'right_arm' in action:
                self.publish_arm_actions(action['left_arm'], action['right_arm'])
            
            # 发布手部动作
            if 'left_hand' in action:
                self.publish_hand_action('left', action['left_hand'])
            if 'right_hand' in action:
                self.publish_hand_action('right', action['right_hand'])
            
            # 更新状态历史
            self.update_state_history(response)
            
            self.get_logger().debug('成功处理推理响应')
            
        except Exception as e:
            self.get_logger().error(f'处理推理响应时出错: {str(e)}')
    
    def publish_arm_actions(self, left_arm_action, right_arm_action):
        """发布机械臂动作"""
        # TODO: 将动作转换为PoseStamped消息
        # TODO: 根据架构图，应该发布wrist poses in camera frame
        pass
    
    def publish_hand_action(self, side: str, hand_action):
        """发布手部动作"""
        # TODO: 发布手部关键点动作
        pass
    
    def update_state_history(self, response: Dict[str, Any]):
        """更新状态历史"""
        current_state = {
            'timestamp': self.get_clock().now().nanoseconds,
            'left_arm': self.left_arm_state,
            'right_arm': self.right_arm_state,
            'left_hand': self.left_hand_keypoints,
            'right_hand': self.right_hand_keypoints
        }
        
        self.state_history.append(current_state)
        
        # 限制历史长度
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
    
    # ==================== WebSocket连接 ====================
    
    def run_websocket(self):
        """在后台线程运行WebSocket连接"""
        self.ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ws_loop)
        self.ws_loop.run_until_complete(self.connect_websocket())
    
    async def connect_websocket(self):
        """连接到WebSocket服务器"""
        retry_delay = 5.0
        
        while True:
            try:
                self.get_logger().info(f'正在连接到模型服务器: {self.model_server_url}')
                
                async with websockets.connect(self.model_server_url) as websocket:
                    self.websocket = websocket
                    self.ws_connected = True
                    self.get_logger().info('WebSocket连接成功')
                    
                    # 保持连接
                    await asyncio.Future()  # 永久等待，直到连接断开
                    
            except Exception as e:
                self.get_logger().error(f'WebSocket连接失败: {str(e)}')
                self.ws_connected = False
                self.websocket = None
                
                self.get_logger().info(f'{retry_delay}秒后重试...')
                await asyncio.sleep(retry_delay)
    
    def destroy_node(self):
        """节点销毁"""
        self.get_logger().info('关闭Model Interface Node')
        self.ws_connected = False
        if self.websocket:
            asyncio.run_coroutine_threadsafe(
                self.websocket.close(),
                self.ws_loop
            )
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = ModelInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'节点运行错误: {str(e)}')
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

