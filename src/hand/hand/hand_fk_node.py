#!/usr/bin/env python3
"""
Hand FK Node - 手部正运动学节点
根据架构图：
- 频率: 80Hz
- 输入: /state/{left,right}_hand/joints (JointState) - hand states (joints)
- 输出: 发送到Hand Control Node
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class HandFKNode(Node):
    """
    手部正运动学节点
    将关节状态转换为末端执行器位姿
    """
    
    def __init__(self):
        super().__init__('hand_fk_node')
        
        # 声明参数
        self.declare_parameter('hand_side', 'left')  # 'left' or 'right'
        self.declare_parameter('frequency', 80.0)    # 80Hz
        
        # 获取参数
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        
        # 订阅手部关节状态
        self.state_sub = self.create_subscription(
            JointState,
            f'/state/{self.hand_side}_hand/joints',
            self.state_callback,
            10
        )
        
        # TODO: 创建发布者（发布到Hand Control Node）
        
        # 创建定时器
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Hand FK Node 已启动')
        self.get_logger().info(f'手部: {self.hand_side}')
        self.get_logger().info(f'频率: {self.frequency} Hz')
        self.get_logger().info(f'订阅: /state/{self.hand_side}_hand/joints')
        
        # TODO: 初始化FK solver
    
    def state_callback(self, msg):
        """处理接收到的关节状态"""
        # TODO: 实现FK计算逻辑
        pass
    
    def timer_callback(self):
        """定时计算和发布"""
        # TODO: 实现FK计算和发布逻辑
        pass


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = HandFKNode()
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

