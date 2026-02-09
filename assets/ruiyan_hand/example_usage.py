"""
RuiYan Hand FK/IK 使用示例

展示如何使用 HandFK 和 HandIK 的核心接口：
- FK: 输入关节角度 -> 输出指尖位置
- IK: 输入指尖位置 -> 输出关节角度
"""

import numpy as np
from hand_fk import HandFK
from hand_ik import HandIK


def example_fk():
    """
    示例1: 正运动学 (FK)
    输入: 关节角度 -> 输出: 指尖位置
    """
    print("\n" + "="*70)
    print("示例1: 正运动学 (FK)")
    print("="*70)
    
    # 1. 创建FK求解器（可选择左手/右手）
    fk_solver = HandFK(
        hand_type='left',      # 'left' 或 'right'
        enable_viewer=False    # 不启用可视化（加快速度）
    )
    
    # 2. 定义关节角度（6个主动关节）
    # 方式A: 使用归一化值 (0-1, 推荐)
    joint_angles_normalized = np.array([
        0.3,  # 拇指第1关节 (0=完全伸展, 1=完全弯曲)
        0.5,  # 拇指第2关节
        0.8,  # 食指
        0.8,  # 中指
        0.8,  # 无名指
        0.8,  # 小指
    ])
    
    # 方式B: 使用弧度值（高级用法）
    # joint_angles_radians = np.array([0.5, 0.3, 1.2, 1.2, 1.2, 1.2])
    
    # 3. 计算FK（返回5个指尖的位置和姿态）
    result = fk_solver.compute_fk(
        joint_positions=joint_angles_normalized,
        use_normalized=True  # True表示输入是归一化值，False表示弧度
    )
    
    # 4. 读取结果
    print("\n指尖位置和姿态:")
    print("-" * 70)
    
    for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        if finger_name in result:
            pos = result[finger_name]['pos']    # [x, y, z] 位置（米）
            quat = result[finger_name]['quat']  # [x, y, z, w] 四元数姿态
            
            finger_cn = {
                'thumb': '拇指', 'index': '食指', 'middle': '中指',
                'ring': '无名指', 'pinky': '小指'
            }[finger_name]
            
            print(f"{finger_cn}:")
            print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            print(f"  姿态: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
    
    # 5. 关闭求解器
    fk_solver.close()
    
    return result


def example_ik():
    """
    示例2: 逆运动学 (IK)
    输入: 指尖位置 -> 输出: 关节角度
    """
    print("\n" + "="*70)
    print("示例2: 逆运动学 (IK)")
    print("="*70)
    
    # 1. 创建IK求解器（可选择左手/右手）
    ik_solver = HandIK(
        hand_type='left',      # 'left' 或 'right'
        enable_viewer=False,   # 不启用可视化（加快速度）
        solver='daqp',         # 求解器类型
        frequency=100.0        # 求解频率(Hz)，默认100Hz
    )
    
    # 2. 定义目标位置（可以只控制部分手指）
    target_poses = {
        # 只需要指定你想控制的手指
        'thumb': {
            'pos': np.array([0.05, 0.04, 0.05]),  # 目标位置 [x, y, z] (米)
            # 'quat': np.array([0, 0, 0, 1])      # 可选：目标姿态 [x, y, z, w]
        },
        'index': {
            'pos': np.array([0.08, 0.03, 0.04]),
        },
        # 可以添加更多手指: 'middle', 'ring', 'pinky'
    }
    
    # 3. 求解IK（返回关节角度）
    result = ik_solver.compute_ik(
        target_poses=target_poses,
        max_iterations=10,      # 最大迭代次数（默认10次）
        return_normalized=False # True返回归一化值，False返回弧度
    )
    
    # 4. 读取结果
    print("\n求解结果:")
    print("-" * 70)
    print(f"迭代次数: {result['iterations']}")
    print(f"位置误差: {result['position_error']*1000:.3f} mm")
    print(f"速度范数: {result['velocity_error']:.6f}")
    print(f"收敛状态: {'✅ 收敛' if result['converged'] else '⚠️  未完全收敛'}")
    
    # 获取主动关节角度（6个关节，弧度）
    joint_angles = result['active_joint_positions']
    print(f"\n关节角度 (弧度):")
    print(f"  {joint_angles}")
    
    # 如果需要归一化值，可以转换：
    joint_angles_normalized = ik_solver._radians_to_normalized(joint_angles)
    print(f"\n关节角度 (归一化 0-1):")
    print(f"  {joint_angles_normalized}")
    
    # 5. 关闭求解器
    ik_solver.close()
    
    return result


def example_fk_ik_loop():
    """
    示例3: FK-IK闭环验证
    用FK计算目标位置 -> 用IK求解关节角度 -> 用FK验证结果
    """
    print("\n" + "="*70)
    print("示例3: FK-IK闭环验证")
    print("="*70)
    
    # 创建FK和IK求解器
    fk_solver = HandFK(hand_type='left', enable_viewer=False)
    ik_solver = HandIK(hand_type='left', enable_viewer=False)
    
    # Step 1: 用FK计算一个目标位置
    print("\nStep 1: 用FK计算目标位置")
    target_joints = np.array([0.3, 0.5, 0.7, 0.7, 0.7, 0.7])
    fk_result = fk_solver.compute_fk(target_joints, use_normalized=True)
    
    target_pos_thumb = fk_result['thumb']['pos']
    target_pos_index = fk_result['index']['pos']
    print(f"目标拇指位置: {target_pos_thumb}")
    print(f"目标食指位置: {target_pos_index}")
    
    # Step 2: 用IK求解到达该位置的关节角度
    print("\nStep 2: 用IK求解关节角度")
    target_poses = {
        'thumb': {'pos': target_pos_thumb},
        'index': {'pos': target_pos_index},
    }
    ik_result = ik_solver.compute_ik(target_poses, max_iterations=10)
    solved_joints = ik_result['active_joint_positions']
    print(f"求解的关节角度: {solved_joints}")
    
    # Step 3: 用FK验证IK结果
    print("\nStep 3: 用FK验证IK结果")
    verify_result = fk_solver.compute_fk(solved_joints, use_normalized=False)
    
    verify_pos_thumb = verify_result['thumb']['pos']
    verify_pos_index = verify_result['index']['pos']
    
    error_thumb = np.linalg.norm(verify_pos_thumb - target_pos_thumb)
    error_index = np.linalg.norm(verify_pos_index - target_pos_index)
    
    print(f"验证拇指位置: {verify_pos_thumb}")
    print(f"验证食指位置: {verify_pos_index}")
    print(f"\n位置误差:")
    print(f"  拇指: {error_thumb*1000:.3f} mm")
    print(f"  食指: {error_index*1000:.3f} mm")
    
    # 关闭求解器
    fk_solver.close()
    ik_solver.close()


def simple_fk_function(joint_angles, hand_type='left', normalized=True):
    """
    简化的FK函数接口
    
    Args:
        joint_angles: 6个关节角度的数组
        hand_type: 'left' 或 'right'
        normalized: True表示输入是0-1归一化值，False表示弧度
    
    Returns:
        dict: 5个指尖的位置和姿态
    """
    fk_solver = HandFK(hand_type=hand_type, enable_viewer=False)
    result = fk_solver.compute_fk(joint_angles, use_normalized=normalized)
    fk_solver.close()
    return result


def simple_ik_function(target_positions, hand_type='left', return_normalized=True):
    """
    简化的IK函数接口

    Args:
        target_positions: dict，格式为 {'thumb': [x,y,z], 'index': [x,y,z], ...}
        hand_type: 'left' 或 'right'
        return_normalized: True返回0-1归一化值，False返回弧度
    
    Returns:
        numpy.ndarray: 6个关节角度
    """
    ik_solver = HandIK(hand_type=hand_type, enable_viewer=False)
    
    # 转换格式
    target_poses = {}
    for finger_name, pos in target_positions.items():
        target_poses[finger_name] = {'pos': np.array(pos)}
    
    result = ik_solver.compute_ik(target_poses, max_iterations=10)
    joint_angles = result['active_joint_positions']
    
    if return_normalized:
        joint_angles = ik_solver._radians_to_normalized(joint_angles)
    
    ik_solver.close()
    return joint_angles


def example_simple_api():
    """
    示例4: 使用简化的函数接口
    """
    print("\n" + "="*70)
    print("示例4: 使用简化的函数接口")
    print("="*70)
    
    # FK: 输入关节 -> 输出位置
    print("\n【FK】输入关节 -> 输出位置:")
    joints = np.array([0.3, 0.5, 0.7, 0.7, 0.7, 0.7])
    positions = simple_fk_function(joints, hand_type='left', normalized=True)
    print(f"输入关节: {joints}")
    print(f"输出拇指位置: {positions['thumb']['pos']}")
    print(f"输出食指位置: {positions['index']['pos']}")
    
    # IK: 输入位置 -> 输出关节
    print("\n【IK】输入位置 -> 输出关节:")
    target_pos = {
        'thumb': [0.05, 0.04, 0.05],
        'index': [0.08, 0.03, 0.04],
    }
    solved_joints = simple_ik_function(target_pos, hand_type='left', return_normalized=True)
    print(f"输入目标位置: {target_pos}")
    print(f"输出关节: {solved_joints}")


if __name__ == '__main__':
    """
    运行所有示例
    """
    print("\n" + "="*70)
    print("RuiYan Hand FK/IK 使用示例")
    print("="*70)
    
    # 运行示例
    example_fk()              # 示例1: FK基本用法
    example_ik()              # 示例2: IK基本用法
    example_fk_ik_loop()      # 示例3: FK-IK闭环验证
    example_simple_api()      # 示例4: 简化API
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)

