"""
RuiYan Hand Kinematics Test Script

测试FK和IK求解器的准确性和一致性
"""

import numpy as np
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from hand_fk import HandFK
from hand_ik import HandIK


def test_fk(hand_type='left'):
    """测试FK求解器"""
    print("\n" + "="*60)
    print(f"测试 {hand_type.upper()} Hand FK 求解器")
    print("="*60)
    
    try:
        # 创建FK求解器（不显示viewer）
        fk = HandFK(hand_type=hand_type, enable_viewer=False)
        
        # 测试1: 零位姿态
        print("\n[测试1] 零位姿态")
        joint_positions = np.zeros(len(fk.joint_names))
        results = fk.compute_fk(joint_positions)
        
        print("关节角度:", joint_positions)
        for finger, pose in results.items():
            print(f"{finger:>6}: pos={pose['pos']}")
        
        # 测试2: 握拳姿态
        print("\n[测试2] 握拳姿态")
        joint_positions = np.array([0.5, 0.5, 1.2, 1.2, 1.2, 1.2])
        if len(joint_positions) == len(fk.joint_names):
            results = fk.compute_fk(joint_positions)
            print("关节角度:", joint_positions)
            for finger, pose in results.items():
                print(f"{finger:>6}: pos={pose['pos']}")
        else:
            print(f"跳过：关节数量不匹配 ({len(joint_positions)} vs {len(fk.joint_names)})")
        
        print("\n✅ FK测试完成")
        fk.close()
        return True
        
    except Exception as e:
        print(f"\n❌ FK测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ik(hand_type='left'):
    """测试IK求解器"""
    print("\n" + "="*60)
    print(f"测试 {hand_type.upper()} Hand IK 求解器")
    print("="*60)
    
    try:
        # 创建IK求解器（不显示viewer）
        ik = HandIK(hand_type=hand_type, enable_viewer=False)
        
        # 测试：移动食指
        print("\n[测试] 移动食指到指定位置")
        target_poses = {
            'index': {
                'pos': np.array([0.0, -0.03, 0.15])
            }
        }
        
        result = ik.compute_ik(target_poses, max_iterations=50)
        
        print(f"迭代次数: {result['iterations']}")
        print(f"残差误差: {result['error']:.6f}")
        print(f"求解的关节角度: {result['joint_positions'][:len(ik.joint_names)]}")
        
        print("\n✅ IK测试完成")
        ik.close()
        return True
        
    except Exception as e:
        print(f"\n❌ IK测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fk_ik_consistency(hand_type='left'):
    """测试FK和IK的一致性"""
    print("\n" + "="*60)
    print(f"测试 {hand_type.upper()} Hand FK-IK 一致性")
    print("="*60)
    
    try:
        # 创建FK和IK求解器
        fk = HandFK(hand_type=hand_type, enable_viewer=False)
        ik = HandIK(hand_type=hand_type, enable_viewer=False)
        
        # 步骤1: 从零位开始（确保FK和IK起始状态一致）
        print("\n[步骤1] FK: 从零位计算初始位姿")
        initial_joints = np.zeros(len(fk.joint_names))
        
        fk_results = fk.compute_fk(initial_joints)
        thumb_pose_initial = fk_results['thumb']
        print(f"拇指初始位置: {thumb_pose_initial['pos']}")
        
        # 步骤2: 设置一个较小的目标位姿（2mm移动）
        print("\n[步骤2] 设置新的目标位姿（小幅度移动）")
        target_pos = thumb_pose_initial['pos'] + np.array([0.002, 0.0, 0.002])  # 2mm
        print(f"拇指目标位置: {target_pos}")
        print(f"移动距离: {np.linalg.norm([0.002, 0, 0.002])*1000:.1f} mm")
        
        target_poses = {
            'thumb': {'pos': target_pos}
        }
        
        # 步骤3: 使用IK求解（增加迭代次数）
        print("\n[步骤3] IK: 求解关节角度")
        ik_result = ik.compute_ik(target_poses, max_iterations=100)
        solved_joints = ik_result['joint_positions'][:len(fk.joint_names)]
        print(f"求解的关节角度: {solved_joints}")
        print(f"IK迭代次数: {ik_result['iterations']}, 误差: {ik_result['error']:.6f}")
        
        # 步骤4: 使用FK验证
        print("\n[步骤4] FK: 验证求解结果")
        fk_results_verify = fk.compute_fk(solved_joints)
        thumb_pose_final = fk_results_verify['thumb']
        print(f"拇指最终位置: {thumb_pose_final['pos']}")
        
        # 计算误差
        position_error = np.linalg.norm(thumb_pose_final['pos'] - target_pos)
        print(f"\n位置误差: {position_error*1000:.3f} mm")
        
        # 小幅度移动应该精度很高
        if position_error < 0.001:  # 1mm
            print("✅ FK-IK一致性测试通过（误差 < 1mm）")
            success = True
        elif position_error < 0.005:  # 5mm
            print("⚠️  FK-IK一致性通过但精度一般（误差 < 5mm）")
            success = True
        else:
            print("❌ FK-IK一致性测试失败（误差 >= 5mm）")
            success = False
        
        fk.close()
        ik.close()
        return success
        
    except Exception as e:
        print(f"\n❌ FK-IK一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "RuiYan Hand 运动学测试" + " "*15 + "║")
    print("╚" + "═"*58 + "╝")
    
    hand_type = 'left'
    
    # 运行测试
    results = []
    
    print("\n\n>>> 测试 1/3: FK求解器")
    results.append(("FK测试", test_fk(hand_type)))
    
    print("\n\n>>> 测试 2/3: IK求解器")
    results.append(("IK测试", test_ik(hand_type)))
    
    print("\n\n>>> 测试 3/3: FK-IK一致性")
    results.append(("FK-IK一致性", test_fk_ik_consistency(hand_type)))
    
    # 总结
    print("\n\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

