import torch
import torch.optim as optim
import sys

def solve_kissing_configuration(n_points, dim, max_steps=2000, attempts=5):
    """
    尝试在 dim 维空间中放置 n_points 个单位球，使其互不重叠。
    即：寻找球面上的 n 个点，使得任意两点 cos(theta) <= 0.5。
    参数:
    - n_points: 尝试放置的球体数量
    - dim: 空间维度
    - attempts: 随机重启次数 (防止陷入局部最优)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for attempt in range(attempts):
        # 1. 随机初始化点分布
        # requires_grad=True 允许 PyTorch 自动计算移动方向
        points = torch.randn(n_points, dim, device=device)
        points = points / points.norm(dim=1, keepdim=True) # 归一化到球面上
        points.requires_grad = True
        
        # 使用 Adam 优化器，带自动学习率衰减
        optimizer = optim.Adam([points], lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # 2. 约束：始终保持点在单位球面上
            # 注意：在计算图中进行归一化，保证梯度正确
            points_norm = points / points.norm(dim=1, keepdim=True)
            
            # 3. 计算所有点对的相似度 (Gram Matrix)
            # cos_matrix[i, j] 是第 i 个点和第 j 个点的余弦相似度
            cos_matrix = torch.mm(points_norm, points_norm.t())
            
            # 4. 定义损失函数 (排斥力)
            # 我们希望所有非对角线元素的 cos 值 <= 0.5
            # mask 掉对角线 (自己和自己的夹角)
            mask = torch.eye(n_points, device=device).bool()
            cos_matrix.masked_fill_(mask, -1.0)
            
            # ReLU: 只对“靠得太近”的点对 (cos > 0.5) 产生 Loss
            violations = torch.relu(cos_matrix - 0.5)
            loss = torch.sum(violations ** 2)
            
            loss.backward()
            optimizer.step()
            
            # 5. 显式投影：每一步更新后，强制拉回到球面上
            with torch.no_grad():
                points.data = points.data / points.data.norm(dim=1, keepdim=True)
            
            scheduler.step(loss)
            
            # 6. 提前终止检查
            if loss.item() < 1e-6:
                # 再次确认最大余弦值是否真的合规 (允许 1e-3 的数值误差)
                if cos_matrix.max().item() <= 0.5 + 1e-3:
                    return True # 成功找到!

    return False # 所有尝试都失败

def find_max_kissing_number(dim):
    """
    主程序：搜索指定维度的最大 Kissing Number
    """
    print(f"=== 计算 {dim} 维空间的的 Kissing Number (Lower Bound) ===")
    
    # 启发式搜索起点
    # 从 2*d 开始
    current_n = 2 * dim
    best_n = 0
    
    while True:
        print(f"尝试放入 {current_n} 个球...", end=" ", flush=True)
        # 维度越高，越容易陷入局部最优，需要更多尝试次数和步数
        attempts = 5 if dim <= 4 else 10
        steps = 2000 if dim <= 4 else 5000
        
        success = solve_kissing_configuration(current_n, dim, max_steps=steps, attempts=attempts)
        
        if success:
            print("成功 ✅")
            best_n = current_n
            current_n += 1
        else:
            print("失败 ❌ (达到当前算法极限)")
            break
            
    print(f"\n结果: 在 {dim} 维空间中，算法找到的最大 Kissing Number 为: {best_n}")
    return best_n

if __name__ == "__main__":
    # 从命令行获取输入，或默认运行演示
    try:
        input_dim = int(input("请输入维度 (例如 3): "))
    except:
        input_dim = 3
        
    find_max_kissing_number(input_dim)