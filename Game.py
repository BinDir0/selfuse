import torch
import numpy as np

class Game():
    def __init__(self, dim, boundary, upper_bound, print_result=1, device='cpu', cosine_set=None, seed_gram=None):
        self.device = torch.device('cpu') 
        self.dim = dim
        self.upper_bound = upper_bound
        self.print_result = print_result
        
        # 存储种子
        self.seed_gram = seed_gram 
        
        if cosine_set is None:
            c_set = [-1.0, -0.5, 0.0, 0.5]
        else:
            c_set = cosine_set
            
        self.cosine_set = torch.tensor(c_set, dtype=torch.float64, device=self.device)
        self.action_size = len(self.cosine_set)
        
        self.best = 0
        self.best_board = None

    def getInitBoard(self):
        board = torch.zeros((2, self.upper_bound, self.upper_bound), dtype=torch.float32, device=self.device)
        
        if self.seed_gram is not None:
            # === 使用 Simulation 提供的种子 ===
            # seed_gram 应该是一个 (k, k) 的矩阵，通常 k=dim
            k = self.seed_gram.shape[0]
            
            # 将种子填入 Gram 矩阵
            board[0, :k, :k] = self.seed_gram.float().to(self.device)
            
            # 设置 Meta Info
            board[1, 1, 0] = float(k) # current_m
            board[1, 1, 1] = 0.0      # is_dead
            board[1, 2, 0] = 0.0      # partial_len
            
        else:
            # Fallback: 从 1 个球开始
            board[0, 0, 0] = 1.0 
            board[1, 1, 0] = 1.0 
            
        return board

    def getBoardSize(self):
        return (self.upper_bound, self.upper_bound)

    def getActionSize(self):
        return self.action_size

    def getNextState(self, board, action):
        if board[1, 1, 1].item() > 0.5: return board 

        next_board = board.clone()
        current_m = int(next_board[1, 1, 0].item())
        partial_len = int(next_board[1, 2, 0].item())
        
        val = float(self.cosine_set[action].item())
        next_board[1, 0, partial_len] = val
        partial_len += 1
        next_board[1, 2, 0] = float(partial_len)
        
        required_len = current_m if current_m < self.dim else self.dim
        
        if partial_len == required_len:
            success = self._try_finalize_sphere(next_board, current_m)
            next_board[1, 0, :] = 0.0 
            next_board[1, 2, 0] = 0.0  
            
            if success:
                next_board[1, 1, 0] += 1.0
            else:
                next_board[1, 1, 1] = 1.0 
        
        return next_board

    def _try_finalize_sphere(self, board, m):
        gram = board[0, :m, :m].double()
        vec = board[1, 0, :].double()
        
        if m < self.dim:
            # m < dim: 构建基底阶段
            # vec 代表新向量与前 m 个向量的内积
            # 此时直接扩充 Gram 矩阵
            new_row = vec[:m]
            temp_gram = torch.zeros((m+1, m+1), device=self.device, dtype=torch.float64)
            temp_gram[:m, :m] = gram
            temp_gram[m, :m] = new_row
            temp_gram[:m, m] = new_row
            temp_gram[m, m] = 1.0
            
            try:
                # Cholesky 检查正定性
                temp_gram.diagonal().add_(1e-7)
                torch.linalg.cholesky(temp_gram)
                temp_gram.diagonal().sub_(1e-7) 
                board[0, :m+1, :m+1] = temp_gram.float()
                return True
            except RuntimeError:
                return False
        else:
            # m >= dim: 满秩阶段
            # vec[:dim] 代表新球与【基底球(前dim个)】的 Cosines (即 b)
            # 我们需要求 Coefficients (即 x)，满足 G * x = b
            
            basis_gram = gram[:self.dim, :self.dim]
            target_cosines = vec[:self.dim]
            
            try:
                # === 关键修复：解方程求系数 ===
                coeffs = torch.linalg.solve(basis_gram, target_cosines)
            except RuntimeError:
                return False # 奇异矩阵，无解
            
            # 1. 模长检查: ||x||^2 = x^T G x = x^T b
            # norm_sq = coeffs @ basis_gram @ coeffs = coeffs @ target_cosines
            norm_sq = torch.dot(coeffs, target_cosines)
            
            if abs(norm_sq - 1.0) > 1e-3: 
                return False
            
            # 2. 计算完整的新行 (与所有球的 Cosine)
            # Full_Cosines = Full_Gram_Cols * coeffs
            if m > self.dim:
                # 获取前 dim 列 (作为基底的投影)
                gram_cols = gram[:m, :self.dim] # (m, dim)
                full_row = torch.mv(gram_cols, coeffs)
            else:
                # 刚好是第 dim 个球，就是 b 本身
                full_row = target_cosines
            
            # === 全量白名单检查 ===
            # 这一步是为了防止数值误差积累出脏数据
            # 检查算出来的所有余弦值是否都在 cosine_set 里
            dists = torch.abs(full_row.unsqueeze(1) - self.cosine_set.unsqueeze(0))
            min_dists, _ = torch.min(dists, dim=1)
            
            if torch.any(min_dists > 2e-3): # 稍微放宽一点点容差，解方程会有误差
                return False
            # ==========================

            board[0, m, :m] = full_row.float()
            board[0, :m, m] = full_row.float()
            board[0, m, m] = 1.0
            return True

    def getValidMoves(self, board):
        if board[1, 1, 1].item() > 0.5:
            return torch.zeros(self.action_size, device=self.device)

        current_m = int(board[1, 1, 0].item())
        partial_len = int(board[1, 2, 0].item())
        required_len = current_m if current_m < self.dim else self.dim

        # === 修复后的逻辑 ===
        
        valids = torch.zeros(self.action_size, device=self.device)
        
        # 策略：除非是填该球的最后一步，否则全部放行 (Ones)
        # 这避免了我们在中间步骤用错误的数学公式误杀好人
        if partial_len < required_len - 1:
            return torch.ones(self.action_size, device=self.device)
            
        # === 只有在最后一步，才进行昂贵的、精确的几何检查 ===
        for i in range(self.action_size):
            # 这里的逻辑非常简单粗暴：试一下，能活就留，不能活就杀
            # 因为只剩下最后一步了，尝试次数 = action_size (约20次)，完全算得过来
            
            # 构建一个临时 board
            # 注意：这里不需要 clone 整个大矩阵，只需要模拟最后一步
            # 但为了代码复用 _try_finalize_sphere，我们还是 clone
            temp_board = board.clone()
            
            val = float(self.cosine_set[i])
            temp_board[1, 0, partial_len] = val
            
            # 调用那个已经修复了 "solve方程" 的正确检查函数
            if self._try_finalize_sphere(temp_board, current_m):
                valids[i] = 1.0
                
        return valids

    def getGameEnded(self, board):
        if board[1, 1, 1].item() > 0.5:
            return int(board[1, 1, 0].item())
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        current_m = int(board[1, 1, 0].item())
        is_dead = int(board[1, 1, 1].item())
        partial_len = int(board[1, 2, 0].item())
        gram_bytes = board[0, :current_m, :current_m].numpy().tobytes()
        partial_bytes = board[1, 0, :partial_len].numpy().tobytes() if partial_len > 0 else b''
        meta_bytes = np.array([current_m, is_dead, partial_len], dtype=np.int32).tobytes()
        return gram_bytes + partial_bytes + meta_bytes