import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import itertools
import time
from functools import wraps

import random
from utils import *

# 全局时间统计字典
time_stats = {}

def time_tracker(func):
    """装饰器：记录函数的调用次数和累计耗时"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if func_name not in time_stats:
            time_stats[func_name] = {'calls': 0, 'total_time': 0.0}
        time_stats[func_name]['calls'] += 1
        time_stats[func_name]['total_time'] += elapsed
        
        return result
    return wrapper

def print_time_stats():
    """打印所有函数的时间统计"""
    print("\n" + "="*60)
    print("函数耗时统计:")
    print("="*60)
    print(f"{'函数名':<25} {'调用次数':>10} {'总耗时(s)':>12} {'平均耗时(ms)':>14}")
    print("-"*60)
    
    # 按总耗时降序排序
    sorted_stats = sorted(time_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    for func_name, stats in sorted_stats:
        calls = stats['calls']
        total = stats['total_time']
        avg_ms = (total / calls * 1000) if calls > 0 else 0
        print(f"{func_name:<25} {calls:>10} {total:>12.4f} {avg_ms:>14.4f}")
    print("="*60 + "\n")

def reset_time_stats():
    """重置时间统计"""
    global time_stats
    time_stats = {}

args_game = dotdict({
    'augumentation': 10,
})

dim4_sol = torch.tensor([[1,1,0,0],[1,-1,0,0],[-1,1,0,0],[-1,-1,0,0],
                         [1,0,1,0],[1,0,-1,0],[-1,0,1,0],[-1,0,-1,0],
                         [1,0,0,1],[1,0,0,-1],[-1,0,0,1],[-1,0,0,-1],
                         [0,1,1,0],[0,1,-1,0],[0,-1,1,0],[0,-1,-1,0],
                         [0,1,0,1],[0,1,0,-1],[0,-1,0,1],[0,-1,0,-1],
                         [0,0,1,1],[0,0,1,-1],[0,0,-1,1],[0,0,-1,-1]])

dim5_sol = torch.tensor([[1,1,0,0,0],[1,-1,0,0,0],[-1,1,0,0,0],[-1,-1,0,0,0],
                         [1,0,1,0,0],[1,0,-1,0,0],[-1,0,1,0,0],[-1,0,-1,0,0],
                         [1,0,0,1,0],[1,0,0,-1,0],[-1,0,0,1,0],[-1,0,0,-1,0],
                         [1,0,0,0,1],[1,0,0,0,-1],[-1,0,0,0,1],[-1,0,0,0,-1],
                         [0,1,1,0,0],[0,1,-1,0,0],[0,-1,1,0,0],[0,-1,-1,0,0],
                         [0,1,0,1,0],[0,1,0,-1,0],[0,-1,0,1,0],[0,-1,0,-1,0],
                         [0,1,0,0,1],[0,1,0,0,-1],[0,-1,0,0,1],[0,-1,0,0,-1],
                         [0,0,1,1,0],[0,0,1,-1,0],[0,0,-1,1,0],[0,0,-1,-1,0],
                         [0,0,1,0,1],[0,0,1,0,-1],[0,0,-1,0,1],[0,0,-1,0,-1],
                         [0,0,0,1,1],[0,0,0,1,-1],[0,0,0,-1,1],[0,0,0,-1,-1]])

@time_tracker
def generate_inverse_mappings(d, order):
    inverse_mapping = [0] * d
    for i, val in enumerate(order):
        inverse_mapping[val] = i
    #print(inverse_mapping)
    return inverse_mapping

@time_tracker
def get_permuted_tensors(tensor1, tensor2):
    """Generate all possible tensors obtained by permuting the axes of A"""
    d = len(tensor1.shape)
    orders = list(itertools.permutations(range(d)))
    orders = orders[1:]     # delete "identity" permutation
    random.shuffle(orders)

    tensors = []
    
    # add origin data
    p1 = tensor1.reshape(-1).tolist()
    tensors.append((tensor2, p1))

    augu_size = min(args_game.augumentation, len(orders))
    for i in range(augu_size):
        permutation1 = tensor1.permute(orders[i]).reshape(-1).tolist()
        permutation2 = torch.index_select(tensor2, 1, torch.tensor(generate_inverse_mappings(d, orders[i])))
        tensors.append((permutation2, permutation1))
    
    return tensors



class Game():
    """
    The Kissing Number Game class.
    """
    def __init__(self, dim, boundary, upper_bound, print_result=1):
        '''
        Input:
            dim: Dimension for kissing number
            boundary: a positive int, we want to search in points with each dimension in [-boundary, boundary] (int)
            upper_bound: The upper bound for the kissing number problem
        '''
        self.dim, self.boundary, self.upper_bound, self.print_result = dim, boundary, upper_bound, print_result
        self.best = 0
        self.best_board = torch.zeros(size=(upper_bound, dim))

        r = range(-boundary, boundary + 1)
        all_actions_list = list(itertools.product(r, repeat=dim))
        self.all_actions = torch.tensor(all_actions_list, dtype=torch.float32)
        self.zero_action_idx = len(self.all_actions) // 2
        self.all_actions_sq_norms = torch.sum(self.all_actions**2, dim=1)
        

    @time_tracker
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        startBoard = torch.zeros(size = (self.upper_bound, self.dim))
        
        # Utilizing results in low-dimensional space
        if self.dim == 5:
            startBoard[:24, :4] = dim4_sol
        if self.dim == 6:
            startBoard[:40, :5] = dim5_sol
        
        return startBoard

    @time_tracker
    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.upper_bound, self.dim)

    @time_tracker
    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        actionSize = (2 * self.boundary + 1) ** self.dim
        # NOTE: for simplicity, we let (0,0,...,0) to be possible but illegal action.
        return actionSize

    @time_tracker
    def getNextState(self, board, action):
        """
        Input:
            board: current board
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        maxdim, _ = torch.max(torch.abs(board), dim=1)
        _, minplace = torch.min(maxdim, dim = 0)
        minplace = minplace.item()
        nextBoard = board.clone()
        nextBoard[minplace, :] = self.getAction(action)
        return nextBoard

    @time_tracker
    def getPointNum(self, board):
        maxdim, _ = torch.max(torch.abs(board), dim=1)
        _, minplace = torch.min(maxdim, dim=0)
        pointnum = minplace.item()
        return pointnum
    
    @time_tracker
    def getAction(self, x):
        return self.all_actions[x]

    @time_tracker
    def getValidMoves(self, board):
        """
        向量化实现的合法动作检查
        """
        pointnum = self.getPointNum(board)
        if pointnum == 0:
            v = torch.ones(len(self.all_actions))
            v[self.zero_action_idx] = 0
            return v

        # 获取当前棋盘上已有的向量 (pointnum, dim)
        placed_vectors = board[:pointnum, :]
        
        # 1. 矩阵乘法计算所有候选动作与已放向量的内积 (num_actions, pointnum)
        # inner_prods[i, j] 是第 i 个候选动作与第 j 个已放向量的点积
        # [num_actions, dim] @ [dim, pointnum] = [num_actions, pointnum]
        inner_prods = torch.mm(self.all_actions, placed_vectors.t()) 
        
        # 2. 获取已放向量的模长平方 (pointnum,)
        placed_sq_norms = torch.sum(placed_vectors**2, dim=1)
        
        # 3. 角度检查公式：4 * (v1·v2)^2 <= |v1|^2 * |v2|^2 或者 v1·v2 <= 0
        # 我们寻找“冲突”的动作：(v1·v2 > 0) 且 (4 * (v1·v2)^2 > |v1|^2 * |v2|^2)
        
        lhs = 4 * (inner_prods ** 2)

        # [num_actions, 1] * [1, pointnum] = [num_actions, pointnum]
        rhs = self.all_actions_sq_norms.unsqueeze(1) * placed_sq_norms.unsqueeze(0)
        
        # 判定冲突条件
        conflicts = (inner_prods > 0) & (lhs > rhs)
        
        # 只要与任何一个已放向量冲突，该动作就非法
        is_invalid = torch.any(conflicts, dim=1)
        
        # 生成返回向量
        valid_moves = (~is_invalid).float()
        valid_moves[self.zero_action_idx] = 0 # 排除全零向量
        
        return valid_moves

    @time_tracker
    def getGameEnded(self, board):
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended. 
            If the game ended, return the reward.
               
        """
        valid = self.getValidMoves(board)
        validnum = torch.sum(valid).item()
        if validnum == 0:
            pointnum = self.getPointNum(board)
            if pointnum > self.best:
                self.best = pointnum
                self.best_board = torch.clone(board)
                if self.print_result:
                    print("End of kissing instance, num of points:", pointnum, flush=True)
                    print(board, flush=True)
                    print(" ", flush=True)
            return pointnum
        else:
            return 0

    @time_tracker
    def getCanonicalForm(self, board, player):
        return board

    @time_tracker
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        point_num = self.getPointNum(board)
        if point_num <= self.dim:
            return [(board,pi)]
        new_shape = tuple([2 * self.boundary + 1] * self.dim)
        pi_tensor = torch.tensor(pi).view(*new_shape)
        return get_permuted_tensors(pi_tensor, board)

    @time_tracker
    def stringRepresentation(self, board):
        # --- 字符串转换极其慢，改用字节流 ---
        # 只转已经放了向量的部分，减少哈希负担
        pointnum = self.getPointNum(board)
        return board[:pointnum, :].numpy().tobytes()

