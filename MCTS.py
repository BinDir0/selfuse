import logging
import math
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch

# 不需要再设置递归深度了
# import sys
# sys.setrecursionlimit(50000) 

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    def __init__(self, game, nnet, args, rank):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.rank = rank 
        self.action_size = game.getActionSize()
        
        self.Qsa = {}  # Q values for s,a
        self.Nsa = {}  # visit counts for s,a
        self.Ns = {}   # visit counts for board s
        self.Ps = {}   # initial policy
        self.Es = {}   # game ended status
        self.Vs = {}   # valid moves

    def getActionProb(self, canonicalBoard, temp=1):
        """
        执行模拟并返回概率
        """
        if self.rank == 0:
            iterator = tqdm(range(self.args.numMCTSSims), desc=f'Rank {self.rank} MCTS', leave=False)
        else:
            iterator = range(self.args.numMCTSSims)

        for _ in iterator:
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        
        # CPU numpy array
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(self.action_size)])
        counts_sum = counts.sum()
        if counts_sum == 0:
            # 如果 MCTS 没有进行任何有效访问（可能全是死局，或者模拟次数不够）
            # 我们尝试直接基于“合法动作”返回均匀分布
            valids = self.game.getValidMoves(canonicalBoard)
            if isinstance(valids, torch.Tensor):
                valids = valids.cpu().numpy()
            
            if valids.sum() > 0:
                probs = valids / valids.sum()
            else:
                # 绝境：没有任何合法动作，均匀乱猜（虽然也没用）
                probs = np.ones(self.action_size) / self.action_size
            
            return probs

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(self.action_size)
            probs[bestA] = 1
            return probs

        counts = counts ** (1. / temp)
        counts_sum = counts.sum()
        probs = counts / counts_sum
        return probs

    def search(self, canonicalBoard):
        """
        迭代式 MCTS 搜索：使用循环代替递归
        """
        # 1. Selection (下潜)
        # ------------------------------------------
        curr_board = canonicalBoard
        path = [] # 记录路径 [(s, a), (s, a), ...]
        
        # 保护机制：循环检测和深度限制
        # 改进：使用路径列表而不是集合，只检测真正的循环（同一条路径中的重复）
        visited_states_in_path = []  # 记录当前路径中访问过的状态序列
        max_depth = getattr(self.args, 'maxMCTSDepth', 1000)  # 最大深度限制
        depth = 0
        
        v = 0
        
        while True:
            # 保护机制：深度限制（提前检查，避免不必要的 stringRepresentation）
            if depth >= max_depth:
                log.warning(f"MCTS search reached max depth {max_depth}, terminating search")
                # 将当前状态标记为死局，返回负值
                s = self.game.stringRepresentation(curr_board)
                if s not in self.Es:
                    self.Es[s] = -1  # 标记为失败
                v = -1
                break
            
            # 性能优化：只在需要时调用 stringRepresentation（这是性能瓶颈）
            s = self.game.stringRepresentation(curr_board)
            
            # 1.1 先检查终止状态（必须在循环检测之前）
            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(curr_board)
            
            if self.Es[s] != 0:
                # 游戏结束，直接返回，不需要检查循环
                v = self.Es[s]
                break # 停止下潜
            
            # 保护机制：真正的循环检测（只检测同一条路径中的重复）
            # 检查当前状态是否在当前路径中出现过（真正的循环）
            # 只在深度 > 5 时检测，因为浅层重复可能是正常的搜索行为
            if depth > 5:
                # 检查最近访问的若干状态（避免检查整个路径，提高性能）
                # 只检查最近 10 个状态，因为真正的循环通常发生在最近的状态中
                recent_states = visited_states_in_path[-10:] if len(visited_states_in_path) > 10 else visited_states_in_path
                if s in recent_states:
                    log.warning(f"MCTS detected true cycle at state (depth={depth}), terminating search")
                    # 检测到真正的循环，标记为死局
                    if s not in self.Es:
                        self.Es[s] = -1
                    v = -1
                    break
            
            # 记录当前状态到路径中
            visited_states_in_path.append(s)

            # 1.2 检查是否为叶子节点 (未扩展)
            if s not in self.Ps:
                # Expand (扩展)
                self.Ps[s], v = self.nnet.predict(curr_board)
                
                valids = self.game.getValidMoves(curr_board)
                if isinstance(valids, torch.Tensor):
                    valids = valids.cpu().numpy()
                
                # --- 显式检查是否有合法动作 ---
                sum_valids = np.sum(valids)
                if sum_valids == 0:
                    # 如果所有动作都不合法，说明这是个死局节点
                    # 不要打印 Warning 了，因为在几何 Packing 问题里这太常见了，会刷屏
                    # log.warning(f"MCTS: No valid moves at leaf state...") 
                    
                    self.Es[s] = -1.0  # 标记为死局/失败
                    v = -1.0           # 此时的价值是负的
                    
                    # 不需要存储 Ps 或 Vs，直接跳出循环开始回溯
                    break 
                # ----------------------------------------

                self.Ps[s] = self.Ps[s] * valids
                sum_Ps_s = np.sum(self.Ps[s])
                
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s
                else:
                    # 只有当 valids 不全为 0，但网络预测的概率全被 mask 掉时进入这里
                    # 此时回退到均匀分布
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s]) # 这里现在安全了，因为 sum_valids > 0

                self.Vs[s] = valids
                self.Ns[s] = 0
                
                # 叶子节点评估完毕，开始回溯
                break

            # 1.3 UCB 选择 (Selection)
            # 如果不是叶子也不是终局，继续往下走
            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1

            # 向量化 UCB 计算 (比循环快)
            # 提取该状态下的所有 Q 和 N
            # 注意：如果状态 s 第一次访问 Ns[s] 为 0，公式依然成立
            
            # 为了速度，这里我们只遍历 valid actions
            # 或者更简单的：如果 action_size 不大 (比如 40-80)，直接循环其实也不慢
            # 这里保持原逻辑结构
            
            nb_visits = self.Ns[s]
            sqrt_nb_visits = math.sqrt(nb_visits + EPS)
            
            for a in range(self.action_size):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        q = self.Qsa[(s, a)]
                        n = self.Nsa[(s, a)]
                        u = q + self.args.cpuct1 * self.Ps[s][a] * sqrt_nb_visits / (1 + n)
                    else:
                        u = self.args.cpuct1 * self.Ps[s][a] * sqrt_nb_visits # Q=0, N=0
                        
                    if u > cur_best:
                        cur_best = u
                        best_act = a
            
            # 保护机制：检查是否找到合法动作
            if best_act == -1:
                # 没有找到合法动作，标记为死局
                log.warning(f"MCTS: No valid action found at state (depth={depth}), marking as dead end")
                self.Es[s] = -1
                v = -1
                break
            
            a = best_act
            
            # 记录路径，用于回溯
            path.append((s, a))
            
            # 执行一步 (GPU)
            curr_board = self.game.getNextState(curr_board, a)
            depth += 1
            
            # 继续循环...

        # 2. Backpropagation (回溯)
        # ------------------------------------------
        # 沿着 path 反向更新 Q 和 N
        # v 是叶子节点的价值 (或终局分数)
        
        for s, a in reversed(path):
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1
            
            self.Ns[s] += 1