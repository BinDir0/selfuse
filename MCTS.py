import logging
import math
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import time
from Game import print_time_stats

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, rank):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.rank = rank # Process rank
        self.action_size = game.getActionSize()
        
        # 向量化存储：以 s 为键，值为 numpy 数组
        self.Qsa = {}  # stores Q values for all actions at state s, shape: (action_size,)
        self.Nsa = {}  # stores visit counts for all actions at state s, shape: (action_size,)
        self.Ns = defaultdict(lambda: 0)  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s (numpy array)

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        if self.rank == 0:
            for i in tqdm(range(self.args.numMCTSSims), desc=f'Rank {self.rank} is searching'):
                self.search(canonicalBoard)
        else: 
            for i in range(self.args.numMCTSSims):
                self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        
        # 向量化获取 counts
        counts = self.Nsa[s] if s in self.Nsa else np.zeros(self.action_size)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(self.action_size)
            probs[bestA] = 1
            return probs

        counts = np.power(counts, 1. / temp)
        counts_sum = np.sum(counts)
        probs = counts / counts_sum if counts_sum > 0 else counts
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current Board
        """

        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard)
        if self.Es[s] != 0:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard).cpu().numpy()
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # 初始化 Qsa 和 Nsa 向量 (numpy)
            self.Qsa[s] = np.zeros(self.action_size, dtype=np.float32)
            self.Nsa[s] = np.zeros(self.action_size, dtype=np.float32)
            return v

        valids = self.Vs[s]
        Qsa_s = self.Qsa[s]
        Nsa_s = self.Nsa[s]
        Ps_s = self.Ps[s]
        
        # 找出有效动作的索引
        valid_actions = np.where(valids > 0)[0]
        
        # 找出有效但未访问的动作，进行扩展
        unvisited_mask = Nsa_s[valid_actions] == 0
        unvisited_actions = valid_actions[unvisited_mask]
        
        for a in unvisited_actions:
            # NOTE: look one step ahead!
            tmp_next_s = self.game.getNextState(canonicalBoard, a)
            tmp_v = self.search(tmp_next_s)
            Qsa_s[a] = tmp_v
            Nsa_s[a] = 1

        # 只对有效动作计算 UCB
        # u = Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a)) * (cpuct1 + log2((1 + cpuct2 + N(s)) / cpuct2))
        Ns_val = self.Ns[s]
        
        Qsa_valid = Qsa_s[valid_actions]
        Nsa_valid = Nsa_s[valid_actions]
        Ps_valid = Ps_s[valid_actions]
        
        exploration_term = Ps_valid * math.sqrt(Ns_val) / (1 + Nsa_valid) * \
                          (self.args.cpuct1 + math.log2((1 + self.args.cpuct2 + Ns_val) / self.args.cpuct2))
        
        ucb_valid = Qsa_valid + exploration_term
        
        # 选择最佳动作（在有效动作中）
        best_idx = np.argmax(ucb_valid)
        a = valid_actions[best_idx]
        
        next_s = self.game.getNextState(canonicalBoard, a)
        v = self.search(next_s)

        # 更新 Q 值和访问计数
        if Nsa_s[a] > 0:
            Qsa_s[a] = (Nsa_s[a] * Qsa_s[a] + v) / (Nsa_s[a] + 1)
            Nsa_s[a] += 1
        else:
            Qsa_s[a] = v
            Nsa_s[a] = 1

        self.Ns[s] += 1
        
        return v
