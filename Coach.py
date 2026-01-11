import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from Game import Game

import multiprocessing
import copy
import wandb
from utils import *
from itertools import chain
import torch

# 强制使用 spawn 模式，这是 GPU 多进程的前提
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

log = logging.getLogger(__name__)


def executeEpisode_parallel(nnet, args, rank, cosine_set):
    """
    Worker 进程：自动分配 GPU 并执行搜索
    """
    # --- 自动分配 GPU ---
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_id = rank % gpu_count # 轮询分配
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    nnet.to(device)
    nnet.eval()

    trainExamples = []

    seed_gram = args.get('seed_gram', None)
    
    # 确保如果是 Tensor 且在 GPU 上，转回 CPU
    if seed_gram is not None and isinstance(seed_gram, torch.Tensor):
        seed_gram = seed_gram.cpu() 

    game = Game(args.dim, args.boundary, args.upper_bound, print_result=0, device='cpu', 
                cosine_set=cosine_set, seed_gram=seed_gram)
    
    board = game.getInitBoard()
    curPlayer = 1 
    episodeStep = 0
    mcts = MCTS(game, nnet, args, rank) 

    # --- 修复：局部跟踪本局的最佳成绩 ---
    episode_best_m = 1  # 初始球数为 1
    episode_best_board = board.clone()

    while True:
        episodeStep += 1
        
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < args.tempThreshold)

        # MCTS 搜索
        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        
        # 数据收集
        sym = game.getSymmetries(canonicalBoard, pi)
        for b, p in sym:
            b_numpy = b.cpu().numpy() if isinstance(b, torch.Tensor) else b
            trainExamples.append([b_numpy, curPlayer, p, None])

        action = np.random.choice(len(pi), p=pi)
        board = game.getNextState(board, action)

        # --- 修复：每一检查当前状态是否打破了记录 ---
        # 读取当前球数 (channel 1, row 1, col 0) 和 是否死亡 (channel 1, row 1, col 1)
        current_m = int(board[1, 1, 0].item())
        is_dead = int(board[1, 1, 1].item())

        # 如果当前状态是存活的，且球数更多，则更新最佳记录
        if is_dead < 0.5 and current_m > episode_best_m:
            episode_best_m = current_m
            episode_best_board = board.cpu().clone()

        r = game.getGameEnded(board)

        if r != 0:
            # Reward 归一化
            min_s = getattr(args, 'min_score', 0)
            max_s = getattr(args, 'max_score', args.upper_bound)
            
            # 注意：这里的 r 是最终球数。
            norm_r = 2.0 * (r - min_s) / (max_s - min_s + 1e-5) - 1.0
            norm_r = max(-1.0, min(1.0, norm_r))

            # 返回训练数据、本局最佳球数、本局最佳棋盘
            return [(x[0], x[2], norm_r) for x in trainExamples], episode_best_m, episode_best_board

def executeEpisode_parallel_pack(nnet, args, iter_num, rank, cosine_set):
    trainExamples = []
    best_num = 0
    # 初始化 CPU tensor
    best_board = torch.zeros(size=(2, args.upper_bound, args.upper_bound)) 
    
    for i in range(iter_num):
        tmpexample, tmpbest, tmpboard = executeEpisode_parallel(nnet, args, rank, cosine_set)
        trainExamples.extend(tmpexample)
        
        # 记录这一批次中最好的
        if tmpbest > best_num:
            best_num = tmpbest
            best_board = tmpboard.clone()
            
    return trainExamples, best_num, best_board
        

class Coach():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args, 0)
        self.trainExamplesHistory = [] 
        self.skipFirstSelfPlay = False 

    def learn(self):
        for i in tqdm(range(1, self.args.numIters + 1), desc='Training'):
            log.info(f'Starting Iter #{i} ...')
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            
            if not self.skipFirstSelfPlay or i > 1:
                iter_num = max(1, self.args.numEps // self.args.num_process)
                
                cpu_net = copy.deepcopy(self.nnet).cpu()
                cosine_set = self.game.cosine_set.cpu().tolist()
                
                with multiprocessing.Pool(processes=self.args.num_process) as pool:
                    results = pool.starmap(executeEpisode_parallel_pack, [(cpu_net, self.args, iter_num, rank, cosine_set) for rank in range(self.args.num_process)])

                results_trainexamples, results_bestnum, results_bestboard = zip(*results)
                train_examples = list(chain.from_iterable(results_trainexamples))
                
                # Update best result
                np_bestnum = np.array(results_bestnum)
                best_index = np.argmax(np_bestnum)
                global_best_num = np_bestnum[best_index]

                print(f"\n[Iter {i}] Best Kissing Number Found: {global_best_num}", flush=True)
                
                m = int(global_best_num)
                # 打印 Gram Matrix 的左上角部分
                if m > 0:
                    # best_board shape is (2, bound, bound), channel 0 is Gram
                    gram_matrix = results_bestboard[best_index][0]
                    print(gram_matrix[:min(m, 10), :min(m, 10)], flush=True)
                print(" ", flush=True)
                
                # 记录到 Game 类中以便保存 (尽管并行时 Game 是新建的，但这里是主进程)
                if global_best_num > self.game.best:
                    self.game.best = global_best_num
                    self.game.best_board = torch.clone(results_bestboard[best_index])
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best_ever.pth.tar')
                
                # Save checkpoint
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                
                try:
                    wandb.log({
                        "global/best_kissing_number": global_best_num,
                        "global/iteration": i
                    })
                except: pass

                iterationTrainExamples = deque(train_examples)

            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
            
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.train(trainExamples)
            
            if i % 10 == 0:
                self.saveTrainExamples(i)
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))

            log.info('NEW MODEL SAVED')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')
            self.skipFirstSelfPlay = True