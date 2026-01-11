import logging
import warnings
import coloredlogs
from Coach import Coach
from Game import Game
from NeuralNet import NeuralNet as nn
from utils import *
import multiprocessing as mp
import os
import wandb  # <--- 新增
from Simulation import get_cosine_set_for_dim 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["WANDB_ENTITY"] = "bindir0-peking-university"

import numpy as np
import torch
torch.set_num_threads(1)

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 500,
    'numEps': 100,              # 每轮迭代自我对弈的次数
    'tempThreshold': 500,       # 温度阈值。注意：现在是微步模式，加一个球需 dim 步。200步大约是20-25个球。
    'updateThreshold': 0.55,    # (单人模式下此参数暂时无用，但保留)
    'maxlenOfQueue': 200000,    # 经验回放池大小
    'numMCTSSims': 600,         # MCTS 模拟次数。Gram 树较深，太小可能搜不到好结果。
    'maxMCTSDepth': 1000,       # MCTS 搜索最大深度限制，防止无限循环
    'arenaCompare': 40,         # (单人模式下无用)
    'cpuct1': 2.0,              # UCB 参数
    'cpuct2': 10000,            # UCB 参数
    'max_candidates': 100,      # Action Space Size Limit
    
    'num_process': 64,          # 并行进程数。GPU spawn 模式下显存开销大，切勿设置过大(如120)，否则OOM。
    
    'dim': 6,                   # 目标维度 (例如 E8 晶格)
    'boundary': 2,              # 边界大小。
    'upper_bound': 75,          # 矩阵最大尺寸。8维已知最优 240，设大一点做 buffer。
    
    # 新增：Reward 归一化参数 (必须与 upper_bound 匹配)
    'min_score': 0,
    'max_score': 75,            # 归一化分母：Score / 250 -> [0, 1] -> [-1, 1]

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    
    'cosine_set': None,
})

def main():
    # --- 新增: 初始化 wandb ---
    wandb.init(
        project="KissingNumber-Gram", 
        name=f"Dim{args.dim}-N{args.num_process}-MCTS{args.numMCTSSims}",
        config=args
    )

    log.info('Loading %s...', Game.__name__)
    discovered_cosines, best_seed_gram = get_cosine_set_for_dim(args.dim)
    args.cosine_set = discovered_cosines
    args.seed_gram = best_seed_gram # 存入 args，方便传给 Coach
    
    log.info(f'Final Cosine Set: {discovered_cosines}')
    if best_seed_gram is not None:
        log.info(f'Using Simulation Seed with size: {best_seed_gram.shape}')
    else:
        log.warning('No valid seed found, starting from scratch.')

    # 2. 初始化 Game (主进程)
    g = Game(args.dim, args.boundary, args.upper_bound, cosine_set=discovered_cosines, seed_gram=best_seed_gram)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
