import logging
import warnings

import coloredlogs

from Coach import Coach
from Game import Game
from NeuralNet import NeuralNet as nn
from utils import *
import multiprocessing as mp
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import torch
torch.set_num_threads(1)

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 500,
    'numEps': 240,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 35,        #
    'updateThreshold': 0.55,     #
    'maxlenOfQueue': 500000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 800,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         #
    'cpuct1': 1.25,
    'cpuct2': 10000,
    'num_process': 120,           # Number of parallel processes for self-play.

    'dim': 6, 
    'boundary': 2, 
    'upper_bound': 79,          # NOTE: Need to be [strictly] greater than known upper bound.

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args.dim, args.boundary, args.upper_bound)

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
    mp.set_start_method('spawn')
    main()
