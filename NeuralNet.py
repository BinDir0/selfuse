import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import time

import numpy as np
from tqdm import tqdm

from utils import *

import torch
import torch.optim as optim

args = dotdict({
    'lr': 1e-3,
    'dropout': 0.1,
    'epochs': 10,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class TransformerEncoder(nn.Module):
    """Wrapper around PyTorch's TransformerEncoder to maintain the same interface."""
    def __init__(self, num_layers, num_heads, hidden_size, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask):
        # mask: (batch_size, seq_len), True means masked position
        # PyTorch's src_key_padding_mask uses the same convention
        return self.encoder(x, src_key_padding_mask=mask)

class KissingNNet(nn.Module):
    def __init__(self, dim, boundary, upper_bound):
        super(KissingNNet, self).__init__()
        self.dim, self.boundary, self.upper_bound = dim, boundary, upper_bound
        self.feature_dim = 128
        self.beforeattention = nn.Linear(dim, self.feature_dim)
        self.transformer = TransformerEncoder(num_layers = 2, num_heads = 4, hidden_size = self.feature_dim, dropout = 0.0)
        self.atte2pi = nn.Linear(self.feature_dim, (2 * boundary + 1) ** dim)
        self.atte2v = nn.Linear(self.feature_dim, 1)
        
        self.pi_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim), requires_grad=True)
        self.v_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim), requires_grad=True)
        self.num_special_token = 2
        nn.init.normal_(self.pi_token, std=1e-6)
        nn.init.normal_(self.v_token, std=1e-6)


    def forward(self, x):
        x = x.view(-1, self.upper_bound * self.dim)
        batch_size = x.shape[0]
        balls = x.view(-1, self.upper_bound, self.dim)

        # Prepare mask and input, size: batch_size * upper_bound
        balls_norm = torch.norm(balls, dim=2)
        mask = balls_norm == 0
        num_balls = ( 1 * ~mask ).sum(dim=1)

        # Prepare input, size: [batch_size, num_balls.max(), feature_dim]
        r = x.view(batch_size, self.upper_bound, self.dim)
        r = r[:, :num_balls.max(), :]
        mask = mask[:, :num_balls.max()]

        # feed into networks
        r = self.beforeattention(r)

        # insert special tokens
        mask = torch.concat(
            [torch.zeros((batch_size, self.num_special_token), dtype=torch.bool, device=x.device),
            mask],
            dim = 1
        )
        r = torch.concat(
            [self.pi_token.repeat(batch_size, 1, 1),
            self.v_token.repeat(batch_size, 1, 1),
            r],
            dim = 1
        )

        r = self.transformer(r, mask)

        pi = self.atte2pi(r[:, 0, :]) # pi_token
        v = self.atte2v(r[:, 1, :]) # v_token
        v = num_balls[:, None] + F.relu(v)

        # NOTE: Now we only return raw logits of the policy distribution!
        return pi, v


class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define own neural
    network, subclass this class and implement the functions below.
    """

    def __init__(self, game):
        self.game = game
        self.nnet = KissingNNet(game.dim, game.boundary, game.upper_bound)

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        if args.cuda: 
            self.nnet = self.nnet.to(torch.device('cuda'))
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        batch_size = min(args.batch_size, len(examples))
        batch_count = int(len(examples) / batch_size)
        total_iters = args.epochs * batch_count

        warm_iters = int(0.1*total_iters)
        lr_lambda = lambda iter_id: 1-(iter_id-warm_iters)/(total_iters-warm_iters) if iter_id >= warm_iters \
                            else iter_id/warm_iters
        scheduler = torch.optim.lr_scheduler.LambdaLR( \
            optimizer, \
            lr_lambda=lr_lambda, \
        )

        for epoch in range(args.epochs):
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(tuple(map(lambda tensor: tensor.numpy(), boards))).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = 10 * l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

        self.nnet = self.nnet.to(torch.device('cpu'))

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # timing
        start = time.time()

        # preparing input
        board = board.float()
        board = board.view(1, self.game.upper_bound, self.game.dim)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return F.softmax(pi, dim=1).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]
    
    def loss_pi(self, targets, outputs):
        # targets are distributions
        # outputs are raw logits
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets)

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint['state_dict'])
