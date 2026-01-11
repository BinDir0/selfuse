import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from utils import *

args = dotdict({
    'lr': 1e-3,
    'dropout': 0.1,
    'epochs': 10,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'feature_dim': 256,
})

class TransformerEncoder(nn.Module):
    """Wrapper around PyTorch's TransformerEncoder."""
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
        
    def forward(self, x, mask=None):
        return self.encoder(x, src_key_padding_mask=mask)

class KissingGramNNet(nn.Module):
    def __init__(self, game):
        super(KissingGramNNet, self).__init__()
        self.dim = game.dim
        self.upper_bound = game.upper_bound
        self.action_size = game.getActionSize()
        self.feature_dim = args.feature_dim
        
        self.embedding = nn.Linear(2 * self.upper_bound, self.feature_dim)
        
        self.transformer = TransformerEncoder(
            num_layers=3, 
            num_heads=4, 
            hidden_size=self.feature_dim, 
            dropout=args.dropout
        )
        
        self.atte2pi = nn.Linear(self.feature_dim, self.action_size)
        self.atte2v = nn.Linear(self.feature_dim, 1)
        
        self.pi_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim), requires_grad=True)
        self.v_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim), requires_grad=True)
        nn.init.normal_(self.pi_token, std=1e-6)
        nn.init.normal_(self.v_token, std=1e-6)

    def forward(self, x):
        batch_size = x.shape[0]
        current_m = x[:, 1, 1, 0].long()
        seq_range = torch.arange(self.upper_bound, device=x.device).unsqueeze(0)
        padding_mask = seq_range >= current_m.unsqueeze(1)
        
        tokens = x.permute(0, 2, 3, 1).contiguous().view(batch_size, self.upper_bound, -1)
        h = self.embedding(tokens)
        
        h = torch.cat([
            self.pi_token.expand(batch_size, -1, -1),
            self.v_token.expand(batch_size, -1, -1),
            h
        ], dim=1)
        
        special_mask = torch.zeros((batch_size, 2), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([special_mask, padding_mask], dim=1)
        
        h = self.transformer(h, mask=full_mask)
        
        pi_out = h[:, 0, :]
        v_out = h[:, 1, :]
        
        pi_logits = self.atte2pi(pi_out)
        v_val = self.atte2v(v_out)
        v = F.relu(v_val)
        
        return pi_logits, v

class NeuralNet():
    def __init__(self, game):
        self.game = game
        self.nnet = KissingGramNNet(game)
        
        if args.cuda:
            self.nnet.cuda()

    # --- 新增：Proxy Methods (透传方法) ---
    # 让 Wrapper 表现得像 PyTorch Module，修复 AttributeError
    def cpu(self):
        self.nnet.cpu()
        return self

    def cuda(self):
        self.nnet.cuda()
        return self

    def to(self, device):
        self.nnet.to(device)
        return self

    def eval(self):
        self.nnet.eval()
        
    def parameters(self):
        return self.nnet.parameters()
        
    def state_dict(self):
        return self.nnet.state_dict()
        
    def load_state_dict(self, state_dict):
        self.nnet.load_state_dict(state_dict)

    # ------------------------------------

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        batch_size = min(args.batch_size, len(examples))
        if batch_size == 0: return
        batch_count = int(len(examples) / batch_size)
        
        total_iters = args.epochs * batch_count
        warm_iters = int(0.1 * total_iters) + 1
        lr_lambda = lambda step: 1 - (step - warm_iters)/(total_iters - warm_iters + 1e-8) if step >= warm_iters else step/warm_iters
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        self.nnet.train() # Set mode
        
        for epoch in range(args.epochs):
            t = tqdm(range(batch_count), desc=f'Epoch {epoch+1}')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                batch = [examples[i] for i in sample_ids]
                boards, pis, vs = list(zip(*batch))
                
                boards = torch.tensor(np.array(boards), dtype=torch.float32)
                target_pis = torch.tensor(np.array(pis), dtype=torch.float32)
                target_vs = torch.tensor(np.array(vs), dtype=torch.float32)

                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = F.cross_entropy(out_pi, target_pis)
                l_v = F.mse_loss(out_v.view(-1), target_vs.view(-1))
                total_loss = l_pi + l_v 

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_postfix(L_pi=l_pi.item(), L_v=l_v.item())

    def predict(self, board):
        self.nnet.eval()
        with torch.no_grad():
            if args.cuda and not board.is_cuda:
                board = board.cuda()
            
            # Add batch dimension if needed
            if board.dim() == 3:
                board = board.unsqueeze(0) 
            
            pi, v = self.nnet(board)
            
            # 搬回 CPU 给 MCTS
            p_numpy = F.softmax(pi, dim=1).cpu().numpy()[0]
            v_val = v.item()
            
            return p_numpy, v_val

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])