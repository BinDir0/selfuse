import torch
import torch.nn as nn
from src.model.common.module_attr_mixin import ModuleAttrMixin

class BaseActionHead(ModuleAttrMixin):
    def forward(self, state, action_query):
        '''
        Args:
            state: dict, containing "wrist" and "hand"
            state = {
                "wrist": torch.Tensor, shape: [B, T, wrist_dim],
                "hand": torch.Tensor, shape: [B, T, hand_dim]
            } 
            proprioception of wrist and hand within the past T steps
            action_query: [B, H, D] action query tokens for action chunk H
        Returns:
            action: dict, containing "wrist" and "hand"
            action = {
                "wrist": torch.Tensor, shape: [B, H, wrist_dim],
                "hand": torch.Tensor, shape: [B, H, hand_dim]
            }
        '''
        pass
