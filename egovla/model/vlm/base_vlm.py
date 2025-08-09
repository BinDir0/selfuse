import torch
import torch.nn as nn
from egovla.model.common.module_attr_mixin import ModuleAttrMixin

class BaseVLM(ModuleAttrMixin):
    def forward(self, images, instruction):
        '''
        Args:
            images: torch.Tensor, shape: [B, T, 3, H, W]
            instruction: str, shape: [B]
        Returns:
            action_query: [B, H, D] action query tokens for action chunk H
        '''
        pass
