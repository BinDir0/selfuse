import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import inspect

from egovla.model.vlm.nvila import NVILA
from egovla.model.action.base_action_head import BaseActionHead
from egovla.model.action.retargeting import RetargetingHead
from egovla.model.common.normalizer import LinearNormalizer
from egovla.policy.base_policy import BasePolicy

from egovla.utils.transformation import rot_matrix_from_6drot

class EgoVLA(BasePolicy):
    def __init__(
        self,
        shape_meta : dict, 
        vlm : NVILA, 
        action_head : BaseActionHead,
        retargeting_head : RetargetingHead,
        loss_config : dict,
        start_ckpt_path = None
    ):
        super(EgoVLA, self).__init__()
        self.shape_meta = shape_meta
        self.vlm = vlm
        self.action_head = action_head
        self.retargeting_head = retargeting_head
        self.normalizer = LinearNormalizer()
        self.loss_config = loss_config
    
    # =========  inference  ============
    # TODO: add a method to predict the action
    # with retargeting to robot's body
    def predict_action(self, inputs):
        raise NotImplementedError("Not implemented")

    # =========  training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # TODO: add a method to get the optimizer only for the retargeting head
    # add modules_to_train to params
    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_params_no_grad = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"num parameters that require grad: {num_params:,}")
        print(f"num parameters that do not require grad: {num_params_no_grad:,}")
        assert num_params_no_grad == 0, "There are parameters that do not require grad"
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, fused=fused_available, weight_decay=0.0)
        return optimizer

    def compute_loss(self, batch):
        '''
        Args:
            inputs: dict, containing "image", "input_ids", "attention_mask", "state/wrist", "state/hand", "action/wrist", "action/hand"
            inputs = {
                "image": torch.Tensor, shape: [B, T, H, W, 3]
                "input_ids": torch.Tensor, shape: [B, L], 
                "attention_mask": torch.Tensor, shape: [B, L],
                "state/wrist": torch.Tensor, shape: [B, T, wrist_dim],
                "state/hand": torch.Tensor, shape: [B, T, hand_dim],  
                "action/wrist": torch.Tensor, shape: [B, H, wrist_dim], # we assume the translation is the first 6 dimensions of the wrist
                "action/hand": torch.Tensor, shape: [B, H, hand_dim]
            }
        Returns:
            loss: torch.Tensor, shape: []
        '''
        image = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        state = {
            "wrist": self.normalizer['state/wrist'](batch["state/wrist"]),
            "hand": self.normalizer['state/hand'](batch["state/hand"])
        }
        action = {
            "wrist": self.normalizer['action/wrist'](batch["action/wrist"]),
            "hand": self.normalizer['action/hand'](batch["action/hand"])
        }

        action_query = self.vlm(image, input_ids, attention_mask)
        action_pred = self.action_head(state, action_query)


        hand_loss = F.mse_loss(action_pred["hand"], action["hand"])
        wrist_trans_loss = F.mse_loss(action_pred["wrist"][:, :, :6], action["wrist"][:, :, :6])
        wrist_rot_pred = torch.cat([rot_matrix_from_6drot(action_pred["wrist"][:, :, 6:12]), 
                                    rot_matrix_from_6drot(action_pred["wrist"][:, :, 12:18])], 
                                    dim=-1)
        wrist_rot_gt = torch.cat([rot_matrix_from_6drot(action["wrist"][:, :, 6:12]), 
                                  rot_matrix_from_6drot(action["wrist"][:, :, 12:18])], 
                                  dim=-1)
        wrist_rot_loss = F.mse_loss(wrist_rot_pred, wrist_rot_gt)

        loss = self.loss_config["hand_loss_weight"] * hand_loss + \
               self.loss_config["wrist_trans_loss_weight"] * wrist_trans_loss + \
               self.loss_config["wrist_rot_loss_weight"] * wrist_rot_loss

        return loss

    # TODO: add a method to compute the loss for the retargeting head
    def compute_retargeting_loss(self, batch):
        raise NotImplementedError("Not implemented")

    def forward(self, batch):
        return self.compute_loss(batch)
