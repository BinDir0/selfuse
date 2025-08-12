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

        self.normalizer.eval()
    
        for param in self.normalizer.parameters():
            param.requires_grad = False

    # TODO: add a method to get the optimizer only for the retargeting head
    # add modules_to_train to params
    def get_optimizer(
            self,
            lr: float,
            weight_decay: float,
            betas: Tuple[float, float],
        ) -> torch.optim.Optimizer:

        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        print(f"Fused AdamW available: {fused_available}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas, fused=fused_available
        )
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
        with torch.no_grad():
            state = {
                "wrist": self.normalizer['state/wrist'](batch["state/wrist"]),
                "hand": self.normalizer['state/hand'](batch["state/hand"])
            }
            action = {
                "wrist": self.normalizer['action/wrist'](batch["action/wrist"]),
                "hand": self.normalizer['action/hand'](batch["action/hand"])
            }
            state = {k: v.detach() for k, v in state.items() }
            action = {k: v.detach() for k, v in action.items() }

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
