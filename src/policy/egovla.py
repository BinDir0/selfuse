import torch
import torch.nn.functional as F
from typing import List
import inspect

from src.model.vlm.nvila import NVILA
from src.model.action.base_action_head import BaseActionHead
from src.model.retargeting.base_retargeting import BaseRetargeting
from src.model.common.normalizer import LinearNormalizer
from src.utils.geometry import rot_matrix_from_6drot
from src.utils.pytorch_util import dict_apply
from .base_policy import BasePolicy

class EgoVLA(BasePolicy):
    def __init__(
        self,
        shape_meta : dict, 
        vlm : NVILA, 
        action_head : BaseActionHead,
        loss_config : dict,
        retargeting_heads : List[BaseRetargeting] = None,
        start_ckpt_path = None
    ):
        super(EgoVLA, self).__init__()
        self.shape_meta = shape_meta
        self.vlm = vlm
        self.action_head = action_head

        self.retargeting_heads = retargeting_heads
        self.normalizer = LinearNormalizer()
        self.loss_config = loss_config

    # =========  inference  ============

    @torch.inference_mode()
    def predict_action(self, input):
        raise NotImplementedError("Not implemented")

    # =========  training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

        self.normalizer.eval()
    
        for param in self.normalizer.parameters():
            param.requires_grad = False

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
                "image": torch.Tensor, shape: [B, T, 3, H, W]
                "input_ids": torch.Tensor, shape: [B, L], 
                "attention_mask": torch.Tensor, shape: [B, L],
                "state/wrist": torch.Tensor, shape: [B, T, wrist_dim],
                "state/hand": torch.Tensor, shape: [B, T, hand_dim],  
                "action/wrist": torch.Tensor, shape: [B, H, wrist_dim], 
                    # we assume the translation is the first 6 dimensions of the wrist
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
                "wrist": batch["state/wrist"],
                "hand": self.normalizer['state/hand'](batch["state/hand"])
            }
            action = {
                "wrist": batch["action/wrist"],
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
                                    dim=1)
        wrist_rot_gt = torch.cat([rot_matrix_from_6drot(action["wrist"][:, :, 6:12]), 
                                  rot_matrix_from_6drot(action["wrist"][:, :, 12:18])], 
                                  dim=1)
        wrist_rot_loss = F.mse_loss(wrist_rot_pred, wrist_rot_gt)

        loss = self.loss_config["hand_loss_weight"] * hand_loss + \
               self.loss_config["wrist_trans_loss_weight"] * wrist_trans_loss + \
               self.loss_config["wrist_rot_loss_weight"] * wrist_rot_loss

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
