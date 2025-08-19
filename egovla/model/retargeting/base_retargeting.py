import torch

from egovla.model.common.module_attr_mixin import ModuleAttrMixin

class BaseRetargeting(ModuleAttrMixin):
    def forward(self, keypoints_pos: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            keypoints_pos: [B, N, 3] positions of hand keypoints in wrist frame
            N: number of keypoints per hand
        Returns:
            qpos: [B, rdof] qpos of the hand dofs
        '''
        pass
