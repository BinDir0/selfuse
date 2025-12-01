import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from .base_retargeting import BaseRetargeting

class RetargetingHead(BaseRetargeting):
    """
    Implements the retargeting MLP described in the EgoVLA paper (Appendix 3.3).

    This model maps 3D fingertip positions (derived from predicted MANO parameters)
    to the specific joint angle commands for a bimanual robot's hands.
    """
    def __init__(self, 
                 shape_meta: Optional[Dict[str, Any]] = None,
                 model_config: Optional[DictConfig] = None):
        """
        Initializes the MLP architecture.

        Args:
            shape_meta (Optional[Dict[str, Any]]): Shape metadata containing input/output shape information and retargeting parameters.
            model_config (Optional[DictConfig]): Model configuration containing local weights path.
        """
        super().__init__()

        self.shape_meta = shape_meta
        self.model_config = model_config

        self.num_keypoints = self.model_config['num_keypoints']
        self.robot_dof = self.shape_meta['robot_dof']
        input_dim = self.num_keypoints * 3 
        output_dim = self.robot_dof

        self.hidden_sizes = model_config.get('hidden_sizes', [64, 128, 64])
        self.local_weights_path = model_config['local_weights_path']

        layers = []
        prev_dim = input_dim
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Load pretrained weights if specified
        if self.local_weights_path is not None:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """
        Load pretrained weights from the specified path.
        """
        try:
            checkpoint = torch.load(self.local_weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights to mlp
            self.mlp.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained weights: {self.local_weights_path}")
        except Exception as e:
            print(f"Warning: Unable to load pretrained weights {self.local_weights_path}: {e}")

    def forward(self, keypoints_pos: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            keypoints_pos: [B, E, N, 3] positions of hand keypoints in wrist frame
            N: number of keypoints per hand
        Returns:
            qpos: [B, E, rdof] qpos of the hand dofs
        '''
        B, E, N, _ = keypoints_pos.shape
        keypoints_pos = keypoints_pos.reshape(B * E, N * 3) # [B*E, N*3]
        qpos = self.mlp(keypoints_pos) # [B*E, rdof]
        return qpos.reshape(B, E, -1) # [B, E, rdof]
