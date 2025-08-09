import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import DictConfig
from typing import Optional, Dict, Any

class RetargetingHead(nn.Module):
    """
    Implements the retargeting MLP described in the EgoVLA paper (Appendix 3.3).

    This model maps 3D fingertip positions (derived from predicted MANO parameters)
    to the specific joint angle commands for a bimanual robot's hands.
    """
    def __init__(self, 
                 shape_meta: Optional[Dict[str, Any]] = None,
                 model_config: Optional[DictConfig] = None,
                 num_keypoints_per_hand: int = 5,
                 robot_per_hand_dof: int = 6):
        """
        Initializes the MLP architecture.

        Args:
            shape_meta (Optional[Dict[str, Any]]): Shape metadata containing input/output shape information and retargeting parameters.
            model_config (Optional[DictConfig]): Model configuration containing local weights path.
            num_keypoints_per_hand (int): Number of keypoints used per hand (e.g., 5 for fingertips).
            robot_per_hand_dof (int): Degrees of Freedom for a single robot hand.
                                  The paper uses Inspire hands with 12 DoFs.
        """
        super().__init__()

        self.shape_meta = shape_meta
        self.model_config = model_config

        self.num_keypoints_per_hand = num_keypoints_per_hand
        self.robot_per_hand_dof = robot_per_hand_dof
        self.hidden_sizes = model_config.get('hidden_sizes', [64, 128, 64])

        self.local_weights_path = model_config['local_weights_path']

        # Input dimension: 2 hands * 5 fingertips/hand * 3 coords (x,y,z) = 30
        input_dim = 2 * self.num_keypoints_per_hand * 3 
        # Output dimension: 2 hands * 6 DoFs/hand = 12
        output_dim = 2 * self.robot_per_hand_dof
        
        # Construct the four-layer MLP using hidden_sizes from config
        layers = []
        prev_dim = input_dim
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Load pretrained weights if specified
        if self.local_weights_path is not None:
            self._load_pretrained_weights()
        
        print("Initialized RetargetingHead:")
        print(f"  - Input Dimension (Fingertip Positions): {input_dim}")
        print(f"  - Output Dimension (Robot Joint Commands): {output_dim}")
        print(f"  - Hidden Sizes: {self.hidden_sizes}")
        print(f"  - Architecture: {input_dim} -> {' -> '.join(map(str, self.hidden_sizes))} -> {output_dim}")
        print(f"  - num_keypoints_per_hand={self.num_keypoints_per_hand}, robot_per_hand_dof={self.robot_per_hand_dof}")
        print(f"  - Weights path: {self.local_weights_path}")

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

    def forward(self, fingertip_positions: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the MLP.

        Args:
            fingertip_positions (torch.Tensor): A batch of 3D fingertip positions for both hands.
                                                Shape: [Batch, H, Input_Dim] (e.g., [B, H, 30])

        Returns:
            torch.Tensor: The predicted robot hand joint commands.
                          Shape: [Batch, H, Output_Dim] (e.g., [B, H, 12])
        """
        return self.mlp(fingertip_positions)
