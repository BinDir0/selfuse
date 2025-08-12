import torch
import torch.nn as nn
from egovla.model.action.base_action_head import BaseActionHead

class ActionHead(BaseActionHead):
    def __init__(self, shape_meta, model_config):
        super().__init__()
        self.shape_meta = shape_meta
        self.model_config = model_config

        feature_dim = model_config["hidden_size"]
        self.wrist_net = nn.Sequential(
            nn.Linear(shape_meta["obs"]["wrist"]["shape"][0] * shape_meta["obs"]["wrist"]["horizon"], feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
        
        self.hand_net = nn.Sequential(
            nn.Linear(shape_meta["obs"]["hand"]["shape"][0] * shape_meta["obs"]["hand"]["horizon"], feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
        
        self.pos_embed = nn.Parameter(torch.randn(
            1, 
            2 + shape_meta["action"]["horizon"], # 2 for wrist and hand
            feature_dim) * 0.02)

        # TODO: Use TransformerEncoder with flash attention
        self.action_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=model_config["num_heads"],
                dim_feedforward=model_config["intermediate_size"],
                dropout=model_config["dropout"],
                batch_first=True,
            ),
            num_layers=model_config["num_layers"],
        )

        self.action_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, shape_meta["action"]["shape"][0]),
        )
        

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
        B, T, D = state["wrist"].shape

        state_wrist = state["wrist"].view(B, -1)
        state_hand = state["hand"].view(B, -1)

        state_wrist = self.wrist_net(state_wrist)  # [B, feature_dim]
        state_hand = self.hand_net(state_hand)     # [B, feature_dim]

        # concat along sequence dimension, 2+H tokens
        # state_wrist: [B, feature_dim] -> [B, 1, feature_dim]
        # state_hand: [B, feature_dim] -> [B, 1, feature_dim]
        # action_query: [B, H, feature_dim]
        # final inputs: [B, 2+H, feature_dim]
        state_wrist = state_wrist.unsqueeze(1)  # [B, 1, feature_dim]
        state_hand = state_hand.unsqueeze(1)    # [B, 1, feature_dim]
        
        inputs = torch.cat([state_wrist, state_hand, action_query], dim=1)  # [B, 2+H, feature_dim]
        inputs = inputs + self.pos_embed
        outputs = self.action_head(inputs)[:, 2:, :]  # only take the output of action_query

        action = self.action_net(outputs)
        action = {
            "wrist": action[:, :, :self.shape_meta["obs"]["wrist"]["shape"][0]],
            "hand": action[:, :, self.shape_meta["obs"]["wrist"]["shape"][0]:],
        }
        return action


def test_action_head():
    # 设置测试参数
    batch_size = 2
    obs_horizon = 6
    action_horizon = 30
    hidden_size = 128
    
    shape_meta = {
        "obs": {
            "wrist": {"shape": [18], "horizon": obs_horizon},
            "hand": {"shape": [30], "horizon": obs_horizon},
        },
        "action": {"shape": [48], "horizon": action_horizon},  # 18 + 30 = 48
    }
    
    model_config = {
        "hidden_size": hidden_size, 
        "num_heads": 8, 
        "intermediate_size": 256, 
        "dropout": 0.1
    }
    
    # 创建模型实例
    action_head = ActionHead(shape_meta, model_config)
    
    # 创建测试输入
    state = {
        "wrist": torch.randn(batch_size, obs_horizon, 18),  # [B, T, wrist_dim]
        "hand": torch.randn(batch_size, obs_horizon, 30),   # [B, T, hand_dim]
    }
    action_query = torch.randn(batch_size, action_horizon, hidden_size)  # [B, H, D]
    
    # 前向传播
    action = action_head(state, action_query)
    
    # 验证输出
    print("=== ActionHead Test Results ===")
    print(f"Input state shapes:")
    print(f"  - wrist: {state['wrist'].shape}")
    print(f"  - hand: {state['hand'].shape}")
    print(f"Input action_query shape: {action_query.shape}")
    print(f"Output action shapes:")
    print(f"  - wrist: {action['wrist'].shape}")
    print(f"  - hand: {action['hand'].shape}")
    
    # 验证输出维度是否正确
    expected_wrist_shape = (batch_size, action_horizon, 18)
    expected_hand_shape = (batch_size, action_horizon, 30)
    
    assert action['wrist'].shape == expected_wrist_shape, f"Expected wrist shape {expected_wrist_shape}, got {action['wrist'].shape}"
    assert action['hand'].shape == expected_hand_shape, f"Expected hand shape {expected_hand_shape}, got {action['hand'].shape}"
    
    print("✓ All shape assertions passed!")
    print("✓ Test completed successfully!")
    
    return action

if __name__ == "__main__":
    test_action_head()
