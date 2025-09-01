import torch
import torch.nn as nn
def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )

class ActionHistoryEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, history_steps):
        super().__init__()
        self.history_steps = history_steps
        self.action_embedding = nn.Linear(action_dim, hidden_size // 2)
        self.temporal_encoder = nn.LSTM(
            hidden_size // 2, 
            hidden_size // 2, 
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_size // 2, hidden_size)
    
    def forward(self, action_history):
        # action_history: [B, history_steps, action_dim]
        embedded = self.action_embedding(action_history)
        lstm_out, _ = self.temporal_encoder(embedded)
        # 取最后一个时间步的输出
        final_hidden = lstm_out[:, -1, :]  # [B, hidden_size//2]
        return self.output_proj(final_hidden).unsqueeze(1)  # [B, 1, hidden_size]
