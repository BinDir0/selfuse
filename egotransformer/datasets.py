from __future__ import annotations

import torch
from torch.utils.data import Dataset

# __getitem__:
#   video (3,T,H,W), existence (T,2), mano_*: trans(T,3), root(T,3), hand_pose(T,45), betas(T,10)
# collate_hand_batch:
#   video (B,T,3,H,W), existence (B,T,2), mano_* 各键 (B,T,d)


class DummyVideoHandDataset(Dataset):
    def __init__(
        self,
        *,
        num_samples: int,
        seq_len: int,
        image_size: int,
        mano_dims: dict[str, int],
        pos_rate: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.image_size = image_size
        self.mano_dims = mano_dims
        self.pos_rate = pos_rate

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        t = self.seq_len
        h = w = self.image_size
        video = torch.randn(3, t, h, w)

        exists = torch.zeros(t, 2)
        for hidx in range(2):
            exists[:, hidx] = (torch.rand(t) < self.pos_rate).float()

        def random_mano() -> dict[str, torch.Tensor]:
            return {
                "trans": torch.randn(t, self.mano_dims["trans"]) * 0.1,
                "root_orient": torch.randn(t, self.mano_dims["root"]) * 0.3,
                "hand_pose": torch.randn(t, self.mano_dims["pose"]) * 0.3,
                "betas": torch.randn(t, self.mano_dims["betas"]) * 0.05,
            }

        return {
            "video": video,
            "existence": exists,
            "mano_left": random_mano(),
            "mano_right": random_mano(),
        }


def collate_hand_batch(samples: list[dict]) -> dict:
    videos = torch.stack([s["video"] for s in samples], dim=0).float()
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()

    existence = torch.stack([s["existence"] for s in samples], dim=0).float()

    def stack_mano(prefix: str) -> dict[str, torch.Tensor]:
        return {
            "trans": torch.stack([s[prefix]["trans"] for s in samples], dim=0).float(),
            "root_orient": torch.stack([s[prefix]["root_orient"] for s in samples], dim=0).float(),
            "hand_pose": torch.stack([s[prefix]["hand_pose"] for s in samples], dim=0).float(),
            "betas": torch.stack([s[prefix]["betas"] for s in samples], dim=0).float(),
        }

    return {
        "video": videos,
        "existence": existence,
        "mano_left": stack_mano("mano_left"),
        "mano_right": stack_mano("mano_right"),
    }
