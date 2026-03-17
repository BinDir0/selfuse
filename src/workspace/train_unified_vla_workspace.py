from __future__ import annotations

import torch

from src.workspace.train_legendvla_deepspeed_workspace import TrainLegendVLAWorkspace


class TrainUnifiedVLAWorkspace(TrainLegendVLAWorkspace):
    def preprocess_batch(self, batch, split_mask: bool = False, sample_fm_time: bool = True):
        del split_mask
        input_ids = batch["input_ids"]
        inputs = {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"].to(self.dtype),
            "image_grid_thw": batch["image_grid_thw"],
            "mm_token_type_ids": batch["mm_token_type_ids"],
            "states": batch["states"].to(self.dtype),
            "answer_start_idx": batch["answer_start_idx"],
            "is_vla_data": batch["is_vla_data"],
            "n_states": batch["n_states"],
            "n_actions": batch["n_actions"],
        }
        if self.objective_func != "train_ar":
            inputs["actions"] = batch["actions"].to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"]
        if self.objective_func != "train_flow":
            inputs["labels"] = batch["labels"]
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(len(input_ids)).to(input_ids.device).to(self.dtype)
        return inputs

    def evaluation(self, accelerator, dataloader, step_log):
        del accelerator, dataloader, step_log
        return
