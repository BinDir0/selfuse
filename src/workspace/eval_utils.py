'''
Evaluation and checkpoint utilities for LegendVLA training workspace.

Extracted from train_legendvla_workspace.py for modularity.
'''

import os
import numpy as np
import torch
from contextlib import contextmanager
import pathlib

from src.utils.metric import get_action_accuracy


def clear_attn_weights(model):
    if hasattr(model, "joint_model") and hasattr(model.joint_model, "attn_weights"):
        model.joint_model.attn_weights = [None] * model.joint_model.num_hidden_layers


def _unwrap_model(workspace):
    if hasattr(workspace.model, 'module'):
        return workspace.model.module
    return workspace.model


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


@contextmanager
def eval_with_averaged_model(accelerator, model, averaged_model):
    """
    A context manager to temporarily load averaged weights into the main model during evaluation.
    """
    if averaged_model.model_avg is not None:
        unwrapped_model = accelerator.unwrap_model(model)

        # Use .clone() to avoid affecting the original dictionary
        # Move to CPU to avoid GPU memory issues
        device = next(iter(unwrapped_model.parameters())).device
        original_state_dict = {k: v.clone().to('cpu') for k, v in unwrapped_model.state_dict().items()}

        averaged_state_dict = averaged_model.averaged_model_state_dict()
        unwrapped_model.load_state_dict(averaged_state_dict)
    model.eval()

    try:
        yield
    finally:
        if averaged_model.model_avg is not None:
            unwrapped_model.load_state_dict(original_state_dict)
            unwrapped_model.to(device)
        model.train()


def save_checkpoint_accelerator(workspace, accelerator, path=None, tag='latest'):
    if path is None:
        path = pathlib.Path(workspace.output_dir).joinpath('checkpoints', f'{tag}')
    else:
        path = pathlib.Path(path)
    path.parent.mkdir(parents=False, exist_ok=True)
    workspace.training_state.update_step = workspace.update_step
    workspace.training_state.global_step = workspace.global_step
    workspace.training_state.epoch = workspace.epoch
    accelerator.save_state(path)


def save_topk_ckpt(workspace, accelerator, topk_manager, step_log):
    # Need to update_bn when the model contains batch norm layers !!!
    if workspace.cfg.checkpoint.save_last_ckpt:
        save_checkpoint_accelerator(workspace, accelerator)

    # sanitize metric names
    metric_dict = dict()
    for key, value in step_log.items():
        new_key = key.replace('/', '_')
        metric_dict[new_key] = value

    # We can't copy the last checkpoint here
    # since save_checkpoint uses threads.
    # therefore at this point the file might have been empty!
    topk_ckpt_path = topk_manager.get_ckpt_path(accelerator, metric_dict)

    if topk_ckpt_path is not None:
        save_checkpoint_accelerator(workspace, accelerator, path=topk_ckpt_path)


def save_interval_ckpt(workspace, accelerator):
    save_dir = os.path.join(workspace.output_dir, 'step_checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    # Need to update_bn when the model contains batch norm layers !!!
    save_checkpoint_accelerator(workspace, accelerator, path=os.path.join(save_dir, f'update_step_{workspace.update_step}'))


# ---------------------------------------------------------------------------
# Evaluation sub-routines
# ---------------------------------------------------------------------------

def compute_batch_action_metrics(
    workspace, accelerator, inputs, eval_thresholds, wrist_trans_dim, wrist_dim,
):
    """Compute action accuracy and L1 metrics for a single eval batch.

    Returns None if no valid actions, otherwise a dict with keys:
        accuracy, l1_loss, l1_parts, per_sample_l1, eval_sample
    """
    gt_actions = inputs['actions']
    actions_valid_mask = inputs['actions_valid_mask']
    if not torch.any(actions_valid_mask):
        return None

    with accelerator.autocast():
        pred_actions = workspace.model("infer_action", inputs)

    B, H, D = gt_actions.shape
    eval_sample = torch.any(actions_valid_mask.reshape(B, -1), dim=1)
    actions_valid_mask = actions_valid_mask[eval_sample]

    # Unnormalize to original scale
    if workspace.use_relative_action:
        gt_actions = workspace.normalizer['actions'].unnormalize(gt_actions[eval_sample])
        pred_actions = workspace.normalizer['actions'].unnormalize(pred_actions[eval_sample])
    else:
        gt_actions = workspace.normalizer['motions'].unnormalize(gt_actions[eval_sample])
        pred_actions = workspace.normalizer['motions'].unnormalize(pred_actions[eval_sample])
    gt_actions = gt_actions * actions_valid_mask
    pred_actions = pred_actions * actions_valid_mask

    accuracy = get_action_accuracy(gt_actions, pred_actions, eval_thresholds)

    abs_diff = torch.abs(pred_actions - gt_actions)
    per_sample_l1 = (
        torch.sum(abs_diff.flatten(start_dim=1), dim=1)
        / torch.sum(actions_valid_mask.flatten(start_dim=1), dim=1)
    )
    l1_loss = torch.sum(abs_diff) / torch.sum(actions_valid_mask)

    # Per-part L1 loss
    l1_parts = {}
    for pname, ps, pe in [
        ("wrist_trans", 0, wrist_trans_dim),
        ("wrist_rot", wrist_trans_dim, wrist_dim),
        ("hand", wrist_dim, D),
    ]:
        pvalid = actions_valid_mask[:, :, ps:pe].sum().clamp(min=1)
        l1_parts[pname] = torch.sum(abs_diff[:, :, ps:pe]) / pvalid

    return {
        "accuracy": accuracy,
        "l1_loss": l1_loss,
        "l1_parts": l1_parts,
        "per_sample_l1": per_sample_l1,
        "eval_sample": eval_sample,
    }


def update_attn_sample_tracker(
    inputs, full_seq_attn_maps, eval_sample, per_sample_l1,
    batch_idx, min_loss_sample, max_loss_sample,
):
    """Update min/max loss sample trackers for attention visualization."""
    if full_seq_attn_maps is None:
        return

    attn_weights = full_seq_attn_maps[:, eval_sample, :, :, :]
    eval_indices = torch.nonzero(eval_sample, as_tuple=False).squeeze(1)
    batch_size = inputs["input_ids"].shape[0]

    def build_sample_inputs(sample_batch_idx):
        sample_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size:
                sample_inputs[key] = _to_numpy(value[sample_batch_idx])
            else:
                sample_inputs[key] = _to_numpy(value)
        return sample_inputs

    min_idx = torch.argmin(per_sample_l1).item()
    max_idx = torch.argmax(per_sample_l1).item()
    min_loss = per_sample_l1[min_idx].item()
    max_loss = per_sample_l1[max_idx].item()
    min_batch_idx = eval_indices[min_idx].item()
    max_batch_idx = eval_indices[max_idx].item()

    if min_loss < min_loss_sample['loss']:
        min_loss_sample['loss'] = min_loss
        min_loss_sample['attn_weights'] = attn_weights[:, min_idx, :, :, :].float().cpu()
        min_loss_sample['inputs'] = build_sample_inputs(min_batch_idx)
        min_loss_sample['metadata'] = {
            'batch_idx': batch_idx, 'sample_idx': min_batch_idx,
            'eval_sample_idx': min_idx, 'l1_loss': min_loss,
        }

    if max_loss > max_loss_sample['loss']:
        max_loss_sample['loss'] = max_loss
        max_loss_sample['attn_weights'] = attn_weights[:, max_idx, :, :, :].float().cpu()
        max_loss_sample['inputs'] = build_sample_inputs(max_batch_idx)
        max_loss_sample['metadata'] = {
            'batch_idx': batch_idx, 'sample_idx': max_batch_idx,
            'eval_sample_idx': max_idx, 'l1_loss': max_loss,
        }


def aggregate_val_losses(val_losses, accelerator, step_log):
    """Reduce per-batch validation losses across all processes and write to step_log."""
    for key in val_losses.keys():
        if len(val_losses[key]) == 0:
            local_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)
            num_nonzero_samples = torch.tensor(0, device=accelerator.device)
        else:
            stacked_losses = torch.stack(val_losses[key])
            # Count non-zero losses (losses > 1e-8 to handle floating point precision)
            nonzero_mask = stacked_losses > 1e-8
            num_nonzero_samples = nonzero_mask.sum().to(accelerator.device)
            local_loss_sum = stacked_losses.sum().to(accelerator.device)

        total_num_nonzero = accelerator.reduce(num_nonzero_samples, reduction='sum')
        total_loss_sum = accelerator.reduce(local_loss_sum, reduction='sum')

        # Average only over non-zero samples
        val_losses[key] = (total_loss_sum / total_num_nonzero.clamp(min=1)).item()
        step_log[f'val_{key}'] = val_losses[key]


def aggregate_action_metrics(
    eval_accuracy, eval_l1_loss, eval_l1_loss_parts,
    eval_thresholds, accelerator, step_log,
):
    """Reduce action metrics across processes, write to step_log, and return averaged values."""
    eval_len = len(eval_accuracy)
    device = accelerator.device

    if eval_len > 0:
        sum_accuracy = torch.stack(eval_accuracy).sum(dim=0).to(device)
        sum_l1 = torch.stack(eval_l1_loss).sum().to(device)
        sum_l1_parts = {k: torch.stack(v).sum().to(device) for k, v in eval_l1_loss_parts.items()}
    else:
        sum_accuracy = torch.zeros(len(eval_thresholds), device=device)
        sum_l1 = torch.tensor(0.0, device=device)
        sum_l1_parts = {k: torch.tensor(0.0, device=device) for k in eval_l1_loss_parts}

    count = torch.tensor(eval_len, device=device)
    sum_accuracy = accelerator.reduce(sum_accuracy, reduction='sum')
    sum_l1 = accelerator.reduce(sum_l1, reduction='sum')
    count = accelerator.reduce(count, reduction='sum')

    avg_accuracy = sum_accuracy / count.clamp(min=1)
    avg_l1 = sum_l1 / count.clamp(min=1)
    avg_l1_parts = {
        k: accelerator.reduce(v, reduction='sum') / count.clamp(min=1)
        for k, v in sum_l1_parts.items()
    }

    step_log['eval_l1_loss'] = avg_l1.item()
    for part_name, part_loss in avg_l1_parts.items():
        step_log[f'eval_l1_{part_name}'] = part_loss.item()
    for i, threshold in enumerate(eval_thresholds):
        step_log[f'eval_acc_{threshold}'] = avg_accuracy[i].item()

    return avg_l1, avg_l1_parts, avg_accuracy


def save_attn_samples(workspace, accelerator, min_loss_sample, max_loss_sample):
    """Save attention weights for min/max loss samples to disk (main process only)."""
    if not accelerator.is_main_process or min_loss_sample['attn_weights'] is None:
        return

    print(f"\nSaving attention weights and inputs for selected samples...")
    selected_samples = {
        'lowest_loss': min_loss_sample,
        'highest_loss': max_loss_sample,
    }

    print(f"Selected samples for visualization:")
    print(f"  Lowest loss: batch {min_loss_sample['metadata']['batch_idx']}, "
          f"sample {min_loss_sample['metadata']['sample_idx']}, loss={min_loss_sample['loss']:.4f}")
    print(f"  Highest loss: batch {max_loss_sample['metadata']['batch_idx']}, "
          f"sample {max_loss_sample['metadata']['sample_idx']}, loss={max_loss_sample['loss']:.4f}")

    for name, sample_data in selected_samples.items():
        if sample_data['attn_weights'] is None:
            continue

        print(f"\nProcessing {name} sample...")

        output_dir = os.path.join(
            workspace.output_dir,
            'attention_visualization',
            f'step_{workspace.update_step}',
            name,
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'attention_and_inputs.npz')
        save_payload = dict(sample_data['inputs'])
        save_payload['attn_weights'] = sample_data['attn_weights'].numpy()
        save_payload['metadata'] = np.array(sample_data['metadata'], dtype=object)
        save_payload['update_step'] = np.array(workspace.update_step)
        try:
            np.savez_compressed(output_path, **save_payload)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Error saving attention data: {e}")


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluation(workspace, accelerator, dataloader, step_log):
    if accelerator.is_main_process:
        print(f"Evaluation step {workspace.update_step} started")
    accelerator.wait_for_everyone()

    model = _unwrap_model(workspace)
    wrist_dim = model.shape_meta["obs"]["state"]["wrist"]["shape"][0]
    wrist_trans_dim = 6  # 2 wrists * 3 xyz, fixed layout

    with (
        torch.compiler.set_stance("force_eager"),
        torch.no_grad(),
        eval_with_averaged_model(accelerator, workspace.model, workspace.model_averaging),
    ):
        val_losses = {"total_loss": [], "ce_loss": [], "diffusion_loss": [], "flow_loss": []}
        eval_thresholds = workspace.cfg.training.eval_thresholds
        eval_accuracy = []
        eval_l1_loss = []
        eval_l1_loss_parts = {"wrist_trans": [], "wrist_rot": [], "hand": []}

        min_loss_sample = {'loss': float('inf'), 'attn_weights': None, 'inputs': None, 'metadata': None}
        max_loss_sample = {'loss': float('-inf'), 'attn_weights': None, 'inputs': None, 'metadata': None}
        save_eval_attn_weights = bool(workspace.cfg.training.save_eval_attn_weights)

        for batch_idx, batch in enumerate(dataloader):
            inputs = workspace.preprocess_batch(batch, split_mask=True, sample_fm_time=True)

            # Compute validation loss
            with accelerator.autocast(), torch.inference_mode():
                loss = workspace.model(
                    workspace.objective_func, inputs,
                    return_attn_weights=save_eval_attn_weights,
                )
            for key, loss_ in loss.items():
                val_losses[key].append(loss_.detach())

            # Attention maps for visualization
            full_seq_attn_maps = None
            if save_eval_attn_weights and hasattr(model, 'attn_weights') and len(model.attn_weights) > 0:
                full_seq_attn_maps = torch.stack(model.attn_weights, dim=0)

            # Action metrics
            if 'actions' in inputs and workspace.objective_func != "train_ar":
                metrics = compute_batch_action_metrics(
                    workspace, accelerator, inputs,
                    eval_thresholds, wrist_trans_dim, wrist_dim,
                )
                if metrics is None:
                    continue
                eval_accuracy.append(metrics["accuracy"])
                eval_l1_loss.append(metrics["l1_loss"])
                for k, v in metrics["l1_parts"].items():
                    eval_l1_loss_parts[k].append(v)

                update_attn_sample_tracker(
                    inputs, full_seq_attn_maps, metrics["eval_sample"],
                    metrics["per_sample_l1"], batch_idx,
                    min_loss_sample, max_loss_sample,
                )

            if workspace.cfg.training.max_eval_steps and batch_idx >= (workspace.cfg.training.max_eval_steps - 1):
                break

        # Aggregate across processes
        aggregate_val_losses(val_losses, accelerator, step_log)
        avg_l1, avg_l1_parts, avg_accuracy = aggregate_action_metrics(
            eval_accuracy, eval_l1_loss, eval_l1_loss_parts,
            eval_thresholds, accelerator, step_log,
        )

        # Print summary
        log_msg = f"Eval | Epoch {workspace.epoch} | L1 Loss: {avg_l1.item():.3f} | "
        log_msg += " | ".join([f"{k}: {v.item():.3f}" for k, v in avg_l1_parts.items()])
        log_msg += " | "
        log_msg += " | ".join([
            f"acc thres {threshold}: {avg_accuracy[i].item():.3f}"
            for i, threshold in enumerate(eval_thresholds)
        ])
        if accelerator.is_main_process:
            print(log_msg)

        save_attn_samples(workspace, accelerator, min_loss_sample, max_loss_sample)

    clear_attn_weights(model)
