'''
Evaluation and checkpoint utilities for LegendVLA training workspace.

Extracted from train_legendvla_deepspeed_workspace.py for modularity.
'''

import os
import numpy as np
import torch
from contextlib import contextmanager
import pathlib

from src.utils.metric import get_action_accuracy


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


# Combine validation and sampling, so we can process data only once.
@torch.compiler.disable()
def evaluation(workspace, accelerator, dataloader, step_log):
    if accelerator.is_main_process:
        print(f"Evaluation step {workspace.update_step} started")
    accelerator.wait_for_everyone()
    with torch.no_grad(), eval_with_averaged_model(accelerator, workspace.model, workspace.model_averaging):
        val_losses = dict()
        eval_thresholds = workspace.cfg.training.eval_thresholds
        eval_accuracy = []
        eval_l1_loss = []

        # Track min/max loss batches for saving (only store these two)
        min_loss_sample = {
            'loss': float('inf'),
            'attn_weights': None,
            'inputs': None,
            'metadata': None,
        }
        max_loss_sample = {
            'loss': float('-inf'),
            'attn_weights': None,
            'inputs': None,
            'metadata': None,
        }
        save_eval_attn_weights = bool(workspace.cfg.training.save_eval_attn_weights)
        for batch_idx, batch in enumerate(dataloader):
            inputs = workspace.preprocess_batch(batch, split_mask=True, sample_fm_time=True)

            # Compute validation loss
            with accelerator.autocast():
                loss = workspace.model(
                    workspace.objective_func,
                    inputs,
                    return_attn_weights=save_eval_attn_weights,
                )
            for key, loss_ in loss.items():
                if key not in val_losses:
                    val_losses[key] = list()
                val_losses[key].append(loss_.detach())

            if hasattr(workspace.model, 'module'):
                model = workspace.model.module
            else:
                model = workspace.model
            full_seq_attn_maps = None
            if save_eval_attn_weights and hasattr(model, 'attn_weights') and len(model.attn_weights) > 0:
                full_seq_attn_maps = torch.stack(model.attn_weights, dim=0) # [num_layers, B, num_heads, seq_len, seq_len]
            # Compute action accuracy if actions are available
            if 'actions' in inputs and workspace.objective_func != "train_ar":
                gt_actions = inputs['actions']
                actions_valid_mask = inputs['actions_valid_mask']
                # Get action predictions
                with accelerator.autocast():
                    pred_actions = workspace.model("infer_action", inputs)

                # ignore invalid actions
                B, H, D = gt_actions.shape
                eval_sample = torch.any(actions_valid_mask.reshape(B, -1), dim=1)
                if not torch.any(eval_sample):
                    continue
                actions_valid_mask = actions_valid_mask[eval_sample]
                if workspace.use_relative_action:
                    gt_actions = workspace.normalizer['actions'].unnormalize(gt_actions[eval_sample])
                    pred_actions = workspace.normalizer['actions'].unnormalize(pred_actions[eval_sample])
                else:
                    gt_actions = workspace.normalizer['motions'].unnormalize(gt_actions[eval_sample])
                    pred_actions = workspace.normalizer['motions'].unnormalize(pred_actions[eval_sample])
                gt_actions = gt_actions * actions_valid_mask
                pred_actions = pred_actions * actions_valid_mask

                # Compute accuracy metrics
                batch_accuracy = get_action_accuracy(
                    gt_actions,
                    pred_actions,
                    eval_thresholds,
                )
                eval_accuracy.append(batch_accuracy)

                abs_diff = torch.abs(pred_actions - gt_actions)
                # Compute L1 loss per sample (not batch average)
                # Calculate per-sample loss for finding min/max
                per_sample_l1_loss = torch.sum(
                    abs_diff.flatten(start_dim=1), dim=1
                ) / torch.sum(actions_valid_mask.flatten(start_dim=1), dim=1)  # [B]

                # Compute batch-level statistics for logging
                actions_valid_num = torch.sum(actions_valid_mask)
                batch_l1_loss = torch.sum(abs_diff) / actions_valid_num
                eval_l1_loss.append(batch_l1_loss)

                # Track min/max loss samples for saving
                if full_seq_attn_maps is not None:
                    # [num_layers, eval_sample, num_heads, seq_len, seq_len]
                    attn_weights = full_seq_attn_maps[:, eval_sample, :, :, :]
                    eval_indices = torch.nonzero(eval_sample, as_tuple=False).squeeze(1)
                    batch_size = inputs["input_ids"].shape[0]

                    def to_numpy(value):
                        if torch.is_tensor(value):
                            return value.detach().float().cpu().numpy()
                        if isinstance(value, np.ndarray):
                            return value
                        return np.array(value)

                    def build_sample_inputs(sample_batch_idx):
                        sample_inputs = {}
                        for key, value in inputs.items():
                            if torch.is_tensor(value) and value.shape[0] == batch_size:
                                sample_inputs[key] = to_numpy(value[sample_batch_idx])
                            else:
                                sample_inputs[key] = to_numpy(value)
                        return sample_inputs

                    # Find min/max loss samples in this batch
                    min_idx = torch.argmin(per_sample_l1_loss).item()
                    max_idx = torch.argmax(per_sample_l1_loss).item()

                    min_loss = per_sample_l1_loss[min_idx].item()
                    max_loss = per_sample_l1_loss[max_idx].item()

                    min_batch_idx = eval_indices[min_idx].item()
                    max_batch_idx = eval_indices[max_idx].item()

                    min_metadata = {
                        'batch_idx': batch_idx,
                        'sample_idx': min_batch_idx,
                        'eval_sample_idx': min_idx,
                        'l1_loss': min_loss,
                    }
                    max_metadata = {
                        'batch_idx': batch_idx,
                        'sample_idx': max_batch_idx,
                        'eval_sample_idx': max_idx,
                        'l1_loss': max_loss,
                    }

                    # Update min loss sample
                    if min_loss < min_loss_sample['loss']:
                        min_loss_sample['loss'] = min_loss
                        min_loss_sample['attn_weights'] = attn_weights[:, min_idx, :, :, :].float().cpu()  # [num_layers, num_heads, seq_len, seq_len]
                        min_loss_sample['inputs'] = build_sample_inputs(min_batch_idx)
                        min_loss_sample['metadata'] = min_metadata

                    # Update max loss sample
                    if max_loss > max_loss_sample['loss']:
                        max_loss_sample['loss'] = max_loss
                        max_loss_sample['attn_weights'] = attn_weights[:, max_idx, :, :, :].float().cpu()  # [num_layers, num_heads, seq_len, seq_len]
                        max_loss_sample['inputs'] = build_sample_inputs(max_batch_idx)
                        max_loss_sample['metadata'] = max_metadata

            if workspace.cfg.training.max_eval_steps and batch_idx >= (workspace.cfg.training.max_eval_steps-1):
                break

        # Process validation loss
        for key in val_losses.keys():
            num_samples = torch.tensor(len(val_losses[key]), device=accelerator.device)
            if len(val_losses[key]) == 0:
                local_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)
            else:
                local_loss_sum = torch.stack(val_losses[key]).sum().to(accelerator.device)
            total_num_samples = accelerator.reduce(num_samples, reduction='sum')
            total_loss_sum = accelerator.reduce(local_loss_sum, reduction='sum')
            val_losses[key] = total_loss_sum / total_num_samples.clamp(min=1)
            val_losses[key] = val_losses[key].item()
            step_log[f'val_{key}'] = val_losses[key]

        # fill eval_accuracy and eval_l1_loss to the same length as dataloader
        # Note: we assume at least one action dimension is available for evaluation
        eval_len = len(eval_accuracy)

        # Process action accuracy metrics
        if eval_len > 0:
            # Average over batches
            sum_eval_accuracy = torch.stack(eval_accuracy).sum(dim=0).to(accelerator.device)
            sum_eval_l1_loss = torch.stack(eval_l1_loss).sum().to(accelerator.device)
        else:
            num_thresholds = len(eval_thresholds)
            sum_eval_accuracy = torch.zeros(num_thresholds, device=accelerator.device)
            sum_eval_l1_loss = torch.tensor(0.0, device=accelerator.device)
        eval_len_tensor = torch.tensor(eval_len, device=accelerator.device)
        # Gather metrics across all processes
        sum_eval_accuracy = accelerator.reduce(sum_eval_accuracy, reduction='sum')
        sum_eval_l1_loss = accelerator.reduce(sum_eval_l1_loss, reduction='sum')
        eval_len_tensor = accelerator.reduce(eval_len_tensor, reduction='sum')

        eval_accuracy = sum_eval_accuracy / eval_len_tensor.clamp(min=1)
        eval_l1_loss = sum_eval_l1_loss / eval_len_tensor.clamp(min=1)

        # Log accuracy metrics
        step_log['eval_l1_loss'] = eval_l1_loss.item()
        for i, threshold in enumerate(eval_thresholds):
            step_log[f'eval_acc_{threshold}'] = eval_accuracy[i].item()

        # Create log message
        log_msg = f"Eval | Epoch {workspace.epoch} | L1 Loss: {eval_l1_loss.item():.3f} | "
        log_msg += " | ".join([
            f"acc thres {threshold}: {eval_accuracy[i].item():.3f}"
            for i, threshold in enumerate(eval_thresholds)
        ])
        if accelerator.is_main_process:
            print(log_msg)

        # Save attention weights and inputs for selected samples
        if accelerator.is_main_process and min_loss_sample['attn_weights'] is not None:
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

            # Visualize each selected sample
            for name, sample_data in selected_samples.items():
                if sample_data['attn_weights'] is None:
                    continue

                print(f"\nProcessing {name} sample...")

                # Create output directory
                output_dir = os.path.join(
                    workspace.output_dir,
                    'attention_visualization',
                    f'step_{workspace.update_step}',
                    name
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
