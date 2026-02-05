"""
Hook-based activation/parameter stats collector for LegendVLA (or any torch.nn.Module).

Metrics and aggregation rules
- overall_mean / overall_std / overall_min / overall_max:
  Computed over all finite values of the tensor after flattening to 1D.
  If the tensor is large, values are uniformly sub-sampled (see sample_size).
- l2_mean / l2_min / l2_max:
  L2 norm is computed per row after reshaping to 2D where the last dimension
  is treated as feature dim: tensor -> [-1, last_dim]. Aggregation is over rows.
- per_dim_mean / per_dim_std / per_dim_min / per_dim_max:
  Per-dimension stats along the last dimension. The tensor is reshaped to
  [-1, last_dim], and statistics are computed across the first dimension.
  If last_dim exceeds max_dim, or if a single module is reused with different
  last_dim sizes, per-dim stats are skipped (per_dim_skipped=True).
- nan_count / inf_count / valid_count:
  Counts of NaN, Inf, and finite values (after flattening).
- outlier_count / outlier_ratio:
  Count/ratio of values with |x - mean| > 3 * std for the sampled/flattened
  values used in overall stats.

Visualization
- per-dim curves (PNG):
  Plots mean, mean±std band, and min/max across the feature dimension.
  Only generated when per_dim_* are available and feature dimension <= max_dims.
- overview bars (PNG):
  For each section (activations/parameters), top-N tensors by overall_std and
  by l2_mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pickle

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.utils.pytorch_util import dict_apply

import torch


def _iter_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(value, name)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_tensors(value, name)
        return


@dataclass
class TensorStatsAccumulator:
    name: str
    max_dim: int = 16384
    sample_size: int = 200000

    valid_count: int = 0
    nan_count: int = 0
    inf_count: int = 0

    overall_sum: float = 0.0
    overall_sumsq: float = 0.0
    overall_min: Optional[float] = None
    overall_max: Optional[float] = None

    l2_sum: float = 0.0
    l2_min: Optional[float] = None
    l2_max: Optional[float] = None
    l2_count: int = 0

    outlier_count: int = 0

    per_dim_sum: Optional[torch.Tensor] = None
    per_dim_sumsq: Optional[torch.Tensor] = None
    per_dim_min: Optional[torch.Tensor] = None
    per_dim_max: Optional[torch.Tensor] = None
    per_dim_count: int = 0
    per_dim_skipped: bool = False
    per_dim_size: Optional[int] = None

    def _update_overall(self, flat: torch.Tensor) -> None:
        valid = flat[torch.isfinite(flat)]
        self.nan_count += int(torch.isnan(flat).sum().item())
        self.inf_count += int(torch.isinf(flat).sum().item())
        if valid.numel() == 0:
            return
        self.valid_count += int(valid.numel())
        self.overall_sum += float(valid.sum().item())
        self.overall_sumsq += float((valid * valid).sum().item())
        cur_min = float(valid.min().item())
        cur_max = float(valid.max().item())
        self.overall_min = cur_min if self.overall_min is None else min(self.overall_min, cur_min)
        self.overall_max = cur_max if self.overall_max is None else max(self.overall_max, cur_max)

        if valid.numel() > 1:
            mean = valid.mean()
            std = valid.std(unbiased=False)
            if std > 0:
                self.outlier_count += int(((valid - mean).abs() > 3 * std).sum().item())

    def _update_l2(self, tensor: torch.Tensor) -> None:
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            rows = tensor.unsqueeze(0)
        else:
            rows = tensor.reshape(-1, tensor.shape[-1])
        if rows.numel() == 0:
            return
        norms = torch.norm(rows, dim=1)
        self.l2_sum += float(norms.sum().item())
        cur_min = float(norms.min().item())
        cur_max = float(norms.max().item())
        self.l2_min = cur_min if self.l2_min is None else min(self.l2_min, cur_min)
        self.l2_max = cur_max if self.l2_max is None else max(self.l2_max, cur_max)
        self.l2_count += int(norms.numel())

    def _update_per_dim(self, tensor: torch.Tensor) -> None:
        if tensor.ndim == 0:
            return
        if tensor.ndim == 1:
            rows = tensor.unsqueeze(0)
        else:
            rows = tensor.reshape(-1, tensor.shape[-1])
        if self.per_dim_size is None:
            self.per_dim_size = rows.shape[-1]
        elif rows.shape[-1] != self.per_dim_size:
            # Same module can be reused with different hidden sizes (e.g., shared Dropout).
            self.per_dim_skipped = True
            return
        if rows.shape[-1] > self.max_dim:
            self.per_dim_skipped = True
            return
        if rows.numel() == 0:
            return
        cur_sum = rows.sum(dim=0)
        cur_sumsq = (rows * rows).sum(dim=0)
        cur_min = rows.min(dim=0).values
        cur_max = rows.max(dim=0).values
        if self.per_dim_sum is None:
            self.per_dim_sum = cur_sum.clone()
            self.per_dim_sumsq = cur_sumsq.clone()
            self.per_dim_min = cur_min.clone()
            self.per_dim_max = cur_max.clone()
        else:
            self.per_dim_sum += cur_sum
            self.per_dim_sumsq += cur_sumsq
            self.per_dim_min = torch.minimum(self.per_dim_min, cur_min)
            self.per_dim_max = torch.maximum(self.per_dim_max, cur_max)
        self.per_dim_count += int(rows.shape[0])

    def update(self, tensor: torch.Tensor) -> None:
        t = tensor.detach().float().cpu()
        flat = t.contiguous().view(-1)
        if flat.numel() > self.sample_size:
            idx = torch.randint(0, flat.numel(), (self.sample_size,))
            flat = flat[idx]
        self._update_overall(flat)
        self._update_l2(t)
        self._update_per_dim(t)

    def summarize(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "name": self.name,
            "valid_count": self.valid_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "outlier_count": self.outlier_count,
            "outlier_ratio": (self.outlier_count / self.valid_count) if self.valid_count else 0.0,
            "overall_mean": None,
            "overall_std": None,
            "overall_min": self.overall_min,
            "overall_max": self.overall_max,
            "l2_mean": None,
            "l2_min": self.l2_min,
            "l2_max": self.l2_max,
            "per_dim_skipped": self.per_dim_skipped,
            "per_dim_mean": None,
            "per_dim_std": None,
            "per_dim_min": None,
            "per_dim_max": None,
        }
        if self.valid_count:
            mean = self.overall_sum / self.valid_count
            var = (self.overall_sumsq / self.valid_count) - (mean * mean)
            summary["overall_mean"] = mean
            summary["overall_std"] = var**0.5 if var > 0 else 0.0
        if self.l2_count:
            summary["l2_mean"] = self.l2_sum / self.l2_count
        if self.per_dim_sum is not None and self.per_dim_count:
            mean = self.per_dim_sum / self.per_dim_count
            var = (self.per_dim_sumsq / self.per_dim_count) - (mean * mean)
            summary["per_dim_mean"] = mean.tolist()
            summary["per_dim_std"] = torch.clamp(var, min=0).sqrt().tolist()
            summary["per_dim_min"] = self.per_dim_min.tolist()
            summary["per_dim_max"] = self.per_dim_max.tolist()
        return summary


@dataclass
class ActivationStatsCollector:
    leaf_only: bool = True
    max_dim: int = 16384
    sample_size: int = 200000
    module_depth: Optional[int] = None
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    stats: Dict[str, TensorStatsAccumulator] = field(default_factory=dict)

    def _get_or_create(self, name: str) -> TensorStatsAccumulator:
        if name not in self.stats:
            self.stats[name] = TensorStatsAccumulator(
                name=name, max_dim=self.max_dim, sample_size=self.sample_size
            )
        return self.stats[name]

    def _hook(self, module_name: str):
        import torch._dynamo as dynamo

        def fn(_module, inputs, outputs):
            for name, tensor in _iter_tensors(inputs, prefix="input"):
                key = f"{module_name}:{name}"
                self._get_or_create(key).update(tensor)
            for name, tensor in _iter_tensors(outputs, prefix="output"):
                key = f"{module_name}:{name}"
                self._get_or_create(key).update(tensor)

        return dynamo.disable(fn)

    def register(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if name == "":
                continue
            if self.module_depth is not None:
                depth = name.count(".") + 1
                if depth != self.module_depth:
                    continue
            if self.leaf_only and any(True for _ in module.children()):
                continue
            handle = module.register_forward_hook(self._hook(name))
            self.handles.append(handle)

    def clear(self) -> None:
        self.stats.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summarize(self) -> Dict[str, Any]:
        return {name: acc.summarize() for name, acc in self.stats.items()}

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summarize(), f, indent=2)


def collect_parameter_stats(
    model: torch.nn.Module,
    max_dim: int = 16384,
    sample_size: int = 200000,
    include_buffers: bool = False,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for name, param in model.named_parameters():
        acc = TensorStatsAccumulator(name=name, max_dim=max_dim, sample_size=sample_size)
        acc.update(param.data)
        summary = acc.summarize()
        summary["shape"] = list(param.shape)
        summary["requires_grad"] = bool(param.requires_grad)
        stats[name] = summary
    if include_buffers:
        for name, buf in model.named_buffers():
            key = f"buffer.{name}"
            acc = TensorStatsAccumulator(name=key, max_dim=max_dim, sample_size=sample_size)
            acc.update(buf.data)
            summary = acc.summarize()
            summary["shape"] = list(buf.shape)
            summary["requires_grad"] = False
            stats[key] = summary
    return stats


def run_with_hooks(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    forward_fn,
    leaf_only: bool = True,
    max_dim: int = 16384,
    sample_size: int = 200000,
    module_depth: Optional[int] = None,
    include_parameters: bool = True,
    include_buffers: bool = False,
    activations_output_path: Optional[str] = None,
    parameters_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    collector = ActivationStatsCollector(
        leaf_only=leaf_only,
        max_dim=max_dim,
        sample_size=sample_size,
        module_depth=module_depth,
    )
    collector.register(model)
    try:
        forward_fn(model, batch)
    finally:
        collector.remove()
    activation_stats = collector.summarize()
    if activations_output_path:
        with open(activations_output_path, "w", encoding="utf-8") as f:
            json.dump(activation_stats, f, indent=2)
    if include_parameters:
        parameter_stats = collect_parameter_stats(
            model,
            max_dim=max_dim,
            sample_size=sample_size,
            include_buffers=include_buffers,
        )
        if parameters_output_path:
            with open(parameters_output_path, "w", encoding="utf-8") as f:
                json.dump(parameter_stats, f, indent=2)
        return {
            "activations": activation_stats,
            "parameters": parameter_stats,
        }
    return activation_stats


def visualize_stats(
    stats: Dict[str, Any],
    output_dir: str,
    max_items: int = 50,
    max_dims: Optional[int] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualize_stats()") from exc

    def _ensure_dir(path: str) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    def _plot_per_dim(name: str, entry: Dict[str, Any], out_dir: Path) -> None:
        mean = entry.get("per_dim_mean")
        std = entry.get("per_dim_std")
        vmin = entry.get("per_dim_min")
        vmax = entry.get("per_dim_max")
        if mean is None or std is None:
            return
        if max_dims is not None and len(mean) > max_dims:
            return
        x = list(range(len(mean)))
        plt.figure(figsize=(10, 4))
        plt.plot(x, mean, label="mean", linewidth=1.0)
        lower = [m - s for m, s in zip(mean, std)]
        upper = [m + s for m, s in zip(mean, std)]
        plt.fill_between(x, lower, upper, alpha=0.2, label="mean±std")
        if vmin is not None and vmax is not None:
            plt.plot(x, vmin, label="min", linewidth=0.8, alpha=0.7)
            plt.plot(x, vmax, label="max", linewidth=0.8, alpha=0.7)
        plt.title(name)
        plt.xlabel("dimension")
        plt.ylabel("value")
        plt.legend(fontsize=8, ncol=4)
        plt.tight_layout()
        plt.savefig(out_dir / f"{_safe_name(name)}_per_dim.png", dpi=150)
        plt.close()

    def _plot_overview(section_name: str, section: Dict[str, Any], out_dir: Path) -> None:
        items: List[Tuple[str, float, float]] = []
        for name, entry in section.items():
            overall_std = entry.get("overall_std")
            l2_mean = entry.get("l2_mean")
            if overall_std is None or l2_mean is None:
                continue
            items.append((name, float(overall_std), float(l2_mean)))
        if not items:
            return
        items = sorted(items, key=lambda x: x[1], reverse=True)[:max_items]
        names = [n for n, _, _ in items]
        stds = [s for _, s, _ in items]
        l2s = [l for _, _, l in items]

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(stds)), stds)
        plt.title(f"{section_name}: top {len(stds)} overall std")
        plt.xlabel("tensor")
        plt.ylabel("overall std")
        plt.xticks(range(len(stds)), names, rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(out_dir / f"{section_name}_overall_std.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(l2s)), l2s)
        plt.title(f"{section_name}: top {len(l2s)} l2 mean")
        plt.xlabel("tensor")
        plt.ylabel("l2 mean")
        plt.xticks(range(len(l2s)), names, rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(out_dir / f"{section_name}_l2_mean.png", dpi=150)
        plt.close()

    root = _ensure_dir(output_dir)
    sections: Dict[str, Dict[str, Any]]
    if "activations" in stats or "parameters" in stats:
        sections = {}
        if "activations" in stats and isinstance(stats["activations"], dict):
            sections["activations"] = stats["activations"]
        if "parameters" in stats and isinstance(stats["parameters"], dict):
            sections["parameters"] = stats["parameters"]
    else:
        sections = {"stats": stats}

    for section_name, section in sections.items():
        sec_dir = _ensure_dir(str(root / section_name))
        _plot_overview(section_name, section, sec_dir)
        for name, entry in section.items():
            if isinstance(entry, dict):
                _plot_per_dim(name, entry, sec_dir)

def sample_fm_time(bsz: int) -> torch.FloatTensor:
    flow_alpha = 1.5
    flow_beta = 1
    flow_t_max = 1 - 0.001
    flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
    z = flow_beta_dist.sample((bsz,))
    t = flow_t_max * (1 - z)  # flip and shift
    return t

def preprocess_batch(model, batch, split_mask: bool = False, is_sample_fm_time: bool = True, dtype: torch.dtype = torch.float32):
    """Preprocess batch for training"""
    input_ids = batch["input_ids"]
    # Get unwrapped model for mask building
    if hasattr(model, 'module'):
        model = model.module
    
    # Build causal mask and position ids
    # We need to move the new created tensors to the same device as the input prepared by the accelerate
    causal_mask, vlm_position_ids, action_position_ids = (
        model.build_causal_mask_and_position_ids(   
            batch["attention_mask"], batch["answer_start_idx"], batch["n_actions"], dtype
        )
    )

    inputs = {
        "input_ids": input_ids,
        "pixel_values": batch["pixel_values"].to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "states": batch["states"].to(dtype),
        "answer_start_idx": batch["answer_start_idx"],
        "is_vla_data": batch["is_vla_data"],
        "n_states": batch["n_states"],
        "n_actions": batch["n_actions"],
    }
    # Add depth_values if available
    if "depth_values" in batch:
        inputs["depth_values"] = batch["depth_values"].to(dtype)
        inputs["has_depth_values"] = batch["has_depth_values"]
    inputs["action_position_ids"] = action_position_ids
    inputs["actions"] = batch["actions"].to(dtype)
    inputs["actions_valid_mask"] = batch["actions_valid_mask"]
    inputs["labels"] = batch["labels"]
    
    if split_mask:
        max_vlm_tokens = input_ids.shape[-1]
        vlm_mask, action_mask = (
            model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
        )
        inputs["vlm_mask"] = vlm_mask
        inputs["action_mask"] = action_mask
    inputs["causal_mask"] = causal_mask

    # Sample flow matching timesteps
    if is_sample_fm_time:
        # We need to move the new created tensors to the same device as the input prepared by the accelerate
        inputs["t"] = sample_fm_time(len(input_ids)).to(input_ids.device).to(dtype)

    return inputs


if __name__ == "__main__":
    def main() -> None: 
        config_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "config"
            / "experiment"
            / "pretrain_legendvla_deepspeed.yaml"
        )
        print(f"Loading config from: {config_path}")
        # Register eval resolver for config
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        
        # Register now resolver for datetime formatting (used by Hydra)
        def now_resolver(format_str: str) -> str:
            """Resolver for ${now:format} interpolation."""
            return datetime.now().strftime(format_str)
        OmegaConf.register_new_resolver("now", now_resolver, replace=True)
        
        # Register hydra resolver (returns empty string for non-hydra contexts)
        def hydra_resolver(key: str) -> str:
            """Resolver for ${hydra:key} interpolation. Returns empty string in test context."""
            return ""
        OmegaConf.register_new_resolver("hydra", hydra_resolver, replace=True)
        cfg = OmegaConf.load(config_path)
        # Populate hydra.job.num for non-hydra runs to satisfy ${hydra.job.num}
        if "hydra" not in cfg:
            cfg.hydra = {}
        if "job" not in cfg.hydra:
            cfg.hydra.job = {}
        if "num" not in cfg.hydra.job:
            cfg.hydra.job.num = 0
        OmegaConf.resolve(cfg)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = bool(getattr(cfg.training, "use_bf16", False))
        dtype = torch.bfloat16 if (use_bf16 and device.type == "cuda") else torch.float32

        model = hydra.utils.instantiate(cfg.policy)
        model.to(device=device, dtype=dtype)
        model.eval()

        def _load_deepspeed_checkpoint(ckpt_dir: str) -> None:
            ckpt_path = Path(ckpt_dir) / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            payload = torch.load(ckpt_path, map_location="cpu")
            state = payload.get("module", payload.get("state_dict", payload))
            if isinstance(state, dict):
                keys = list(state.keys())
                if keys and all(k.startswith("module.") for k in keys):
                    state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)

        if getattr(cfg.training, "load_pretrained_pi05_weights", False):
            model.load_pretrained_pi05_weights()
        elif getattr(cfg.training, "load_pretrained_vlm_weights", False):
            model.load_pretrained_vlm_weights()
        resume_ckpt_dir = getattr(cfg.training, "resume_checkpoint_path", None)
        if resume_ckpt_dir:
            _load_deepspeed_checkpoint(resume_ckpt_dir)

        dataset = hydra.utils.instantiate(cfg.dataset)
        vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
        dataset.vla_dataset.set_preprocessor(vla_processor)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_preprocessor(vlm_processor)

        normalizer_path = getattr(cfg.training, "normalizer_path", None)
        if normalizer_path:
            with open(normalizer_path, "rb") as f:
                normalizer = pickle.load(f)
        else:
            normalizer = dataset.vla_dataset.get_normalizer()
        dataset.vla_dataset.set_normalizer(normalizer)

        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=dataset.get_sampler(**cfg.dataloader.batch_sampler),
            collate_fn=dataset.get_collator(),
            **cfg.dataloader.loader,
        )

        batch = next(iter(dataloader))
        batch = preprocess_batch(model, batch, split_mask=True, is_sample_fm_time=True, dtype=dtype)
        batch = dict_apply(batch, lambda x: x.to(device))
        mask_vla = batch["is_vla_data"].bool()
        mask_vlm = ~mask_vla

        def _select_batch(input_batch: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
            sub_batch = {}
            for key, value in input_batch.items():
                if torch.is_tensor(value) and value.size(0) == mask.size(0):
                    sub_batch[key] = value[mask]
                else:
                    sub_batch[key] = value
            return sub_batch

        def _forward_fn(current_batch: Dict[str, Any]):
            with torch.no_grad():
                model("train", current_batch)

        root_dir = Path(__file__).resolve().parents[2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = root_dir / "outputs" / f"legendvla_stats_{timestamp}"
        vla_dir = out_root / "vla"
        vlm_dir = out_root / "vlm"
        vla_dir.mkdir(parents=True, exist_ok=True)
        vlm_dir.mkdir(parents=True, exist_ok=True)

        if mask_vla.any():
            vla_batch = _select_batch(batch, mask_vla)
            vla_stats = run_with_hooks(
                model,
                vla_batch,
                forward_fn=lambda m, b: _forward_fn(b),
                activations_output_path=str(vla_dir / "activations.json"),
                parameters_output_path=str(vla_dir / "parameters.json"),
            )
            visualize_stats(vla_stats, str(vla_dir / "viz"))

        if mask_vlm.any():
            vlm_batch = _select_batch(batch, mask_vlm)
            vlm_stats = run_with_hooks(
                model,
                vlm_batch,
                forward_fn=lambda m, b: _forward_fn(b),
                activations_output_path=str(vlm_dir / "activations.json"),
                parameters_output_path=str(vlm_dir / "parameters.json"),
            )
            visualize_stats(vlm_stats, str(vlm_dir / "viz"))

    main()
