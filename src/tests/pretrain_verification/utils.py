"""
Shared utilities for pretrain verification scripts.

Provides CheckResult tracking, output directory management,
and common visualization helpers.
"""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Check result tracking
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhaseReport:
    """Collects CheckResults and writes summary JSON."""

    def __init__(self, phase_name: str, output_dir: str | Path):
        self.phase_name = phase_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checks: list[CheckResult] = []

    def add(self, result: CheckResult) -> None:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.message}")
        self.checks.append(result)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def save(self) -> Path:
        report = {
            "phase": self.phase_name,
            "timestamp": datetime.now().isoformat(),
            "all_passed": self.all_passed,
            "total": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": sum(1 for c in self.checks if not c.passed),
            "checks": [c.to_dict() for c in self.checks],
        }
        path = self.output_dir / "report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return path

    def print_summary(self) -> None:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = total - passed
        status = "ALL PASSED" if self.all_passed else "SOME FAILED"
        print(f"\n{'='*60}")
        print(f"  {self.phase_name}: {status} ({passed}/{total} passed, {failed} failed)")
        if not self.all_passed:
            for c in self.checks:
                if not c.passed:
                    print(f"    FAIL: {c.name} - {c.message}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def get_output_dir(phase: str) -> Path:
    """Return outputs/pretrain_verification/{phase}/ under the project root."""
    project_root = Path(__file__).resolve().parents[3]
    out = project_root / "outputs" / "pretrain_verification" / phase
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Visualization helpers (matplotlib-based)
# ---------------------------------------------------------------------------

def safe_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping visualization")
        return None


def plot_bar_chart(
    values: np.ndarray | list,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    highlight_mask: np.ndarray | None = None,
    highlight_color: str = "red",
    normal_color: str = "steelblue",
    figsize: tuple = (12, 4),
):
    """Bar chart with optional highlighted bars for anomalies."""
    plt = safe_import_plt()
    if plt is None:
        return
    values = np.asarray(values)
    x = np.arange(len(values))
    colors = np.full(len(values), normal_color, dtype=object)
    if highlight_mask is not None:
        colors[np.asarray(highlight_mask, dtype=bool)] = highlight_color

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_histogram_panel(
    data: np.ndarray,
    title: str,
    save_path: Path,
    n_cols: int = 8,
    bins: int = 50,
    figsize_per_cell: tuple = (2.5, 2.0),
):
    """Panel of histograms, one per last-dim column of data [N, D]."""
    plt = safe_import_plt()
    if plt is None:
        return
    n_dims = data.shape[1]
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows))
    axes = np.atleast_2d(axes)
    for dim in range(n_dims):
        r, c = divmod(dim, n_cols)
        ax = axes[r, c]
        ax.hist(data[:, dim], bins=bins, color="steelblue", alpha=0.8)
        ax.set_title(f"dim {dim}", fontsize=7)
        ax.tick_params(labelsize=5)
    # Hide unused subplots
    for dim in range(n_dims, n_rows * n_cols):
        r, c = divmod(dim, n_cols)
        axes[r, c].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_image_grid(
    images: list[np.ndarray],
    title: str,
    save_path: Path,
    n_cols: int = 6,
):
    """Plot a grid of images. Each image should be [H, W, 3] uint8."""
    plt = safe_import_plt()
    if plt is None:
        return
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for i, img in enumerate(images):
        r, c = divmod(i, n_cols)
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        axes[r, c].set_title(f"frame {i}", fontsize=8)
    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_loss_curves(
    loss_dict: dict[str, list[float]],
    title: str,
    save_path: Path,
    figsize: tuple = (10, 5),
):
    """Plot multiple loss curves on the same figure."""
    plt = safe_import_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=figsize)
    for name, values in loss_dict.items():
        ax.plot(values, label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scatter_2d(
    points: np.ndarray,
    colors: np.ndarray | None,
    title: str,
    save_path: Path,
    xlabel: str = "x",
    ylabel: str = "y",
    cmap: str = "viridis",
    colorbar_label: str = "",
    figsize: tuple = (6, 5),
):
    """2D scatter plot with optional color coding."""
    plt = safe_import_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(points[:, 0], points[:, 1], c=colors, cmap=cmap, s=15, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if colors is not None:
        cb = fig.colorbar(sc, ax=ax)
        if colorbar_label:
            cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Common data helpers
# ---------------------------------------------------------------------------

def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """Quick stats for a tensor."""
    t_float = t.detach().float()
    return {
        "mean": t_float.mean().item(),
        "std": t_float.std().item(),
        "min": t_float.min().item(),
        "max": t_float.max().item(),
        "nan_count": int(torch.isnan(t_float).sum().item()),
        "inf_count": int(torch.isinf(t_float).sum().item()),
        "shape": list(t.shape),
    }


def assert_check(condition: bool, name: str, message: str, details: dict | None = None) -> CheckResult:
    """Create a CheckResult from a boolean assertion."""
    return CheckResult(
        name=name,
        passed=condition,
        message=message,
        details=details or {},
    )


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

# Description of what each check verifies, keyed by check number prefix.
CHECK_DESCRIPTIONS: dict[str, str] = {
    # Phase 0: Normalizer
    "0.1": "Normalizer 统计量分布审查：验证 scale/offset/q01/q99 每维度值，"
           "确认 ignored dims (6-18) scale=1.0, offset=0.0",
    "0.2": "正/反归一化往返一致性：normalize → unnormalize 重建误差 < 1e-5",
    "0.3": "归一化后数据分布：每维度 histogram，非 ignored dims 的 p99 abs < 2.0",
    # Phase 1: Data pipeline
    "1.1": "Collator 输出 schema：验证所有 key 存在且形状正确 "
           "(states=[B,18,48], actions=[B,64,48])",
    "1.2": "VLA/VLM 交错比率：5:1 配置下 VLA 占比应在 [0.78, 0.88]",
    "1.3": "视觉输入验证：video frames=6, grid_thw 与 pixel_values 一致",
    "1.4": "Token 序列结构：VLA labels 全为 -100, VLM labels 在 answer_start 后有效",
    "1.5": "actions_valid_mask 一致性：valid rows = n_actions, VLM 样本全 False",
    "1.6": "Chat Template & Tokenization：decode 文本 + 特殊 token 高亮 + token type heatmap",
    "1.7": "States/Actions 2D 投影：wrist 3D 位置通过相机内参投影到图像平面",
    "1.8": "数据增强一致性：同一 video 的 6 帧共享相同 color jitter 参数",
    # Phase 2: Embeddings
    "2.1": "State/Action/AR Encoder 输出分布：mean/std/min/max, per-dim 曲线",
    "2.2": "Time Embedding 结构：dist(t=0, t=1) > 1.0, PCA 可视化",
    "2.3": "Fourier Feature 确定性：buffer requires_grad=False, 多次调用输出完全一致",
    # Phase 3: Forward stability
    "3.1": "Activation/Parameter 全局审计：NaN/Inf 检测 + AdaLNZero gate 监控 "
           "(已知 grad spike 风险)",
    "3.2": "Prefix KV Cache：shape=[num_layers,B,kv_heads,seq_len,head_dim], "
           "flow_expert.detach_prefix_kv=True 下 requires_grad=False",
    "3.3": "BFloat16 精度：bf16 vs fp32 loss 相对误差 < 10% (仅 CUDA)",
    # Phase 4: Loss
    "4.1": "CE Loss：纯 VLA batch → 0, 混合 batch > 0, 初始值 ≈ ln(151936) ≈ 11.9",
    "4.2": "Flow Loss：手工计算 target_v = actions - (1-σ_min)·noise, loss vs t 曲线",
    "4.3": "DiffLoss：chunk 构造 (unfold size=4), hidden state 采集位置",
    "4.4": "Total Loss 加权：total = 0.1·CE + 20.0·Diff + 1.0·Flow, 分量平衡检查",
    "4.5": "梯度流验证：3 组参数 (VLM/action_expert/diffloss) 均有非零梯度, "
           "flow_expert.detach_prefix_kv=True 阻断 backbone 梯度",
    # Phase 5: RTC
    "5.1": "RTC Delay 采样分布：exp 策略, 100k 次采样, P(delay=0) > 2×P(delay=7)",
    "5.2": "Prefix/Postfix Mask：forced_delay=3, 位置 0-2 为 prefix (t=1.0)",
    "5.3": "RTC Noisy Actions：prefix 位置 = clean actions, postfix = psi_t 插值",
    # Phase 6: Flow matching
    "6.1": "psi_t 插值：t=0 → 纯噪声, t=1 → ≈target, t=0.5 → 线性中点",
    "6.2": "Velocity Target：target_v = actions - (1-σ_min)·noise, atol=1e-6",
    "6.3": "Beta 时间采样：z~Beta(1.5,1.0), t=0.999·(1-z), E[t]≈0.40",
    "6.4": "Euler 积分闭环：oracle velocity, 10 步从噪声恢复 target, error < 0.05",
    # Phase 7: DiffLoss
    "7.1": "Zero-Init 验证：AdaLN modulation & FinalLayer 权重为零, 初始输出 abs < 0.1",
    "7.2": "Chunk 构造：build_dense_diffloss_inputs shape 和 hidden state 对齐",
    "7.3": "采样质量：初始 DiffLoss 采样值有限 (finite)",
    # Phase 8: Training loop
    "8.1": "Optimizer 参数组：无重叠、无遗漏, dim≥2 有 weight_decay",
    "8.2": "LR Schedule：cosine with warmup=2000, 模拟 10k 步曲线",
    "8.3": "Gradient Clipping：max_norm=1.0, clip 后 grad norm ≤ 1.0",
    "8.4": "[P0] Mini Training 收敛：100 步, loss 下降 > 5%, 无 NaN",
    "8.5": "[P0] 单 Batch 过拟合：500 步, final_loss < 0.3 × initial_loss",
    # Phase 9: Inference
    "9.1": "Flow 推理形状：输出 [B,64,48], 全部 finite, 无效位置为 0",
    "9.2": "RTC 推理一致性：prev_action_chunk 前 d 步 pinned (inference_delay=d)",
    "9.3": "Train-Infer 闭环：过拟合 300 步后推理, 平均 L1 < 1.0",
    "9.4": "AR 推理 (DiffLoss)：infer_vla 返回有效 generated_actions",
    "9.5": "VLM 推理：infer_vlm 返回有效 generated_ids, 终止于 EOS",
}


def _embed_image_b64(path: Path) -> str:
    """Read a PNG file and return a base64 data URI."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def _get_check_number(name: str) -> str:
    """Extract check number like '4.2' from name like '4.2 flow_loss'."""
    parts = name.split(" ", 1)
    return parts[0] if parts else name


_HTML_TEMPLATE_HEAD = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LegendVLA Pretrain Verification Report</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,"Segoe UI",Roboto,sans-serif;
     background:#f5f5f5;color:#333;padding:2rem;max-width:1200px;margin:0 auto}
h1{margin-bottom:.3rem}
.ts{color:#888;font-size:.85rem;margin-bottom:1.5rem}
.summary{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap}
.scard{background:#fff;border-radius:8px;padding:1rem 1.5rem;
       box-shadow:0 1px 3px rgba(0,0,0,.1);min-width:120px}
.scard.pass{border-left:4px solid #4caf50}
.scard.fail{border-left:4px solid #f44336}
.scard .num{font-size:2rem;font-weight:700}
.scard .lab{color:#888;font-size:.85rem}
.phase{background:#fff;border-radius:8px;margin-bottom:1.5rem;
       box-shadow:0 1px 3px rgba(0,0,0,.1);overflow:hidden}
.ph{padding:1rem 1.5rem;background:#fafafa;border-bottom:1px solid #eee}
.ph h2{font-size:1.1rem;display:inline}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;
       font-size:.75rem;font-weight:600;margin-left:.5rem}
.badge.pass{background:#e8f5e9;color:#2e7d32}
.badge.fail{background:#ffebee;color:#c62828}
.pb{padding:1rem 1.5rem}
.chk{padding:.75rem 0;border-bottom:1px solid #f0f0f0}
.chk:last-child{border-bottom:none}
.ch{display:flex;align-items:center;gap:.5rem}
.cb{display:inline-block;width:50px;text-align:center;padding:2px 0;
    border-radius:4px;font-size:.7rem;font-weight:600;flex-shrink:0}
.cb.pass{background:#e8f5e9;color:#2e7d32}
.cb.fail{background:#ffebee;color:#c62828}
.cn{font-weight:600;font-size:.95rem}
.cd{color:#666;font-size:.85rem;margin:.25rem 0 0 58px}
.cm{color:#999;font-size:.8rem;margin:.25rem 0 0 58px;font-style:italic}
.cdet{margin:.5rem 0 0 58px}
.cdet summary{cursor:pointer;font-size:.8rem;color:#1976d2}
.cdet pre{background:#f5f5f5;padding:.5rem;border-radius:4px;
          font-size:.75rem;overflow-x:auto;margin-top:.25rem;max-height:300px;overflow-y:auto}
.imgs{padding:1rem 1.5rem;border-top:1px solid #eee}
.imgs h3{font-size:.95rem;color:#666;margin-bottom:.75rem}
.igrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:1rem}
.iitem{text-align:center}
.iitem img{max-width:100%;border:1px solid #eee;border-radius:4px}
.iitem .cap{font-size:.75rem;color:#888;margin-top:.25rem}
</style>
</head>
<body>
<h1>LegendVLA Pretrain Verification Report</h1>
"""


def generate_html_report(output_root: Path) -> Path:
    """Aggregate all phase report.json + PNGs into a single self-contained HTML report.

    Scans output_root for subdirectories containing report.json, collects
    check results and images, and writes report.html.

    Returns the path to the generated HTML file.
    """
    report_files = sorted(output_root.glob("*/report.json"))
    if not report_files:
        print("  No report.json files found, skipping HTML generation.")
        return output_root / "report.html"

    phases = []
    total_checks = 0
    total_passed = 0

    for rpath in report_files:
        with open(rpath, encoding="utf-8") as f:
            data = json.load(f)
        phase_dir = rpath.parent
        pngs = sorted(phase_dir.glob("*.png")) + sorted(phase_dir.glob("**/*.png"))
        # Deduplicate while preserving order
        seen = set()
        unique_pngs = []
        for p in pngs:
            if p not in seen:
                seen.add(p)
                unique_pngs.append(p)
        phases.append({"data": data, "dir": phase_dir, "pngs": unique_pngs})
        total_checks += data.get("total", 0)
        total_passed += data.get("passed", 0)

    total_failed = total_checks - total_passed
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = [_HTML_TEMPLATE_HEAD]
    html_parts.append(f'<p class="ts">Generated: {now}</p>')

    # Summary cards
    all_ok = total_failed == 0
    html_parts.append('<div class="summary">')
    html_parts.append(
        f'<div class="scard pass"><div class="num">{total_passed}</div>'
        f'<div class="lab">Passed</div></div>'
    )
    fail_cls = "fail" if total_failed > 0 else "pass"
    html_parts.append(
        f'<div class="scard {fail_cls}"><div class="num">{total_failed}</div>'
        f'<div class="lab">Failed</div></div>'
    )
    html_parts.append(
        f'<div class="scard {"pass" if all_ok else "fail"}">'
        f'<div class="num">{len(phases)}</div>'
        f'<div class="lab">Phases</div></div>'
    )
    html_parts.append('</div>')

    # Per-phase sections
    for phase in phases:
        data = phase["data"]
        phase_name = data.get("phase", "Unknown")
        phase_passed = data.get("all_passed", False)
        badge = "pass" if phase_passed else "fail"
        badge_text = "PASS" if phase_passed else "FAIL"
        p_pass = data.get("passed", 0)
        p_total = data.get("total", 0)

        html_parts.append('<div class="phase">')
        html_parts.append(
            f'<div class="ph"><h2>{phase_name}</h2>'
            f'<span class="badge {badge}">{badge_text} {p_pass}/{p_total}</span></div>'
        )
        html_parts.append('<div class="pb">')

        for check in data.get("checks", []):
            c_name = check.get("name", "")
            c_passed = check.get("passed", False)
            c_msg = check.get("message", "")
            c_details = check.get("details", {})
            c_num = _get_check_number(c_name)
            c_desc = CHECK_DESCRIPTIONS.get(c_num, "")
            cb_cls = "pass" if c_passed else "fail"
            cb_text = "PASS" if c_passed else "FAIL"

            html_parts.append('<div class="chk">')
            html_parts.append(
                f'<div class="ch"><span class="cb {cb_cls}">{cb_text}</span>'
                f'<span class="cn">{c_name}</span></div>'
            )
            if c_desc:
                html_parts.append(f'<div class="cd">{c_desc}</div>')
            if c_msg:
                html_parts.append(f'<div class="cm">{c_msg}</div>')
            if c_details:
                details_json = json.dumps(c_details, indent=2, ensure_ascii=False, default=str)
                html_parts.append(
                    f'<details class="cdet"><summary>Details</summary>'
                    f'<pre>{details_json}</pre></details>'
                )
            html_parts.append('</div>')

        html_parts.append('</div>')  # .pb

        # Images
        if phase["pngs"]:
            html_parts.append('<div class="imgs">')
            html_parts.append(f'<h3>Visualizations ({len(phase["pngs"])} images)</h3>')
            html_parts.append('<div class="igrid">')
            for png_path in phase["pngs"]:
                try:
                    data_uri = _embed_image_b64(png_path)
                    caption = png_path.stem
                    html_parts.append(
                        f'<div class="iitem"><img src="{data_uri}" alt="{caption}" '
                        f'loading="lazy"><div class="cap">{caption}</div></div>'
                    )
                except Exception:
                    pass
            html_parts.append('</div></div>')

        html_parts.append('</div>')  # .phase

    html_parts.append('</body></html>')

    out_path = output_root / "report.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\n  HTML report: {out_path}")
    return out_path
