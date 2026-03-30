import time
import gradio as gr
import os
import sys
import subprocess
import shutil
import joblib
from collections import deque
from pathlib import Path
import torch
import numpy as np
from easydict import EasyDict

from lib.pipeline.frame_source import build_frame_source

from scripts.extract_frames import extract_frames_decord
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import interpolate_slam_cameras_at_video_frames, load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


def _stage_html(stage_idx: int, title: str, state: str, seconds: float | None = None, err: str | None = None) -> str:
    """state: pending | running | done | error"""
    dot = ""
    status_line = ""
    if state == "pending":
        dot = ""
    elif state == "running":
        dot = "<span class='spinner' aria-label='running'></span>"
    elif state == "done":
        dot = "<span class='check'>✓</span>"
        # seconds under check icon
        if seconds is None:
            status_line = ""
        else:
            status_line = f"{seconds:.0f}s"
    else:  # error
        dot = "<span class='err'>×</span>"
        status_line = "ERR"

    # Minimal: dot (spinner/check) + title + optional seconds.
    return (
        f'<div class="step step-{state}">'
        f'<div class="step-dot">{dot}</div>'
        f'<div class="step-status">{status_line}</div>'
        f'<div class="step-title">{title}</div>'
        f"</div>"
    )


def _video_path(upload) -> str | None:
    if upload is None:
        return None
    if isinstance(upload, str):
        return upload
    if isinstance(upload, dict) and "name" in upload:
        return upload["name"]
    return str(upload)


def _ensure_extracted_frames(video_path: str) -> None:
    """Populate <parent>/<stem>/extracted_images/*.jpg expected by build_frame_source."""
    vp = Path(video_path)
    out_dir = vp.parent / vp.stem / "extracted_images"
    if out_dir.is_dir():
        n = sum(
            1
            for name in os.listdir(out_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if n > 0:
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    n = extract_frames_decord(str(vp), str(out_dir), quality=95, format="jpg", verbose=False)
    if n <= 0:
        raise RuntimeError(f"Failed to extract frames (invalid MP4/MOV?): {video_path}")


def render_reconstruction_progress(input_video, img_focal):
    path = _video_path(input_video)
    pending = lambda i, t: _stage_html(i, t, "pending")
    running = lambda i, t: _stage_html(i, t, "running")
    done = lambda i, t, s: _stage_html(i, t, "done", s)

    titles = (
        "Detect",
        "Motion",
        "SLAM",
        "Infiller",
    )

    detail_orig_hidden = None
    detail_cam_hidden = None

    def yield_ui(
        s1, s2, s3, s4, status_html: str, detail_video_original, detail_video_cam
    ):
        return (
            s1,
            s2,
            s3,
            s4,
            status_html,
            detail_video_original,
            detail_video_cam,
        )

    if not path or not os.path.isfile(path):
        err = "Upload a valid video."
        e = lambda i: _stage_html(i, titles[i - 1], "error", err=err)
        yield yield_ui(e(1), e(2), e(3), e(4), err, detail_orig_hidden, detail_cam_hidden)
        return

    args = EasyDict()
    args.video_path = path
    args.input_type = "file"
    args.checkpoint = "./weights/hawor/checkpoints/hawor.ckpt"
    args.infiller_weight = "./weights/hawor/checkpoints/infiller.pt"
    args.vis_mode = "world"
    args.img_focal = img_focal

    t_wall0 = time.perf_counter()

    repo_root = Path(__file__).resolve().parent
    batch_infer_script = repo_root / "scripts" / "batch_infer.py"
    stage_order = ["detect_track", "motion", "slam", "infiller"]
    # Keep UI simple: single GPU by default
    gpus = "0"

    # Store all UI pipeline artifacts under shared path requested by user.
    # (Use /share_data/jixinhao as canonical path.)
    shared_root = Path("/share_data/jixinhao")
    run_root = shared_root / "ui_preview_runs" / f"ui_runs_{int(time.time())}_{os.getpid()}"
    run_root.mkdir(parents=True, exist_ok=True)

    # Copy uploaded video into shared run folder, so stage outputs are also
    # generated under /share_data/jixinhao instead of /tmp/gradio.
    inputs_dir = run_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    src_video = Path(path).resolve()
    dst_video = inputs_dir / src_video.name
    if src_video != dst_video:
        shutil.copy2(str(src_video), str(dst_video))
    path = str(dst_video)

    video_list_file = run_root / "videos.txt"
    video_list_file.write_text(path + "\n")
    ui_outputs_dir = run_root / "ui_outputs"
    ui_outputs_dir.mkdir(parents=True, exist_ok=True)

    frame_source = None
    start_idx = 0
    end_idx = 0
    seq_folder = None

    def _run_batch_infer_one_stage(stage: str, *, run_dir: Path, extra_args: list[str]):
        cmd = [
            sys.executable,
            str(batch_infer_script),
            "--video_list",
            str(video_list_file),
            "--gpus",
            gpus,
            "--stages",
            stage,
            "--scheduler_mode",
            "legacy",
            "--img_focal",
            str(img_focal),
            "--metric3d_batch_size",
            "16",
            "--run_dir",
            str(run_dir),
            "--slam_backend",
            "dpvo",
        ]
        # Any4D only affects SLAM depth backend.
        # Passing it here keeps the behavior consistent with your request.
        cmd.append("--any4d")
        cmd.extend(extra_args)

        env = os.environ.copy()

        # Keep only a short tail for errors (avoid buffering huge output).
        tail = deque(maxlen=200)
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"batch_infer failed (rc={rc}) for stage={stage}\n"
                + "".join(tail)[-8000:]
            )

    # --- Stage 1 (detect_track) ---
    yield yield_ui(
        running(1, titles[0]),
        pending(2, titles[1]),
        pending(3, titles[2]),
        pending(4, titles[3]),
        "",
        detail_orig_hidden,
        detail_cam_hidden,
    )
    try:
        _ensure_extracted_frames(path)
        frame_source = build_frame_source(path)
        end_idx = int(len(frame_source))
        start_idx = 0
        seq_folder = str(Path(path).resolve().parent / Path(path).resolve().stem)
        t_a = time.perf_counter()
        _run_batch_infer_one_stage(
            "detect_track",
            run_dir=run_root / "detect_track",
            extra_args=[],
        )
        dt1 = time.perf_counter() - t_a
    except Exception as ex:
        yield yield_ui(
            _stage_html(1, titles[0], "error", err=str(ex)),
            pending(2, titles[1]),
            pending(3, titles[2]),
            pending(4, titles[3]),
            f"Stage 1 failed: {ex}",
            detail_orig_hidden,
            detail_cam_hidden,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        running(2, titles[1]),
        pending(3, titles[2]),
        pending(4, titles[3]),
        "",
        detail_orig_hidden,
        detail_cam_hidden,
    )

    # --- Stage 2 (motion) ---
    try:
        t_b = time.perf_counter()
        _run_batch_infer_one_stage(
            "motion",
            run_dir=run_root / "motion",
            extra_args=[],
        )
        dt2 = time.perf_counter() - t_b
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            _stage_html(2, titles[1], "error", err=str(ex)),
            pending(3, titles[2]),
            pending(4, titles[3]),
            "Stage 2 failed",
            detail_orig_hidden,
            detail_cam_hidden,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        running(3, titles[2]),
        pending(4, titles[3]),
        "",
        detail_orig_hidden,
        detail_cam_hidden,
    )

    # --- Stage 3 (slam: dpvo + any4d) ---
    try:
        t_c = time.perf_counter()
        _run_batch_infer_one_stage(
            "slam",
            run_dir=run_root / "slam",
            extra_args=[],
        )
        dt3 = time.perf_counter() - t_c
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            done(2, titles[1], dt2),
            _stage_html(3, titles[2], "error", err=str(ex)),
            pending(4, titles[3]),
            "Stage 3 failed",
            detail_orig_hidden,
            detail_cam_hidden,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        done(3, titles[2], dt3),
        running(4, titles[3]),
        "",
        detail_orig_hidden,
        detail_cam_hidden,
    )

    # --- Stage 4 (infiller) ---
    try:
        t_d = time.perf_counter()
        _run_batch_infer_one_stage(
            "infiller",
            run_dir=run_root / "infiller",
            extra_args=[],
        )
        dt4 = time.perf_counter() - t_d
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            done(2, titles[1], dt2),
            done(3, titles[2], dt3),
            _stage_html(4, titles[3], "error", err=str(ex)),
            "Stage 4 failed",
            detail_orig_hidden,
            detail_cam_hidden,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        done(3, titles[2], dt3),
        done(4, titles[3], dt4),
        "",
        detail_orig_hidden,
        detail_cam_hidden,
    )

    # --- Visualization ---
    world_file = os.path.join(seq_folder, "world_space_res.pth")
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_file)

    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    n_slam = int(R_c2w_sla_all.shape[0])
    backend_txt = os.path.join(seq_folder, "SLAM", "slam_backend.txt")
    use_dpvo_vis = False
    if os.path.isfile(backend_txt):
        with open(backend_txt, "r") as bf:
            use_dpvo_vis = bf.read().strip().lower() == "dpvo"
    if use_dpvo_vis or n_slam < vis_end:
        vis_idx = np.arange(vis_start, vis_end, dtype=np.int64)
        R_c2w_sla_all, t_c2w_sla_all = interpolate_slam_cameras_at_video_frames(slam_path, vis_idx)
        R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
        t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    faces = get_mano_faces()
    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ]
    )
    faces_right = np.concatenate([faces, faces_new], axis=0)

    hand = "right"
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(
        pred_trans[hand_idx : hand_idx + 1, vis_start:vis_end],
        pred_rot[hand_idx : hand_idx + 1, vis_start:vis_end],
        pred_hand_pose[hand_idx : hand_idx + 1, vis_start:vis_end],
        betas=pred_betas[hand_idx : hand_idx + 1, vis_start:vis_end],
    )
    right_verts = pred_glob_r["vertices"][0]
    right_dict = {"vertices": right_verts.unsqueeze(0), "faces": faces_right}

    faces_left = faces_right[:, [0, 2, 1]]
    hand = "left"
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(
        pred_trans[hand_idx : hand_idx + 1, vis_start:vis_end],
        pred_rot[hand_idx : hand_idx + 1, vis_start:vis_end],
        pred_hand_pose[hand_idx : hand_idx + 1, vis_start:vis_end],
        betas=pred_betas[hand_idx : hand_idx + 1, vis_start:vis_end],
    )
    left_verts = pred_glob_l["vertices"][0]
    left_dict = {"vertices": left_verts.unsqueeze(0), "faces": faces_left}

    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, left_dict["vertices"].cpu())
    right_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, right_dict["vertices"].cpu())

    try:
        t_vis = time.perf_counter()
        output_pth_cam = os.path.join(seq_folder, f"vis_cam_{vis_start}_{vis_end}")
        os.makedirs(output_pth_cam, exist_ok=True)
        vis_video_cam_path = run_vis2_on_video_cam(
            left_dict,
            right_dict,
            output_pth_cam,
            img_focal,
            frame_source=frame_source,
            R_w2c=R_w2c_sla_all[vis_start:vis_end],
            t_w2c=t_w2c_sla_all[vis_start:vis_end],
            interactive=False,
        )
        # Some viewer versions save `video.mp4` instead of `video_0.mp4`.
        if not vis_video_cam_path or not os.path.isfile(vis_video_cam_path):
            candidates = (
                os.path.join(output_pth_cam, "aitviewer", "video.mp4"),
                os.path.join(output_pth_cam, "aitviewer", "video_0.mp4"),
            )
            for c in candidates:
                if os.path.isfile(c):
                    vis_video_cam_path = c
                    break
        stable_cam_path = None
        if vis_video_cam_path and os.path.isfile(vis_video_cam_path):
            stable_cam_path = str(ui_outputs_dir / "cam.mp4")
            shutil.copy2(vis_video_cam_path, stable_cam_path)
        stable_orig_path = None
        if path and os.path.isfile(path):
            src_ext = Path(path).suffix or ".mp4"
            stable_orig_path = str(ui_outputs_dir / f"original{src_ext}")
            shutil.copy2(path, stable_orig_path)
        dt_vis = time.perf_counter() - t_vis
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            done(2, titles[1], dt2),
            done(3, titles[2], dt3),
            done(4, titles[3], dt4),
            "Visualization failed",
            detail_orig_hidden,
            detail_cam_hidden,
        )
        return

    total = time.perf_counter() - t_wall0
    summary = (
        f"**Done** · {total:.2f}s | "
        f"S1 {dt1:.1f}s · S2 {dt2:.1f}s · S3 {dt3:.1f}s · S4 {dt4:.1f}s"
    )

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        done(3, titles[2], dt3),
        done(4, titles[3], dt4),
        f"<span class='bingo'>✓ Bingo</span><br><span style='font-size:11px;opacity:.75'>{os.path.basename(stable_cam_path or vis_video_cam_path or '')}</span>",
        stable_orig_path or path,
        stable_cam_path or vis_video_cam_path,
    )


def _open_detail(video_original_path, video_cam_path):
    """Open detail compare section: original (left) vs cam (right)."""
    if not video_cam_path:
        # 保持闭合：只显示主页面的 World 结果，避免丑的空对比
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )
    return (
        gr.update(value=video_original_path, visible=True),
        gr.update(value=video_cam_path, visible=True),
    )


header = """
<div style="text-align: center;">
    <h1>Hand Motion Reconstruction</h1>
    <p style="opacity: 0.85; max-width: 42rem; margin: 0 auto;">Upload → Run. Detail after completion.</p>
</div>
"""

with gr.Blocks(
    title="Hand Motion Reconstruction",
) as demo:
    gr.Markdown(header)

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.File(
                label="Video",
                file_types=[".mp4", ".mov", ".avi", ".mkv"],
                type="filepath",
            )
            upload_ok = gr.HTML(value="", visible=False)
            img_focal = gr.Number(label="Focal", value=600)
            submit = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            with gr.Row(elem_classes=["stepper-row"]):
                stage1 = gr.HTML(value=_stage_html(1, "Detect", "pending"))
                stage2 = gr.HTML(value=_stage_html(2, "Motion", "pending"))
                stage3 = gr.HTML(value=_stage_html(3, "SLAM", "pending"))
                stage4 = gr.HTML(value=_stage_html(4, "Infiller", "pending"))

        with gr.Column(scale=1):
            overall_status = gr.HTML(value="", visible=True)
            # no extra buttons; show comparison directly after completion

    with gr.Row():
        detail_video_original = gr.Video(
            label="Original",
            interactive=False,
            visible=True,
        )
        detail_video_cam = gr.Video(
            label="Cam",
            interactive=False,
            visible=True,
        )

    submit.click(
        fn=render_reconstruction_progress,
        inputs=[input_video, img_focal],
        outputs=[
            stage1,
            stage2,
            stage3,
            stage4,
            overall_status,
            detail_video_original,
            detail_video_cam,
        ],
    )

    gr.Examples([["./example/video_0.mp4"]], inputs=input_video)

demo.launch(
    debug=True,
    allowed_paths=["/share_data/jixinhao/ui_preview_runs"],
    css="""

.gradio-container {
    max-width: 1120px;
    margin: auto;
}

/* Overall minimal “paper UI” look */
.gradio-container .prose, .gradio-container .markdown {
    color: #0f172a !important;
}

.gradio-container .block{
    border-radius: 18px;
}

.step{
    flex: 1 1 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 12px;
    border-radius: 16px;
    border: 1px solid rgba(15, 23, 42, 0.10);
    background: rgba(255, 255, 255, 0.75);
}
.step-dot{
    width: 28px;
    height: 28px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 12px;
    color: rgba(15, 23, 42, 0.65);
    background: rgba(2, 6, 23, 0.04);
}
.step-title{
    font-size: 12px;
    font-weight: 720;
    color: rgba(15, 23, 42, 0.95);
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.step-status{
    margin-top: 0px;
    font-size: 12px;
    font-weight: 650;
    opacity: 0.90;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
}
.step-status:empty{
    display: none;
}

.step-pending{
    border-color: rgba(15, 23, 42, 0.10);
    background: rgba(15, 23, 42, 0.03);
}
.step-pending .step-dot{
    background: rgba(15, 23, 42, 0.05);
    color: rgba(15, 23, 42, 0.35);
}
.step-pending .step-status{ color: rgba(15, 23, 42, 0.85); }

.step-running{
    border-color: rgba(245, 158, 11, 0.35);
    background: rgba(254, 243, 199, 0.60);
}
.step-running .step-dot{
    background: rgba(254, 243, 199, 0.95);
    color: rgba(146, 64, 14, 0.95);
}
.step-running .step-status{ color: rgba(146, 64, 14, 0.95); }

.step-done{
    border-color: rgba(34, 197, 94, 0.35);
    background: rgba(220, 252, 231, 0.60);
}
.step-done .step-dot{
    background: rgba(220, 252, 231, 0.95);
    color: rgba(22, 101, 52, 0.95);
}
.step-done .step-status{ color: rgba(22, 101, 52, 0.95); }

.step-error{
    border-color: rgba(239, 68, 68, 0.35);
    background: rgba(254, 226, 226, 0.65);
}
.step-error .step-dot{
    background: rgba(254, 226, 226, 0.95);
    color: rgba(153, 27, 27, 0.95);
}
.step-error .step-status{ color: rgba(153, 27, 27, 0.95); }

/* Additional polish */
body{
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 60%);
}
.step{
    padding: 8px 10px;
    border-radius: 14px;
}

.spinner{
    width: 14px;
    height: 14px;
    border-radius: 999px;
    border: 2px solid rgba(15, 23, 42, 0.25);
    border-top-color: currentColor;
    animation: spin 0.9s linear infinite;
    display: inline-block;
    color: inherit;
}
.check{
    font-size: 16px;
    font-weight: 900;
    line-height: 1;
    color: currentColor;
}
.err{
    font-size: 16px;
    font-weight: 900;
    line-height: 1;
    color: currentColor;
}

@keyframes spin{
    to{ transform: rotate(360deg); }
}
.summary-md{
    font-size: 12px;
    font-weight: 560;
    opacity: 0.88;
    margin-top: 2px;
}
.results-title{
    margin-top: 6px;
    margin-bottom: 2px;
    font-size: 13px;
    opacity: 0.80;
    font-weight: 650;
}
.stepper-row{
    gap: 10px;
}

.upload-ok{
    color: rgba(22, 101, 52, 0.95);
    font-weight: 800;
}
.bingo{
    color: rgba(22, 101, 52, 0.95);
    font-weight: 900;
    font-size: 14px;
}
""",
)
