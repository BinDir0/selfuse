import time
import gradio as gr
import os
from pathlib import Path
import torch
import numpy as np
from easydict import EasyDict

from scripts.extract_frames import extract_frames_decord
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import interpolate_slam_cameras_at_video_frames, load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video


def _stage_html(stage_idx: int, title: str, state: str, seconds: float | None = None, err: str | None = None) -> str:
    """state: pending | running | done | error"""
    if state == "pending":
        bg, line = "#d1d5db", "等待中"
    elif state == "running":
        bg, line = "#fde68a", "运行中…"
    elif state == "done":
        bg = "#4ade80"
        line = f"完成 · {seconds:.2f}s" if seconds is not None else "完成"
    else:
        bg, line = "#fca5a5", err or "失败"
    return (
        f'<div style="padding:10px 14px;border-radius:10px;background:{bg};margin-bottom:8px;'
        f'color:#111;font-family:system-ui,sans-serif;">'
        f"<b>Stage {stage_idx}: {title}</b><br>"
        f'<span style="opacity:.9">{line}</span></div>'
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
        raise RuntimeError(f"无法从视频抽取帧（请确认文件为有效 MP4/MOV 等）: {video_path}")


def render_reconstruction_progress(input_video, img_focal):
    path = _video_path(input_video)
    pending = lambda i, t: _stage_html(i, t, "pending")
    running = lambda i, t: _stage_html(i, t, "running")
    done = lambda i, t, s: _stage_html(i, t, "done", s)

    titles = (
        "Detect / Track",
        "Motion estimation",
        "SLAM",
        "Infiller",
    )

    NO_V = (None, None)

    def yield_ui(s1, s2, s3, s4, summary: str, video_orig, video_world):
        return s1, s2, s3, s4, summary, video_orig, video_world

    if not path or not os.path.isfile(path):
        err = "请上传有效视频文件。"
        e = lambda i: _stage_html(i, titles[i - 1], "error", err=err)
        yield yield_ui(e(1), e(2), e(3), e(4), err, *NO_V)
        return

    args = EasyDict()
    args.video_path = path
    args.input_type = "file"
    args.checkpoint = "./weights/hawor/checkpoints/hawor.ckpt"
    args.infiller_weight = "./weights/hawor/checkpoints/infiller.pt"
    args.vis_mode = "world"
    args.img_focal = img_focal

    t_wall0 = time.perf_counter()

    # --- Stage 1 ---
    yield yield_ui(
        running(1, titles[0]),
        pending(2, titles[1]),
        pending(3, titles[2]),
        pending(4, titles[3]),
        "进行中：Stage 1（抽帧 + 检测 / 追踪）×",
        *NO_V,
    )
    try:
        _ensure_extracted_frames(path)
        t_a = time.perf_counter()
        start_idx, end_idx, seq_folder, frame_source = detect_track_video(args)
        dt1 = time.perf_counter() - t_a
    except Exception as ex:
        yield yield_ui(
            _stage_html(1, titles[0], "error", err=str(ex)),
            pending(2, titles[1]),
            pending(3, titles[2]),
            pending(4, titles[3]),
            f"Stage 1 失败: {ex}",
            *NO_V,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        running(2, titles[1]),
        pending(3, titles[2]),
        pending(4, titles[3]),
        "进行中：Stage 2 ×",
        *NO_V,
    )

    # --- Stage 2 ---
    try:
        t_b = time.perf_counter()
        frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
        dt2 = time.perf_counter() - t_b
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            _stage_html(2, titles[1], "error", err=str(ex)),
            pending(3, titles[2]),
            pending(4, titles[3]),
            f"Stage 2 失败: {ex}",
            *NO_V,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        running(3, titles[2]),
        pending(4, titles[3]),
        "进行中：Stage 3 ×",
        *NO_V,
    )

    # --- Stage 3 (SLAM + load trajectory) ---
    try:
        t_c = time.perf_counter()
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        dt3 = time.perf_counter() - t_c
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            done(2, titles[1], dt2),
            _stage_html(3, titles[2], "error", err=str(ex)),
            pending(4, titles[3]),
            f"Stage 3 失败: {ex}",
            *NO_V,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        done(3, titles[2], dt3),
        running(4, titles[3]),
        "进行中：Stage 4 ×",
        *NO_V,
    )

    # --- Stage 4 ---
    try:
        t_d = time.perf_counter()
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
            args, start_idx, end_idx, frame_chunks_all
        )
        dt4 = time.perf_counter() - t_d
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            done(2, titles[1], dt2),
            done(3, titles[2], dt3),
            _stage_html(4, titles[3], "error", err=str(ex)),
            f"Stage 4 失败: {ex}",
            *NO_V,
        )
        return

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        done(3, titles[2], dt3),
        done(4, titles[3], dt4),
        "进行中：可视化 ×",
        *NO_V,
    )

    # --- Visualization: align camera path with demo.py --vis_mode world ----------
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
        if args.vis_mode == "world":
            output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
            os.makedirs(output_pth, exist_ok=True)
            vis_video_path = run_vis2_on_video(
                left_dict,
                right_dict,
                output_pth,
                img_focal,
                frame_source=frame_source,
                R_c2w=R_c2w_sla_all[vis_start:vis_end],
                t_c2w=t_c2w_sla_all[vis_start:vis_end],
                interactive=False,
            )
        else:
            raise NotImplementedError("vis_mode must be world for this demo")
        dt_vis = time.perf_counter() - t_vis
    except Exception as ex:
        yield yield_ui(
            done(1, titles[0], dt1),
            done(2, titles[1], dt2),
            done(3, titles[2], dt3),
            done(4, titles[3], dt4),
            f"可视化失败: {ex}",
            *NO_V,
        )
        return

    total = time.perf_counter() - t_wall0
    summary = (
        f"**总耗时** {total:.2f}s（含可视化 {dt_vis:.2f}s）| "
        f"Stage1 {dt1:.2f}s · Stage2 {dt2:.2f}s · Stage3 {dt3:.2f}s · Stage4 {dt4:.2f}s  "
        f"· 左：原视频 · 右：同 `demo.py --vis_mode world --headless` 导出"
    )

    yield yield_ui(
        done(1, titles[0], dt1),
        done(2, titles[1], dt2),
        done(3, titles[2], dt3),
        done(4, titles[3], dt4),
        summary,
        path,
        vis_video_path,
    )


header = """
<div style="text-align: center;">
    <h1>视频手部运动重建</h1>
    <p style="opacity: 0.85; max-width: 42rem; margin: 0 auto;">上传视频并设置焦距后点击提交，流水线将完成检测追踪、运动估计、SLAM 与补全，并输出 World 空间可视化对比。</p>
</div>
"""

with gr.Blocks(
    title="视频手部运动重建",
) as demo:
    gr.Markdown(header)

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Input video", sources=["upload"])
            img_focal = gr.Number(label="Focal Length", value=600)
            submit = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("**流水线进度**（灰=等待，黄=运行，绿=完成）")
            stage1 = gr.HTML()
            stage2 = gr.HTML()
            stage3 = gr.HTML()
            stage4 = gr.HTML()
            timing_md = gr.Markdown()

    gr.Markdown("### 结果对比（完成后显示）")
    with gr.Row():
        video_original = gr.Video(
            label="原视频（输入）",
            interactive=False,
        )
        video_world = gr.Video(
            label="World 可视化（与 demo.py --vis_mode world --headless 写入的影片一致）",
            interactive=False,
        )

    submit.click(
        fn=render_reconstruction_progress,
        inputs=[input_video, img_focal],
        outputs=[stage1, stage2, stage3, stage4, timing_md, video_original, video_world],
    )

    gr.Examples([["./example/video_0.mp4"]], inputs=input_video)

demo.queue()
demo.launch(
    debug=True,
    css=".gradio-container { max-width: 1100px; margin: auto; }",
)
