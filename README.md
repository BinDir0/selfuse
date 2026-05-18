# RoWaH / HaWoR

This repository serves two main workflows:

1. HaWoR batch inference over videos and shard-based datasets
2. Dataset production through the official single-video pipeline

The codebase has accumulated several generations of scripts. The official path is documented below; older helper scripts are still kept for compatibility, but they are no longer the recommended entrypoints.

## Official Entrypoints

Dataset pipeline:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_single_video.example.yaml
```

Batch inference only:

```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3 \
  --stages detect_track,motion,slam,infiller
```

See:

- [docs/dataset_pipeline.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/dataset_pipeline.md)
- [docs/buildai_6000h_infill_build_runbook.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/buildai_6000h_infill_build_runbook.md)
- [docs/repo_map.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/repo_map.md)
- [docs/legacy_tools.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/legacy_tools.md)
- [configs/README.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/configs/README.md)

## Environment Setup

Use two conda environments. The official single-video pipeline resolves these
names automatically:

- `hawor`: frame preparation, detection/tracking, motion, infiller, filtering,
  WebDataset build, and validation
- `any4d`: SLAM/depth stage when Any4D dense depth is enabled

Keeping Any4D separate avoids the Torch/CUDA/dependency conflicts that tend to
show up when HaWoR and Any4D are forced into one environment.

### 1. HaWoR environment

```bash
conda create -n hawor python=3.10 -y
conda activate hawor

pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0
```

Extra dependencies commonly needed by HaWoR/DPVO/DROID:

```bash
pip install --no-build-isolation mmcv
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable
pip install --no-build-isolation git+https://github.com/mattloper/chumpy

# DPVO's setup.py expects Eigen 3.4.0 at thirdparty/DPVO/thirdparty/eigen-3.4.0
# (not vendored / not a submodule — fetch it once):
( cd thirdparty/DPVO && mkdir -p thirdparty \
  && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O thirdparty/eigen-3.4.0.zip \
  && unzip -q -o thirdparty/eigen-3.4.0.zip -d thirdparty/ )

cd thirdparty/DPVO && pip install . --no-build-isolation && cd ../..
cd thirdparty/DROID-SLAM && python setup.py install && cd ../..
```

`requirements.txt` covers only what the pipeline and batch inference need.
The viewers/demo (aitviewer, moderngl-window, pyrender, gradio, HTML4Vision)
are optional and pin `numpy<2`, so they must be installed without dependency
resolution to avoid downgrading numpy:

```bash
pip install --no-deps -r requirements-viewer.txt
```

If `torch-scatter==2.1.2` from `requirements.txt` does not match your CUDA/PyTorch
wheel, reinstall the matching PyG wheel for your local CUDA/PyTorch pair.

### 2. Any4D environment

```bash
conda create -n any4d python=3.12 -y
conda activate any4d

# Any4D pins torch~=2.6.0 (see thirdparty/Any4D/pyproject.toml). torch 2.6
# wheels are NOT published on the cu121 index — use cu124 (tested default) or
# cu126. Pick the CUDA runtime your driver stack supports.
pip install torch==2.6.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd thirdparty/Any4D
pip install -e .
cd ../..

# The SLAM subprocess still launches this repo's batch_infer.py, so the Any4D
# env also needs pipeline-control, frame/scale, and DPVO dependencies.
pip install joblib tqdm natsort opencv-python-headless pycocotools evo pytorch-minimize

# DPVO needs Eigen 3.4.0 at thirdparty/DPVO/thirdparty/eigen-3.4.0 (see step 1).
# Skip the fetch if you already populated it in the hawor env setup.
( cd thirdparty/DPVO && mkdir -p thirdparty \
  && [ -f thirdparty/eigen-3.4.0/Eigen/Core ] \
  || ( wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O thirdparty/eigen-3.4.0.zip \
       && unzip -q -o thirdparty/eigen-3.4.0.zip -d thirdparty/ ) )
cd thirdparty/DPVO && pip install . --no-build-isolation && cd ../..
```

Install the Any4D checkpoint:

```bash
mkdir -p checkpoints
wget -P checkpoints https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth
```

Recommended runtime exports:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HAWOR_ANY4D_REPO_ROOT=thirdparty/Any4D
export HAWOR_ANY4D_CHECKPOINT_PATH=checkpoints/any4d_4v_combined.pth
```

Put those exports in your shell startup file or the job launcher used for
pipeline runs. Both subprocess environments inherit them.

### 3. Verify environments

```bash
conda env list | grep -E 'hawor|any4d'

conda run -n hawor python - <<'PY'
import sys, torch, cv2, joblib, webdataset
print("hawor python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("hawor env ok")
PY

conda run -n any4d python - <<'PY'
import sys, torch
from lib.pipeline.any4d_depth import resolve_any4d_paths
print("any4d python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print(resolve_any4d_paths())
print("any4d env ok")
PY

bash tools/ops/validate_setup.sh
```

For the simplified `video:` config, do not add runtime paths. The orchestrator
will find `hawor` and `any4d` with `conda env list`. Nested compatibility configs
may still specify `runtimes.hawor_python` and `runtimes.slam_python` explicitly.

## Weights

Required model files:

- `weights/hawor/checkpoints/hawor.ckpt`
- `weights/hawor/checkpoints/infiller.pt`
- `weights/hawor/model_config.yaml`
- `weights/external/detector.pt`
- `thirdparty/DPVO/models/dpvo.pth` when using DPVO
- `weights/external/droid.pth` when using DROID-SLAM
- `thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth` when using Metric3D
- `checkpoints/any4d_4v_combined.pth` or `thirdparty/Any4D/checkpoints/any4d_4v_combined.pth` when using Any4D

HaWoR / WiLoR checkpoints:

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/
```

MANO assets must be downloaded separately from the official MANO site and placed at:

- `_DATA/data/mano/MANO_RIGHT.pkl`
- `_DATA/data_left/mano_left/MANO_LEFT.pkl`

## Common Usage

Run the official single-video-to-WebDataset path:

```yaml
# configs/my_video.yaml
video: /path/to/input.mp4
# output_root is optional; defaults to /path/to/input.hawor_pipeline
```

```bash
python scripts/run_dataset_pipeline.py --config configs/my_video.yaml
```

The default run extracts frames, runs HaWoR/Any4D stages, filters, builds a trainable WebDataset, and validates image/lowdim/MANO/meta/depth outputs. Annotation is skipped unless `annotation.command` is configured, so empty instruction/language fields are valid for this first single-video path.

The pipeline supports three raw-video clipping choices before `prepare`:

- No clipping: omit `clip` or set `clip.mode: none`. The selected dataset adapter runs normally, and later stages use the original prepared clips.
- Heuristic clipping: set `clip.mode: heuristic`. The pipeline writes clipped MP4 files, extracts frames from those clips, and redirects later stages to the clipped-video directory.
- API semantic clipping: set `clip.mode: api`. The multimodal API returns segment boundaries and language in one pass; the pipeline writes clipped MP4 files plus annotation sidecars and redirects later stages to the clipped-video directory.

No clipping is the default:

```yaml
video: /path/to/input.mp4

# clip:
#   mode: none
```

Heuristic clipping example:

```yaml
video: /path/to/input.mp4

clip:
  mode: heuristic
  config: lib/clip/heuristic_clip_config.yaml
```

The heuristic path currently supports raw-video `single_video`, `video_folder`,
and `buildai` inputs.

The API path combines semantic clipping and language annotation in one model
call per raw video:

```yaml
clip:
  mode: api
  prompt_file: lib/annotation/prompts/with_clip/annotation_general_clip.txt
  annotation_suffix: _qwen-annotation.json
  workers: 4
  target_fps: 5.0
```

If `annotate` is included in `--stages`, the pipeline skips the separate
annotation command because the API clipping step already wrote the sidecars.
Provide the DashScope key with `DASHSCOPE_API_KEY`, `clip.api_key`, or
`clip.api_keys_file`; avoid committing keys in config files.

For BuildAI configs, provide the raw video root separately from any existing
processed shard root:

```yaml
dataset:
  adapter: buildai
  start_factory_id: 1
  end_factory_id: 3

paths:
  video_root: /path/to/raw/buildai/videos
  final_dataset_root: /path/to/output/webdataset

clip:
  mode: heuristic
  config: lib/clip/heuristic_clip_config.yaml
```

If `start_factory_id` and `end_factory_id` are set, the clipper scans matching
`factoryNNN` subdirectories under `paths.video_root`. After clipping, the run
is redirected to the generic `video_folder` adapter, so the external BuildAI
preprocess script is not invoked for that run.

The old BuildAI clipping launcher exported `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`
for systems where OpenCV/FFmpeg/decord load the wrong libffi. If your machine
hits a libffi symbol error during clipping or video decoding, export the same
variable before running the full pipeline:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

When using `lib/clip/run_heuristic_video_clipper.sh` directly, set
`HAWOR_CLIP_LD_PRELOAD=/path/to/libffi.so.7` if your libffi path differs.

Direct use of `lib/clip/heuristic_video_clipper.py` also honors
`HAWOR_CLIP_CONFIG` through `lib/clip/clip_config.py`, but the normal dataset
pipeline should prefer the `clip.config:` field in the main YAML. The legacy
`BUILDAI_PIPELINE_CONFIG` variable is still accepted as a fallback.
The heuristic clipper uses `opencv-python`; if `paths.model_path` or
`clip.model_path` points to a detector checkpoint, it also uses `ultralytics`
for the detection gate. Both are part of the core requirements.

Run with the generic language annotation stage:

```bash
export DASHSCOPE_API_KEY=sk-...

python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages prepare,annotate,infer,filter,build,validate
```

`lib/annotation/api_annotation.py` is manifest-driven: it reads `{prepared_state}` and writes
clip sidecars under `{annotation_root}`. The default annotation prompt lives at
`lib/annotation/prompts/without_clip/annotation_industrial_egocentric.txt`. To use a different prompt, edit
the config's `annotation.command` and pass another file with `--prompt_file`:

```yaml
annotation:
  command: >
    {hawor_python} {project_root}/lib/annotation/api_annotation.py
    --prepared_state {prepared_state}
    --annotation_root {annotation_root}
    --annotation_suffix _qwen-annotation.json
    --prompt_file /path/to/custom_prompt.txt
```

For the sake of security, do not commit API keys. Provide the DashScope key at runtime with
`DASHSCOPE_API_KEY`, `--api_key`, or `--api_keys_file`; the environment variable
is the recommended path for normal runs. The annotation stage requires the
`dashscope` package, which is listed in `requirements.txt`.

If you do not run annotation inside the pipeline but still want language in the
final WebDataset, place existing annotation sidecars under `paths.annotation_root`
before `build`, and set `build.annotation_suffix` to the matching suffix. The
standard filename is:

```text
{paths.annotation_root}/{clip_id}{build.annotation_suffix}
```

Each sidecar must be a JSON object with `status: "Valid"` and either
`instruction` or `hierarchy`/`global_analysis` containing `level1` through
`level5` strings. `language` is optional; if omitted, the build can still use
the instruction list. When `build.require_annotation: true`, missing, invalid,
or empty sidecars fail the build. When it is `false`, missing annotations are
allowed and the exported samples contain empty `instruction`, `instruction_num: 0`,
and `language: null`.

For no-clipping runs, `clip_id` is the original prepared clip id. For heuristic
clipping, `clip_id` is generated from the clipped video path relative to the
clipped video root, with path parts joined by `__`, so external annotations
must use those clipped ids. For API clipping, the sidecars are written
automatically next to the clipped-video run under the configured annotation
root.

Extract frames from a single video:

```bash
python scripts/extract_frames.py --video_path /path/to/video.mp4
```

Run batch inference in one shot:

```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend dpvo \
  --any4d
```

Run step by step:

```bash
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1 --stages detect_track,motion --scheduler_mode wave
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1,2,3 --stages slam --scheduler_mode wave --slam_backend dpvo --any4d
python scripts/batch_infer.py --video_list videos.txt --gpus 0,1 --stages infiller --scheduler_mode wave
```

Visualize:

```bash
python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

Inspect a batch run:

```bash
python tools/ops/analyze_run.py /path/to/run_dir
```

Validate the local environment:

```bash
bash tools/ops/validate_setup.sh
```

## Dataset Pipeline

The official dataset-production path is adapter-driven:

1. `prepare`: dataset-specific preprocessing plus prepared clip state generation
2. `annotate`: optional clip-level language sidecars
3. `infer`: HaWoR stages over the prepared clips
4. `filter`: build-equivalent quality filtering
5. `build`: final WebDataset export
6. `validate`: source/output checks

The `annotate` stage is an external command hook. When configured to call
`lib/annotation/api_annotation.py`, it produces standard sidecars containing `instruction`,
`instruction_num`, `language`, and `hierarchy/global_analysis`; the final
WebDataset build reads those fields into each `*.meta.json`.

Before `prepare`, an optional `clip` block can run raw-video temporal clipping.
Available modes are `none`, `heuristic`, and `api`. `none` is the default and
keeps the original adapter path untouched. `heuristic` uses
`lib/clip/heuristic_clip_config.yaml`; `api` uses
`lib/annotation/api_annotation_with_clip.py` to produce both clipped videos and
annotation sidecars. The two clipping modes redirect the rest of the run to the
new clipped-video directory.

Annotation is independent unless `clip.mode: api` is used. You can run clipping
without annotation by using `clip.mode: heuristic` and setting
`build.require_annotation: false`, or by providing external sidecars in
`paths.annotation_root`. You can run annotation without clipping by keeping
`clip.mode: none` and configuring the normal `annotation.command`. You can skip
both only when the build and validation settings allow empty instruction fields.

Recommended full run:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_single_video.example.yaml
```

Nested BuildAI/HOT3D/FPHA configs remain supported as a migration path. The pipeline keeps an internal prepared clip state on purpose so annotation, filtering, and rebuilding all operate on the same stable clip set.

For details, see [docs/dataset_pipeline.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/dataset_pipeline.md).

## Repository Notes

- `scripts/` keeps user-facing entrypoints and compatibility shims.
- `tools/` contains operational helpers and ad hoc utilities.
- `lib/pipeline/` contains the maintained library code for inference, prepared clip state, filtering, and build/export.
- `deprecated/` contains older one-off scripts and notes that are no longer part of the official path.

If you are unsure where to start, use:

1. [README.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/README.md)
2. [docs/dataset_pipeline.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/dataset_pipeline.md)
3. [docs/repo_map.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/repo_map.md)
