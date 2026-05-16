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

cd thirdparty/DPVO && pip install . --no-build-isolation && cd ../..
cd thirdparty/DROID-SLAM && python setup.py install && cd ../..
```

If `torch-scatter==2.1.2` from `requirements.txt` does not match your CUDA/PyTorch
wheel, reinstall the matching PyG wheel for your local CUDA/PyTorch pair.

### 2. Any4D environment

```bash
conda create -n any4d python=3.12 -y
conda activate any4d

# Install the PyTorch wheel appropriate for this machine. cu121 is the tested
# default for this repo; use the official PyTorch selector if your driver stack
# needs a different CUDA runtime.
pip install torch==2.6.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd thirdparty/Any4D
pip install -e .
cd ../..

# The SLAM subprocess still launches this repo's batch_infer.py, so the Any4D
# env also needs pipeline-control, frame/scale, and DPVO dependencies.
pip install joblib tqdm natsort opencv-python-headless pycocotools evo pytorch-minimize
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
