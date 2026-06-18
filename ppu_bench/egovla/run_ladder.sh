#!/bin/bash
# Drive EgoVLA's own verification ladder on the PPU, safe-baseline first.
# Run from the EgoVLA repo root inside the PPU vendor container, AFTER applying
# the edits in PORTING.md. Each stage is non-fatal so one failure doesn't stop
# the sweep; check the per-stage PASS/FAIL and the JSON/PNG reports it writes.
#
#   EGOVLA_REPO=~/code/EgoVLA bash run_ladder.sh
#
# Optional paths (skip data-dependent checks if unset):
#   NORMALIZER=~/normalizers/.../normalizer.pkl
#   MODEL_PATH=~/checkpoints/Qwen3-VL-2B-Instruct
#   CKPT=~/checkpoints/update_step_100000
#   CONFIG=src/config/experiment/legendvla_qwen3_vl.yaml

set -u
REPO="${EGOVLA_REPO:-$PWD}"
CONFIG="${CONFIG:-src/config/experiment/legendvla_qwen3_vl.yaml}"
cd "$REPO" || { echo "bad EGOVLA_REPO=$REPO"; exit 1; }

# --- Critical: do NOT inherit NVIDIA-fabric NCCL env (would hang on PPU CCL) ---
unset NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_IB_GID_INDEX
export EGOVLA_DISABLE_NV_NCCL=1                                       # honored by patched debug_start.sh
export EGOVLA_DEVICE_PEAK_TFLOPS="${EGOVLA_DEVICE_PEAK_TFLOPS:-148}"  # PPU bf16 peak -> correct MFU%
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=disabled

stage() { echo; echo "==== $* ===="; }
run()   { echo "+ $*"; "$@"; echo "  -> exit $?"; }

stage "0. environment"
run python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available(),'n',torch.cuda.device_count())"

stage "1. pretrain_verification (no-data phases: forward/loss/flow/convergence)"
run python -m src.tests.pretrain_verification.run_all --phases 3 4 6 8 --skip-visual
if [ -n "${NORMALIZER:-}" ]; then
  stage "1b. pretrain_verification data phases (0,1)"
  run python -m src.tests.pretrain_verification.run_all --phases 0 1 --skip-visual \
      --config-path "$CONFIG" --normalizer-path "$NORMALIZER"
fi

stage "2. full_chain_verification CPU parts (data/collator/flow-math)"
run python -m src.tests.full_chain_verification.run_all --parts 1 2 4 --skip-visual

stage "3. full_chain_verification GPU parts (backbone/expert/inference-parity)"
run python -m src.tests.full_chain_verification.run_all --parts 3 5 6 --skip-visual --config-path "$CONFIG"

stage "4. determinism (batch invariants)"
run python -m pytest src/tests/test_batch_invariants.py -q

stage "5. gradient checkpointing + qwen3vl components"
run python -m pytest src/tests/test_gradient_checkpointing.py src/tests/test_qwen3vl_components.py -q

stage "6. single-card training smoke (10 steps; PPU overlay = compile OFF + sdpa)"
# Uses the legendvla_qwen3_vl_ppu overlay from ppu.patch (safe baseline baked in).
run torchrun --nnodes=1 --nproc_per_node=1 --master_port=29555 train.py \
    experiment=legendvla_qwen3_vl_ppu \
    training.max_train_steps=10 training.eval_every=20 \
    dataloader.loader.num_workers=0 \
    dataloader.loader.persistent_workers=False \
    dataloader.loader.prefetch_factor=null

if [ -n "${CKPT:-}" ]; then
  stage "7. checkpoint load + inference parity"
  run python -m src.tests.full_chain_verification.part7_e2e_training \
      --config-path "$CONFIG" ${NORMALIZER:+--normalizer-path "$NORMALIZER"} --checkpoint-path "$CKPT"
fi

echo; echo "==== ladder done. Review outputs/{pretrain,full_chain}_verification/ ===="
