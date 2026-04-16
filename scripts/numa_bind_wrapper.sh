#!/bin/bash
# Per-rank NUMA binding + main-process CPU isolation wrapper.
#
# Invoked by torchrun with --no-python:
#   torchrun ... --no-python bash scripts/numa_bind_wrapper.sh train.py [args...]
#
# Each rank's main process is pinned to a small set of dedicated cores
# (numactl --physcpubind) so NCCL/FSDP collectives are not preempted by
# dataloader workers. Workers fork from main inheriting that narrow
# affinity; src/utils/cpu_affinity.py re-pins them to the per-NUMA worker
# pool exported via DATALOADER_WORKER_CPUS.
#
# Reference: https://man7.org/linux/man-pages/man8/numactl.8.html

set -e

LOCAL_RANK="${LOCAL_RANK:-0}"

# === Per-rank CPU layout (8xA800 dual-socket) ===
# Edit these lines directly when hardware changes; no formulas.
# Verify with `lscpu -e | column -t` (CPU<->NUMA<->core)
# and `cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list`
# (HT pairing — don't assume a fixed offset).
#
# Each RANK_MAIN_N: 2 phys + 2 HT siblings = 4 logical cores reserved
# for that rank's main process (and NCCL/FSDP helper threads it spawns).
RANK_MAIN_0="0,1,64,65"
RANK_MAIN_1="2,3,66,67"
RANK_MAIN_2="4,5,68,69"
RANK_MAIN_3="6,7,70,71"
RANK_MAIN_4="32,33,96,97"
RANK_MAIN_5="34,35,98,99"
RANK_MAIN_6="36,37,100,101"
RANK_MAIN_7="38,39,102,103"

# Worker pool per NUMA: all cores left over after the 4 main reservations
# in that NUMA. Shared by all dataloader workers on that NUMA.
WORKER_POOL_NUMA0="8-31,72-95"
WORKER_POOL_NUMA1="40-63,104-108"
# =================================================

# Autodetect NUMA from sysfs via GPU PCI BDF.
# nvidia-smi outputs BDF as "00000000:17:00.0" (8-char domain);
# sysfs uses "0000:17:00.0" (4-char domain), so strip the first 4 chars.
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$LOCAL_RANK" \
    | tr 'A-Z' 'a-z' | cut -c5-)
NUMA_NODE=$(cat "/sys/bus/pci/devices/${GPU_PCI}/numa_node" 2>/dev/null || echo -1)

# Fallback for kernels/containers where sysfs numa_node is -1 (rare).
# Standard 8-GPU 2-socket layout: GPU 0-3 -> NUMA 0, GPU 4-7 -> NUMA 1.
if [ "$NUMA_NODE" -lt 0 ]; then
    NUMA_NODE=$((LOCAL_RANK / 4))
fi

eval MAIN_CORES=\"\$RANK_MAIN_${LOCAL_RANK}\"
if [ "$NUMA_NODE" = "0" ]; then
    WORKER_POOL="$WORKER_POOL_NUMA0"
else
    WORKER_POOL="$WORKER_POOL_NUMA1"
fi

echo "[rank $LOCAL_RANK] NUMA $NUMA_NODE, main=$MAIN_CORES, workers=$WORKER_POOL" >&2

export DATALOADER_WORKER_CPUS="$WORKER_POOL"

exec numactl --physcpubind="$MAIN_CORES" --membind="$NUMA_NODE" \
    python -u "$@"
