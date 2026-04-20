#!/bin/bash

# 使用方法: ./launch_vitra.sh <NODE_RANK>
# NODE_RANK: 当前服务器是第几台 (0, 1, 2, ... 7)

if [ -z "$1" ]; then
  echo "Error: Please provide NODE_RANK (0-7)"
  echo "Usage: ./launch_vitra.sh 0"
  exit 1
fi

NODE_RANK=$1
TOTAL_NODES=8
CPUS_PER_NODE=64

# 计算总 CPU 数 (应该等于 512)
TOTAL_CPUS=$((TOTAL_NODES * CPUS_PER_NODE))

# 计算当前节点的任务范围
START_ID=$((NODE_RANK * CPUS_PER_NODE))
END_ID=$((START_ID + CPUS_PER_NODE - 1))

SESSION_NAME="vitra_check_node_${NODE_RANK}"

echo "=================================================="
echo "Launching VITRA Check on Node ${NODE_RANK}"
echo "Task Range: ${START_ID} to ${END_ID}"
echo "Total CPUS: ${TOTAL_CPUS}"
echo "Tmux Session: ${SESSION_NAME}"
echo "=================================================="

# 1. 创建一个新的后台 Tmux session
tmux new-session -d -s $SESSION_NAME

# 2. 循环启动进程
# 我们不创建 64 个 window，那样太慢了。
# 我们创建 4 个 window，每个 window 跑 16 个后台任务 (nohup 模式)，
# 或者更简单：在一个 window 里循环后台启动所有任务。

# 方案：在 Tmux 的第 0 个窗口中，批量后台启动所有 Python 进程
# 这样你只需要 attach 进去看 htop 即可，不用切 64 个窗口

CMD="for i in \$(seq $START_ID $END_ID); do \
    echo \"Starting worker \$i ...\"; \
    python check_vitra_multi_cpu.py --cpu_id \$i --total_cpus $TOTAL_CPUS > /dev/null 2>&1 & \
    sleep 0.5; \
done; wait"

# 发送命令到 tmux session
tmux send-keys -t $SESSION_NAME "$CMD" C-m

echo "All workers launched inside tmux session '${SESSION_NAME}'."
echo "Use 'tmux attach -t ${SESSION_NAME}' to view progress."
echo "Use 'htop' to check CPU usage."