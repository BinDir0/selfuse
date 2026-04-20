#!/bin/bash

# 使用方法: ./launch_vitra_tmux.sh <NODE_RANK>
# NODE_RANK: 0 到 7 (对应你的8台服务器)

if [ -z "$1" ]; then
  echo "Error: Please provide NODE_RANK (0-7)"
  echo "Usage: ./launch_vitra_tmux.sh 0"
  exit 1
fi

NODE_RANK=$1
TOTAL_NODES=5
CPUS_PER_NODE=16
TOTAL_CPUS=$((TOTAL_NODES * CPUS_PER_NODE))

# 计算当前节点的任务范围
START_ID=$((NODE_RANK * CPUS_PER_NODE))
END_ID=$((START_ID + CPUS_PER_NODE - 1))

SESSION_NAME="vitra_node_${NODE_RANK}"
CONDA_ENV="any4d"  # <--- 在这里指定你的环境名

echo "=================================================="
echo "Initializing Tmux Session: $SESSION_NAME"
echo "Target Env: $CONDA_ENV"
echo "Task Range: $START_ID - $END_ID"
echo "=================================================="

# 1. 创建 Session (默认带有窗口 0)
tmux new-session -d -s $SESSION_NAME -n "cpu_${START_ID}"

# 2. 定义一个函数来发送命令，避免重复代码
# 参数1: 窗口目标 (例如 vitra_node_0:5)
# 参数2: CPU_ID
run_task() {
    target=$1
    cpu_id=$2
    
    # 这里的关键是先激活环境，再跑代码
    # C-m 代表回车键
    tmux send-keys -t "$target" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$target" "python /share_data/yifan/projects/depth/Any4D/scripts/check_vitra_reorient.py --cpu_id ${cpu_id} --total_cpus ${TOTAL_CPUS}" C-m
}

# --- 启动第 1 个任务 (在窗口 0) ---
run_task "${SESSION_NAME}:0" "$START_ID"

# --- 循环启动剩余 15 个任务 ---
for ((i=1; i<16; i++)); do
    CURRENT_CPU_ID=$((START_ID + i))
    
    # 创建新窗口
    tmux new-window -t $SESSION_NAME -n "cpu_${CURRENT_CPU_ID}"
    
    # 发送激活命令和运行命令
    run_task "${SESSION_NAME}:${i}" "$CURRENT_CPU_ID"
    
    # 稍微排队一下，避免瞬间并发过高导致 tmux 响应丢失
    # sleep 0.05
done

echo "Done! 64 windows launched."
echo "attach cmd: tmux attach -t ${SESSION_NAME}"