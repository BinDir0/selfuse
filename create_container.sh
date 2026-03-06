#!/bin/bash
set -euo pipefail

# if [ $# -lt 2 ]; then
#   echo "Docker Image starter
#   usage: $0 <container_name> <image[:tag]>
#   example: $0 a2d_container psibot/ros-x86-humble-a2d-1-5-0:humble"
#   exit 1
# fi

CONTAINER_NAME="$1"
IMAGE=psibot/ros-x86-pip:humble

# 若已有同名容器，提示并退出（更安全；需要重启时请先手动 rm -f）
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container '${CONTAINER_NAME}' already exists. Remove it first:"
  echo "  docker rm -f ${CONTAINER_NAME}"
  exit 1
fi

# 自动检测和设置显示环境
# 1) 自动检测DISPLAY
if [ -z "${DISPLAY:-}" ]; then
  echo "Auto-detecting DISPLAY..."
  # 尝试常见的DISPLAY值
  for d in :0 :1 :10 :11 :99; do
    if DISPLAY="$d" xdpyinfo >/dev/null 2>&1; then
      export DISPLAY="$d"
      echo "Found DISPLAY: $DISPLAY"
      break
    fi
  done
  
  # 如果还是没找到，尝试从进程中检测
  if [ -z "${DISPLAY:-}" ]; then
    DETECTED_DISPLAY=$(ps aux | grep -E "(X|Xorg|Xwayland)" | grep -v grep | head -1 | sed -n 's/.*:\([0-9]\+\).*/:\1/p' 2>/dev/null || true)
    if [ -n "$DETECTED_DISPLAY" ]; then
      export DISPLAY="$DETECTED_DISPLAY"
      echo "Using DISPLAY: $DISPLAY"
    else
      export DISPLAY=":0"
      echo "Using default DISPLAY: $DISPLAY"
    fi
  fi
fi

# 2) 自动检测和设置XAUTHORITY
XA="${XAUTHORITY:-}"
if [ -z "$XA" ] || [ ! -f "$XA" ]; then
  # 优先查找用户主目录
  XA="${HOME}/.Xauthority"
  if [ ! -f "$XA" ]; then
    # 在用户运行时目录查找
    XA=$(find "/run/user/$(id -u)" -maxdepth 3 -type f -name "Xauth*" 2>/dev/null | head -n1 || true)
  fi
  if [ ! -f "$XA" ]; then
    # 在系统临时目录查找
    XA=$(find /tmp -maxdepth 2 -type f -name "Xauth*" -user "$(whoami)" 2>/dev/null | head -n1 || true)
  fi
  if [ ! -f "$XA" ]; then
    # 最后尝试创建一个新的
    XA="${HOME}/.Xauthority"
    touch "$XA" 2>/dev/null || XA=""
  fi
fi
export XAUTHORITY="$XA"

# 3) 设置X11访问权限
if command -v xhost >/dev/null 2>&1; then
  xhost +local: >/dev/null 2>&1 || true
  xhost +SI:localuser:root >/dev/null 2>&1 || true
  xhost +SI:localuser:"$(whoami)" >/dev/null 2>&1 || true
fi

if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ] && command -v xauth >/dev/null 2>&1; then
  xauth list "$DISPLAY" 2>/dev/null | while read line; do
    echo "$line" | xauth -f "$XAUTHORITY" merge - 2>/dev/null || true
  done
fi

# GPU 检测
GPU_OPTIONS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_OPTIONS+=(--gpus all)
fi

# 可选：把宿主 a2d_sdk 映射进容器（如需请取消注释并改路径）
# HOST_A2D_SDK="/data/A2D_docker_haoyi/a2d_sdk"
# SDK_MOUNT_OPT=(-v "${HOST_A2D_SDK}:/root/a2d_sdk")
SDK_MOUNT_OPT=()

# X授权文件挂载
XAUTH_MOUNT_OPT=()
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
  XAUTH_MOUNT_OPT=(-v "$XAUTHORITY":/root/.Xauthority:ro)
fi

# 显示相关挂载
EXTRA_DISPLAY_MOUNTS=(
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
)

if [ -d "/dev/dri" ]; then
  EXTRA_DISPLAY_MOUNTS+=(--device /dev/dri)
fi

if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -S "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}" ]; then
  EXTRA_DISPLAY_MOUNTS+=(-v "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}:/tmp/${WAYLAND_DISPLAY}:rw")
fi

# 启动容器
echo "Starting container..."
docker run -d \
  ${GPU_OPTIONS[@]} \
  --privileged \
  --network=host \
  --ipc=host \
  --name "${CONTAINER_NAME}" \
  -w /root/workspace/legendvla-inference \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e DISPLAY="${DISPLAY}" \
  -e XAUTHORITY=/root/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -e QT_GRAPHICSSYSTEM=native \
  -e QT_LOGGING_RULES="qt.qpa.xcb.xcb_error.debug=false" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
  -e WGPU_BACKEND=gl \
  -e LIBGL_ALWAYS_INDIRECT=0 \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
  -e XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}" \
  -e DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e VENVPYTHONPATH=/usr/bin/python3 \
  -e ROS_DOMAIN_ID=42 \
  -e ROS_LOCALHOST_ONLY=1 \
  -e PROJECT_DIR=/root/workspace/legendvla-inference \
  ${EXTRA_DISPLAY_MOUNTS[@]} \
  -v /home/user/Documents/LegendVLA-Inference:/root/workspace/legendvla-inference \
  ${XAUTH_MOUNT_OPT[@]} \
  ${SDK_MOUNT_OPT[@]} \
  "${IMAGE}" \
  bash -lc '
    set -e
    echo "Container started, DISPLAY=$DISPLAY"
    # 关闭防火墙
    # 说明：
    # - 当前容器使用 --network=host，与宿主机共用同一网络命名空间；
    #   因此下述操作等同于在“宿主机”上关闭或清空防火墙规则。
    # - 如果宿主机本身未启用防火墙，本段不会产生额外影响。
    # 策略：尝试禁用 ufw/firewalld，并 flush nftables/iptables（含 IPv4/IPv6），所有命令失败均忽略错误。
    {
      if command -v ufw >/dev/null 2>&1; then
        ufw disable || true
      fi
      if command -v firewall-cmd >/dev/null 2>&1; then
        firewall-cmd --state >/dev/null 2>&1 && firewall-cmd --set-default-zone=trusted || true
        firewall-cmd --permanent --set-default-zone=trusted || true
        firewall-cmd --reload || true
      fi
      if command -v nft >/dev/null 2>&1; then
        nft flush ruleset || true
      fi
      if command -v iptables >/dev/null 2>&1; then
        iptables -F || true
        iptables -t nat -F || true
        iptables -t mangle -F || true
        iptables -X || true
      fi
      if command -v ip6tables >/dev/null 2>&1; then
        ip6tables -F || true
        ip6tables -t nat -F || true
        ip6tables -t mangle -F || true
        ip6tables -X || true
      fi
    } || true
    trap : TERM INT
    sleep infinity & wait
  '

# 验证容器启动
sleep 2
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' is running"
else
  echo "Failed to start container '${CONTAINER_NAME}'"
  exit 1
fi

# 打开交互壳
echo "Opening shell..."
docker exec -it "${CONTAINER_NAME}" bash
