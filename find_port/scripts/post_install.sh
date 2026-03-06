#!/bin/bash
# Post-installation script for Docker container deployment tool
# Creates a desktop shortcut for easy access

set -e

# 获取脚本所在目录的父目录（即项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 桌面路径
DESKTOP_DIR="$HOME/Desktop"
DESKTOP_FILE="$DESKTOP_DIR/start-tele-operation.desktop"

# 检查桌面目录是否存在
if [ ! -d "$DESKTOP_DIR" ]; then
    echo "错误: 桌面目录不存在，请确保使用Ubuntu桌面版本"
    exit 1
fi

# 检查start.py是否存在
if [ ! -f "$PROJECT_DIR/start.py" ]; then
    echo "错误: 找不到 start.py 文件"
    exit 1
fi

# 创建桌面快捷方式
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Start Tele-Operation
Comment=启动遥操作系统
Exec=gnome-terminal -- bash -c "cd $PROJECT_DIR && python3 start.py"
Icon=cheese
Terminal=true
Categories=Development;Utility;
StartupNotify=false
EOF

# 设置桌面文件权限
chmod +x "$DESKTOP_FILE"

# 如果是Gnome桌面，信任该桌面文件
if command -v gio &> /dev/null; then
    gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

# 同时在应用程序菜单中创建快捷方式
LOCAL_APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$LOCAL_APPS_DIR"
LOCAL_DESKTOP_FILE="$LOCAL_APPS_DIR/start-tele-operation.desktop"

cp "$DESKTOP_FILE" "$LOCAL_DESKTOP_FILE"
chmod +x "$LOCAL_DESKTOP_FILE"

echo "桌面快捷方式创建成功: $DESKTOP_FILE"

