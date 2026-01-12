#!/bin/bash

# SSH config 同步和免密登录配置脚本
# 功能：
# 1. 将当前 config 文件同步到所有远程主机
# 2. 在所有主机之间（包括本地）配置免密登录

CONFIG_FILE="$HOME/.ssh/config"
PUB_KEY_FILE="$HOME/.ssh/id_rsa.pub"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查必要文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config 文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$PUB_KEY_FILE" ]; then
    log_error "公钥文件不存在: $PUB_KEY_FILE"
    log_info "正在生成 SSH 密钥..."
    ssh-keygen -t rsa -f "$HOME/.ssh/id_rsa" -N "" -q
    if [ ! -f "$PUB_KEY_FILE" ]; then
        log_error "密钥生成失败"
        exit 1
    fi
fi

# 解析 config 文件，提取所有主机名
extract_hosts() {
    grep -E "^Host " "$CONFIG_FILE" | awk '{print $2}' | grep -v "^\*$"
}

log_info "开始同步 SSH config 和配置免密登录..."

# 提取所有主机
HOSTS=($(extract_hosts))

if [ ${#HOSTS[@]} -eq 0 ]; then
    log_error "未找到任何主机配置"
    exit 1
fi

log_info "找到 ${#HOSTS[@]} 台主机: ${HOSTS[*]}"

# 提示输入密码（如果需要）
log_step "如果需要密码进行首次连接，请输入密码（直接回车跳过如果已配置免密）:"
read -s SSH_PASSWORD
echo ""

# 检查是否安装了 sshpass
if [ -n "$SSH_PASSWORD" ]; then
    if command -v sshpass >/dev/null 2>&1; then
        USE_SSHPASS=true
        log_info "检测到 sshpass，将使用密码进行连接"
    else
        log_error "未安装 sshpass，无法使用密码自动登录"
        log_info "请安装 sshpass: sudo apt-get install sshpass 或 sudo yum install sshpass"
        log_info "或者先手动配置到第一台主机的免密登录，然后重新运行脚本"
        exit 1
    fi
else
    USE_SSHPASS=false
    log_info "未输入密码，将使用已配置的免密登录"
fi

# SSH 连接函数（支持密码）
ssh_with_password() {
    local host=$1
    shift
    if [ "$USE_SSHPASS" = true ] && [ -n "$SSH_PASSWORD" ]; then
        sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$host" "$@" 2>/dev/null
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$host" "$@" 2>/dev/null
    fi
}

# SCP 复制函数（支持密码）
scp_with_password() {
    local src=$1
    local dst=$2
    if [ "$USE_SSHPASS" = true ] && [ -n "$SSH_PASSWORD" ]; then
        sshpass -p "$SSH_PASSWORD" scp -o StrictHostKeyChecking=no -o ConnectTimeout=5 -q "$src" "$dst" 2>/dev/null
    else
        scp -o StrictHostKeyChecking=no -o ConnectTimeout=5 -q "$src" "$dst" 2>/dev/null
    fi
}

# SSH-copy-id 函数（支持密码）- 手动实现
ssh_copy_id_with_password() {
    local target_host=$1
    local pub_key_file=${2:-"$PUB_KEY_FILE"}
    
    # 读取公钥
    if [ ! -f "$pub_key_file" ]; then
        return 1
    fi
    
    local pub_key=$(cat "$pub_key_file")
    local key_fingerprint=$(echo "$pub_key" | cut -d' ' -f2)
    
    # 使用 sshpass 或普通 ssh 执行命令
    if [ "$USE_SSHPASS" = true ] && [ -n "$SSH_PASSWORD" ]; then
        sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$target_host" "
            mkdir -p ~/.ssh && chmod 700 ~/.ssh
            touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
            if ! grep -q '$key_fingerprint' ~/.ssh/authorized_keys 2>/dev/null; then
                echo '$pub_key' >> ~/.ssh/authorized_keys
            fi
        " >/dev/null 2>&1
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$target_host" "
            mkdir -p ~/.ssh && chmod 700 ~/.ssh
            touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
            if ! grep -q '$key_fingerprint' ~/.ssh/authorized_keys 2>/dev/null; then
                echo '$pub_key' >> ~/.ssh/authorized_keys
            fi
        " >/dev/null 2>&1
    fi
}

# 步骤1: 将 config 文件复制到所有远程主机
log_step "步骤1: 同步 config 文件到所有远程主机..."
for host in "${HOSTS[@]}"; do
    log_info "正在同步 config 到 $host..."
    if scp_with_password "$CONFIG_FILE" "$host:~/.ssh/config"; then
        log_info "✓ $host config 同步成功"
    else
        log_warn "✗ $host config 同步失败"
    fi
done

# 步骤2: 从本地 ssh-copy-id 到所有主机
log_step "步骤2: 从本地配置免密登录到所有主机..."
for host in "${HOSTS[@]}"; do
    log_info "正在配置本地到 $host 的免密登录..."
    if ssh_copy_id_with_password "$host"; then
        log_info "✓ 本地到 $host 免密登录配置完成"
    else
        log_warn "✗ 本地到 $host 免密登录配置失败"
    fi
done

# 步骤3: 从每个主机 ssh-copy-id 到其他所有主机
log_step "步骤3: 从每个主机配置免密登录到其他所有主机..."
for source_host in "${HOSTS[@]}"; do
    log_info "正在从 $source_host 配置免密登录..."
    
    # 确保源主机有公钥
    ssh_with_password "$source_host" "mkdir -p ~/.ssh && chmod 700 ~/.ssh" || continue
    ssh_with_password "$source_host" "test -f ~/.ssh/id_rsa.pub || ssh-keygen -t rsa -f ~/.ssh/id_rsa -N '' -q" || continue
    
    # 获取源主机的公钥
    source_pub_key=$(ssh_with_password "$source_host" "cat ~/.ssh/id_rsa.pub")
    if [ -z "$source_pub_key" ]; then
        log_warn "无法获取 $source_host 的公钥，跳过"
        continue
    fi
    
    # 从本地将源主机的公钥复制到所有目标主机（包括自己）
    # 这样可以避免在远程主机上需要密码的问题
    for target_host in "${HOSTS[@]}"; do
        log_info "  配置 $source_host -> $target_host..."
        key_fingerprint=$(echo "$source_pub_key" | cut -d' ' -f2)
        
        # 检查目标主机是否已有该公钥
        if ssh_with_password "$target_host" "grep -q '$key_fingerprint' ~/.ssh/authorized_keys 2>/dev/null"; then
            log_info "  ✓ $source_host -> $target_host 已存在，跳过"
            continue
        fi
        
        # 添加公钥到目标主机
        if ssh_with_password "$target_host" "
            mkdir -p ~/.ssh && chmod 700 ~/.ssh
            touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
            echo '$source_pub_key' >> ~/.ssh/authorized_keys
        "; then
            log_info "  ✓ $source_host -> $target_host 配置完成"
        else
            log_warn "  ✗ $source_host -> $target_host 配置失败"
        fi
    done
done

log_info ""
log_info "=========================================="
log_info "所有操作完成！"
log_info "已同步 config 文件到所有主机"
log_info "已配置所有主机之间的免密登录"
log_info "=========================================="
