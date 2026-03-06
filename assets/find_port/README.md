# Docker 容器部署工具

专为部署分支设计的轻量级容器管理工具，专注于从 DockerHub 拉取和部署 Jenkins 编译的镜像。

## 核心特性

**纯部署导向**：专为生产环境设计，只支持镜像拉取和容器部署，无开发构建功能

**智能设备映射**：自动识别和映射 USB 设备（PSI 手套、灵巧手等），支持接口号和序列号精确匹配

**声明式配置**：YAML 驱动的配置管理，支持版本控制和团队协作

**零停机部署**：自动停止旧容器，启动新容器，确保服务连续性

## 快速开始

### 首次使用

```bash
# 安装依赖
pip install -r requirements.txt

# 登录私有 DockerHub（如需要）
bash scripts/login.sh

# 创建桌面快捷方式（可选）
bash scripts/post_install.sh
```

### 每次部署

```bash
# 标准部署
python3 start.py

# 强制重新拉取镜像并部署
python3 start.py --force

# 预览部署计划（不执行）
python3 start.py --dry-run
```

## 配置文件

### 主配置文件

配置文件位于 `config/deploy.yaml`，`config.yaml` 是其软链接。

```yaml
# 容器基础配置
container:
  name: "ros-0.8rc"
  image:
    repository: "dockerhub.cn-northwest-4.ksyunkcr.com/psibot/a2d-tele/x86/release/0.8.1rc1"
    tag: "latest"
  command: "bash"

# 硬件资源配置
resources:
  gpu:
    enabled: false
    options: "--gpus all"
  network: "host"
  privileged: true

# 挂载点配置
volumes:
  - type: "bind"
    source: "/tmp/.X11-unix"
    target: "/tmp/.X11-unix"
    enabled: true
    options: "rw"

# USB设备自动映射
devices:
  - name: "psi_glove_left"
    enabled: true
    usb_vendor: "0483"
    usb_product: "5739"
    container_path: "/dev/psi_glove_left"
  
  - name: "ry_hand_left"
    enabled: true
    usb_vendor: "1a86"
    usb_product: "55d5"
    usb_interface: "00"
    container_path: "/dev/ry_hand_left"

# 环境变量配置
environment:
  ROS_DOMAIN_ID: 0
  DISPLAY: ":0"
  auto_detect_display: true
```

### 配置项说明

#### 容器配置
- `name`: 容器名称
- `image.repository`: 镜像仓库地址
- `image.tag`: 镜像标签
- `exec_mode`: 执行模式
  - `interactive`: 交互式模式，直接执行command（用于bash等）
  - `command`: 命令模式，通过bash -c执行，支持环境变量加载和shell特性
- `command`: 容器启动后执行的命令
  - 在`command`模式下，可以使用`source`加载环境变量，然后执行ROS2等命令
  - 示例：`"source /opt/psi/setup.bash && ros2 launch scripts_pack start_gui.launch.py"`

#### 设备映射
- `name`: 设备名称
- `enabled`: 是否启用设备映射
- `usb_vendor`: USB 厂商 ID
- `usb_product`: USB 产品 ID
- `usb_interface`: USB 接口号（可选，用于区分同一设备的多个接口）
- `usb_serial`: USB 序列号（可选，用于区分相同型号的设备）
- `container_path`: 容器内设备路径

#### 挂载点配置
- `source`: 宿主机路径
- `target`: 容器内路径
- `enabled`: 是否启用挂载
- `options`: 挂载选项（rw/ro）

## 启动参数

```bash
# 预览模式 - 查看执行计划
python3 start.py --dry-run

# 强制重建 - 重新拉取镜像并创建容器
python3 start.py --force

# 自定义配置 - 使用指定配置文件
python3 start.py --config production.yaml

# 详细日志 - 显示调试信息
python3 start.py --verbose

# 组合使用
python3 start.py --config dev.yaml --force --verbose
```

## 使用场景示例

### 场景一：交互式bash终端
```yaml
container:
  exec_mode: "interactive"
  command: "bash"
```
启动后会进入容器的bash终端，可以手动执行命令。

### 场景二：自动启动ROS2应用
```yaml
container:
  exec_mode: "command"
  command: "source /opt/psi/setup.bash && ros2 launch scripts_pack start_gui.launch.py"
```
启动后会自动加载ROS2环境并执行launch文件。

### 场景三：执行多个命令
```yaml
container:
  exec_mode: "command"
  command: "source /opt/ros/humble/setup.bash && source /opt/psi/setup.bash && ros2 run my_package my_node"
```
可以使用`&&`连接多个命令，按顺序执行。

## 工作原理

**部署流程**：
1. 检查 Docker 环境
2. 读取配置文件获取镜像信息
3. 检查本地镜像是否存在
4. 如果不存在或使用 `--force`，则拉取镜像
5. 停止并删除现有容器（如果存在）
6. 创建新容器并映射所有配置的设备
7. 进入容器

**设备映射**：
- 自动扫描系统中的 USB 设备
- 根据厂商 ID 和产品 ID 匹配设备
- 支持通过接口号和序列号进行精确匹配
- 自动映射到容器内的指定路径

**故障处理**：
- 镜像拉取失败时停止部署
- 容器创建失败时提供详细错误信息
- 支持 `Ctrl+C` 中断操作

## 文件结构

```
docker/
├── start.py           # 主启动脚本（集成所有功能）
├── config.yaml        # 软链接 -> config/deploy.yaml
├── config/
│   └── deploy.yaml    # 主配置文件
├── scripts/
│   └── login.sh       # DockerHub 登录脚本
├── requirements.txt   # Python 依赖
└── README.md          # 本文档
```

## 依赖要求

- Python 3.6+
- Docker
- PyYAML

## 注意事项

- 此工具专为部署分支设计，不支持镜像构建功能
- 需要 Docker 环境已正确安装和配置
- USB 设备映射需要适当的权限
- 建议在生产环境中使用 `--dry-run` 预览部署计划