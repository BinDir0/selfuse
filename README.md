# LegendVLA-Inference

## 环境配置

1. **准备工作**：安装 Docker 并下载[镜像文件](https://drive.google.com/file/d/1ygYfRh0LoD9PB28IjDLI68VKSwroVUA3/view?usp=sharing)。
2. **导入镜像**：
```bash
docker load -i teleop.tar
```
3. **修改宿主机的UDP缓冲区参数**：
```bash
# 创建独立的 ROS 2 优化配置文件
sudo sh -c 'cat > /etc/sysctl.d/60-ros2-realsense.conf <<EOF
net.core.rmem_max=2147483647
net.core.wmem_max=2147483647
net.core.rmem_default=2147483647
net.core.wmem_default=2147483647
EOF'

# 立即应用
sudo sysctl --system
```
4. **创建容器**：修改 `create_container.sh` 第 139 行的宿主机路径为本地项目目录，然后执行：
```bash
./create_container.sh legendvla-inference
```
5. **安装依赖**：在container中执行：
```bash
pip install -r requirements.txt
sudo apt install vim
sudo apt install tree
sudo apt-get update && sudo apt-get install ros-$ROS_DISTRO-rmw-cyclonedds-cpp -y
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc && source ~/.bashrc
```
6. **设置命令缩写**：在container中执行：
```bash
echo '
# ROS2 Aliases
alias cb="colcon build"
alias s="source install/setup.bash"
alias camera="ros2 launch camera camera_node.launch.py"
alias hand="ros2 launch hand dual_hands.launch.py"
alias arm="ros2 launch arm dual_arms.launch.py"
alias interface="ros2 launch model_interface model_interface.launch.py"
alias mock="ros2 launch mock_robot_data mock_data.launch.py"
sudo ip link set lo multicast on
' >> ~/.bashrc && source ~/.bashrc
```


## 推理流程

### 1. 初始启动
1. **启动服务器**：首先在GPU服务器上启动模型推理服务器端。

在训练目录下，修改 `src/config/experiment/inference.yaml` 中如下字段：
- `model_config_path` 为期望的模型 config 路径（可以直接在 checkpoint 文件夹下找到）
- `checkpoint_path` 为期望的模型 checkpoint 路径
- `serving` 为服务器 ip、端口等。

在训练代码主目录下，执行：

```
./scripts/run_server.sh
```
   
2. **启动客户端**：

（1）在宿主机进入本文件夹的`assets/find_port`，执行`python find.py`，把左右手的端口填入`src/hand/launch/dual_hands.launch.py`。

（2）把四组手眼标定文件放到本文件夹下的一个统一文件夹中，比如：

```bash
root@zyf-4090:~/workspace/legendvla-inference/examples# tree -L 2
.
└── calibration_outputs
    ├── xiaozi1-chest-cam-left-arm-20251219-221432
    ├── xiaozi1-chest-cam-right-arm-20251219-222808
    ├── xiaozi1-head-cam-left-arm-20251219-220859
    └── xiaozi1-head-cam-right-arm-20251219-222411

5 directories, 0 files
```
并把容器中这个文件夹路径（此例中为`/root/workspace/legendvla-inference/examples/calibration_outputs`）写到`src/model_interface/launch/model_interface.launch.py`中的`calibration_path`项。

（3）根据模型服务端所在的服务器，设置`src/model_interface/launch/model_interface.launch.py`中的`model_server_port`项。对应关系如下表：

| 服务器 | 对应端口号 | 服务器 | 对应端口号 | 服务器 | 对应端口号 | 服务器 | 对应端口号 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| pro-01 | 18020 | pro-04 | 18021 | pro-05 | 18022 | pro-07 | 18023 |
| pro-09 | 18024 | pro-10 | 18025 | pro-11 | 18026 | pro-14 | 18033 |
| pro-15 | 18034 | pro-16 | 18035 | pro-17 | 18036 | pro-18 | 18037 |
| pro-38 | 18028 | pro-48 | 18029 | pro-49 | 18030 | pro-50 | 18031 
| pro-51 | 18032 |  |  |  |  |  |  |

（4）在宿主机进入本文件夹的`assets/`，执行`python host_interaction_server.py`，启动用户指令输入终端。

（5）在docker容器进入本文件夹，执行构建和启动命令，启动相机、机械臂、机械手及推理客户端节点。

### 2. 任务循环 (Loop)
系统按以下流程循环执行任务：
1. **配置任务**：邀请用户在终端输入语言指令，并选择运行模式（`deploy` 或 `debug`）。
2. **触发开始**：踩下脚踏板 1，系统播放“开始执行”提示音并启动任务。
3. **过程控制**：
    - `deploy` 模式：连续执行。期间按空格键或踩下脚踏板 2 可实现 暂停/继续（播放“暂停/继续执行”提示音）。
    - `debug` 模式：单步执行。每按一次空格键或踩下脚踏板 2，机器人向前执行一步预测动作。
4. **归位与重置**：踩下脚踏板 3，系统执行归位动作并结束当前任务（播放“系统归位”提示音）。
5. **进入下一轮**：系统会再次提示输入新的指令和模式。
