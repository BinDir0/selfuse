#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import logging
import glob
from pathlib import Path
from functools import reduce
from operator import getitem
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Any, Iterator
from contextlib import contextmanager
from enum import Enum

try:
    import yaml
except ImportError:
    print("错误: 缺少 yaml 模块，请运行: pip install PyYAML", file=sys.stderr)
    sys.exit(1)


class ContainerStatus(Enum):
    """容器状态枚举"""
    RUNNING = "running"
    STOPPED = "stopped"
    NOT_EXISTS = "not_exists"
    ERROR = "error"


@dataclass(frozen=True)
class VolumeConfig:
    """挂载点配置"""
    source: str
    target: str
    enabled: bool = True
    options: str = "rw"
    
    @property
    def expanded_source(self) -> str:
        """展开用户路径"""
        return os.path.expanduser(self.source)


@dataclass(frozen=True)
class DeviceConfig:
    """设备配置"""
    name: str
    path: Optional[str] = None
    enabled: bool = True
    options: str = "rw"
    # USB设备相关字段
    usb_vendor: Optional[str] = None
    usb_product: Optional[str] = None
    usb_interface: Optional[str] = None
    usb_serial: Optional[str] = None
    container_path: Optional[str] = None
    
    @property
    def is_usb_device(self) -> bool:
        """是否为USB设备"""
        return self.usb_vendor is not None and self.usb_product is not None
    
    @property
    def effective_container_path(self) -> str:
        """有效的容器路径"""
        return self.container_path or f"/dev/{self.name}"


@dataclass(frozen=True)
class ResourceConfig:
    """资源配置"""
    network: str = "bridge"
    privileged: bool = False
    gpu_enabled: bool = False
    gpu_options: str = "--gpus all"


@dataclass(frozen=True)
class EnvironmentConfig:
    """环境变量配置"""
    ros_domain_id: int = 0
    ros_localhost_only: int = 0
    display: str = ":0"
    auto_detect_display: bool = True
    fallback_displays: List[str] = field(default_factory=lambda: [":0", ":1", ":10", ":11", ":99"])
    pulse_server: Optional[str] = None
    
    def get_display(self) -> str:
        """获取显示环境变量"""
        if self.auto_detect_display:
            return os.environ.get('DISPLAY', self.display)
        return self.display
    
    def get_pulse_server(self) -> str:
        """获取 PULSE_SERVER 环境变量"""
        if self.pulse_server:
            return self.pulse_server
        # 默认自动生成
        uid = os.getuid()
        return f"unix:/run/user/{uid}/pulse/native"


@dataclass(frozen=True)
class ContainerConfig:
    """容器配置"""
    name: str
    image_repository: str
    image_tag: str
    command: str = "bash"
    restart: str = "no"


@dataclass
class DockerCommand:
    """Docker命令构建器"""
    base_cmd: List[str] = field(default_factory=lambda: ['docker', 'run', '-d'])
    name: Optional[str] = None
    image: Optional[str] = None
    network: Optional[str] = None
    privileged: bool = False
    gpu_options: Optional[str] = None
    volumes: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    
    def build(self) -> List[str]:
        """构建完整的Docker命令"""
        cmd = self.base_cmd.copy()
        
        if self.name:
            cmd.extend(['--name', self.name])
        
        if self.network:
            cmd.extend(['--network', self.network])
        
        if self.privileged:
            cmd.append('--privileged')
        
        if self.gpu_options:
            cmd.extend(self.gpu_options.split())
        
        cmd.extend(self.volumes)
        cmd.extend(self.devices)
        cmd.extend(self.environment)
        
        if self.image:
            cmd.extend([self.image, 'tail', '-f', '/dev/null'])
        
        return cmd


class TTYDeviceFinder:
    """TTY设备查找器，用于根据USB设备信息查找对应的tty设备"""
    
    def __init__(self):
        self.tty_devices = self._scan_tty_devices()
    
    def _scan_tty_devices(self) -> List[str]:
        """扫描系统中的所有tty设备"""
        tty_devices = []
        for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
            tty_devices.extend(glob.glob(pattern))
        return sorted(tty_devices)
    
    def _get_device_info(self, tty_device: str) -> Dict[str, str]:
        """获取tty设备的详细信息"""
        try:
            result = subprocess.run([
                'udevadm', 'info', '-q', 'property', '-n', tty_device
            ], capture_output=True, text=True, check=True)
            
            device_info = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    device_info[key] = value
            
            return device_info
        except subprocess.CalledProcessError:
            return {}
        except Exception:
            return {}
    
    def find_by_usb_id(self, vendor_id: str, product_id: str, 
                      usb_interface: Optional[str] = None, 
                      usb_serial: Optional[str] = None) -> Optional[str]:
        """
        根据USB设备的vendor_id和product_id查找对应的tty设备
        支持通过USB接口号或序列号进行更精确的识别
        """
        for tty_device in self.tty_devices:
            device_info = self._get_device_info(tty_device)
            
            # 基本匹配：vendor和product ID
            if (device_info.get('ID_VENDOR_ID') == vendor_id and 
                device_info.get('ID_MODEL_ID') == product_id):
                
                # 如果指定了USB接口号，进行精确匹配
                if usb_interface is not None:
                    if device_info.get('ID_USB_INTERFACE_NUM') != usb_interface:
                        continue
                
                # 如果指定了序列号，进行精确匹配
                if usb_serial is not None:
                    if device_info.get('ID_SERIAL') != usb_serial:
                        continue
                
                return tty_device
        
        return None


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @contextmanager
    def _safe_file_operation(self):
        """安全的文件操作上下文管理器"""
        try:
            yield
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except PermissionError:
            raise PermissionError(f"没有权限读取配置文件 {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"读取配置文件失败: {e}")
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with self._safe_file_operation():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError("配置文件格式无效：根节点必须是字典")
        
        return config
    
    def _get_config_value(self, config: Dict[str, Any], path: str, default: Any = "") -> Any:
        """获取配置值"""
        try:
            keys = path.strip('.').split('.')
            return reduce(getitem, keys, config)
        except (KeyError, TypeError):
            return default
    
    def parse_volumes(self, config: Dict[str, Any]) -> List[VolumeConfig]:
        """解析挂载点配置"""
        volumes = []
        volume_configs = config.get('volumes', [])
        
        for vol_config in volume_configs:
            if isinstance(vol_config, dict):
                volumes.append(VolumeConfig(
                    source=vol_config.get('source', ''),
                    target=vol_config.get('target', ''),
                    enabled=vol_config.get('enabled', True),
                    options=vol_config.get('options', 'rw')
                ))
        
        return volumes
    
    def parse_devices(self, config: Dict[str, Any]) -> List[DeviceConfig]:
        """解析设备配置"""
        devices = []
        device_configs = config.get('devices', [])
        
        for dev_config in device_configs:
            if isinstance(dev_config, dict):
                devices.append(DeviceConfig(
                    name=dev_config.get('name', 'GPU'),
                    path=dev_config.get('path'),
                    enabled=dev_config.get('enabled', True),
                    options=dev_config.get('options', 'rw'),
                    usb_vendor=dev_config.get('usb_vendor'),
                    usb_product=dev_config.get('usb_product'),
                    usb_interface=dev_config.get('usb_interface'),
                    usb_serial=dev_config.get('usb_serial'),
                    container_path=dev_config.get('container_path')
                ))
        
        return devices
    
    def parse_resources(self, config: Dict[str, Any]) -> ResourceConfig:
        """解析资源配置"""
        resources = config.get('resources', {})
        gpu = resources.get('gpu', {})
        
        return ResourceConfig(
            network=resources.get('network', 'bridge'),
            privileged=resources.get('privileged', False),
            gpu_enabled=gpu.get('enabled', False),
            gpu_options=gpu.get('options', '--gpus all')
        )
    
    def parse_environment(self, config: Dict[str, Any]) -> EnvironmentConfig:
        """解析环境变量配置"""
        env = config.get('environment', {})
        
        return EnvironmentConfig(
            display=env.get('DISPLAY', ':0'),
            auto_detect_display=env.get('auto_detect_display', True),
            fallback_displays=env.get('fallback_displays', [':0', ':1', ':10', ':11', ':99']),
            pulse_server=env.get('PULSE_SERVER')
        )
    
    def parse_container(self, config: Dict[str, Any]) -> ContainerConfig:
        """解析容器配置"""
        container = config.get('container', {})
        image = container.get('image', {})
        
        return ContainerConfig(
            name=container.get('name', 'docker-container'),
            image_repository=image.get('repository', 'ubuntu'),
            image_tag=image.get('tag', 'latest'),
            command=container.get('command', 'bash'),
            restart=container.get('restart', 'no')
        )


class DeviceMapper:
    """设备映射器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.finder = TTYDeviceFinder()
        self._dry_run_mode = False
    
    def map_usb_device(self, device: DeviceConfig) -> Optional[str]:
        """映射USB设备"""
        if not device.is_usb_device:
            return None
        
        try:
            return self.finder.find_by_usb_id(
                device.usb_vendor,
                device.usb_product,
                device.usb_interface,
                device.usb_serial
            )
        except Exception as e:
            self.logger.error(f"设备查找失败: {e}")
            return None
    
    def map_devices(self, devices: List[DeviceConfig]) -> Iterator[tuple[DeviceConfig, Optional[str], str]]:
        """映射所有设备"""
        failed_devices = []
        mapped_devices = []
        
        # 先映射所有设备
        for device in devices:
            if not device.enabled:
                continue
            
            if device.is_usb_device:
                tty_device = self.map_usb_device(device)
                if tty_device:
                    mapped_devices.append((device, tty_device, f"{tty_device}:{device.effective_container_path}:rw"))
                else:
                    error_msg = f"未找到USB设备 {device.usb_vendor}:{device.usb_product}"
                    failed_devices.append((device.name, error_msg))
                    mapped_devices.append((device, None, error_msg))
            elif device.path and os.path.exists(device.path):
                mapped_devices.append((device, device.path, f"{device.path}:{device.options}"))
            else:
                error_msg = f"设备路径不存在: {device.path}"
                failed_devices.append((device.name, error_msg))
                mapped_devices.append((device, None, error_msg))
        
        # 检查是否有设备映射失败
        if failed_devices:
            if self._dry_run_mode:
                # dry-run 模式下显示失败设备的简略信息
                for device_name, error_msg in failed_devices:
                    print(f"✗ {device_name}")
            else:
                # 正常模式下显示详细错误信息并退出
                self.logger.error("未找到所有必需的设备，无法启动容器")
                for device_name, error_msg in failed_devices:
                    self.logger.error(f"  缺失设备: {device_name} - {error_msg}")
                print("\n按任意键退出...")
                try:
                    input()
                except:
                    pass
                sys.exit(1)
        
        # 返回所有映射的设备
        for item in mapped_devices:
            yield item


class DockerRunner:
    """Docker容器运行器"""
    
    def __init__(self, config_file: str = "config.yaml", logger: Optional[logging.Logger] = None):
        self.script_dir = Path(__file__).parent.absolute()
        self.config_file = self.script_dir / config_file
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._dry_run_mode = False
        
        # 初始化组件
        self.config_loader = ConfigLoader(self.config_file)
        self.device_mapper = DeviceMapper(self.logger)
        
        # 加载配置
        try:
            self.raw_config = self.config_loader.load_config()
            self._parse_configs()
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise
    
    def _parse_configs(self):
        """解析所有配置"""
        self.container_config = self.config_loader.parse_container(self.raw_config)
        self.resource_config = self.config_loader.parse_resources(self.raw_config)
        self.environment_config = self.config_loader.parse_environment(self.raw_config)
        self.volumes = self.config_loader.parse_volumes(self.raw_config)
        self.devices = self.config_loader.parse_devices(self.raw_config)
    
    def _get_container_status(self, container_name: str) -> ContainerStatus:
        """获取容器状态"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--format', '{{.Names}}'],
                capture_output=True, text=True, check=True
            )
            containers = result.stdout.strip().split('\n')
            if container_name not in containers:
                return ContainerStatus.NOT_EXISTS
            
            running_result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True, check=True
            )
            running_containers = running_result.stdout.strip().split('\n')
            return ContainerStatus.RUNNING if container_name in running_containers else ContainerStatus.STOPPED
        except subprocess.CalledProcessError:
            return ContainerStatus.ERROR
    
    def _stop_and_remove_container(self, container_name: str) -> bool:
        """停止并删除容器"""
        try:
            status = self._get_container_status(container_name)
            if status == ContainerStatus.NOT_EXISTS:
                self.logger.info(f"容器 {container_name} 不存在，无需删除")
                return True
            
            if status == ContainerStatus.RUNNING:
                self.logger.info(f"停止容器: {container_name}")
                subprocess.run(['docker', 'stop', container_name], capture_output=True, check=True)
            
            self.logger.info(f"删除容器: {container_name}")
            subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"停止/删除容器失败: {e}")
            return False
    
    def _build_volume_args(self) -> List[str]:
        """构建挂载点参数"""
        volume_args = []
        for volume in self.volumes:
            if volume.enabled:
                volume_args.extend(['-v', f'{volume.expanded_source}:{volume.target}:{volume.options}'])
        return volume_args
    
    def _build_device_args(self) -> List[str]:
        """构建设备参数"""
        device_args = []
        for device, tty_device, device_arg in self.device_mapper.map_devices(self.devices):
            if tty_device:
                device_args.extend(['--device', device_arg])
                self._log_device_mapping(device, tty_device, device_arg)
        return device_args
    
    def _log_device_mapping(self, device: DeviceConfig, tty_device: str, device_arg: str):
        """记录设备映射日志"""
        if self._dry_run_mode:
            # dry-run 模式下只显示简略信息
            print(f"✓ {device.name}")
        else:
            # 正常模式显示详细信息
            if device.is_usb_device:
                log_info = f"{tty_device} -> {device.effective_container_path} ({device.name})"
                if device.usb_interface is not None:
                    log_info += f" [接口:{device.usb_interface}]"
                if device.usb_serial is not None:
                    log_info += f" [序列号:{device.usb_serial[:20]}...]"
                print(f"添加USB设备映射: {log_info}")
            else:
                print(f"添加设备映射: {device.path}")
    
    def _build_environment_args(self) -> List[str]:
        """构建环境变量参数"""
        env_args = []
        
        # ROS环境变量
        env_args.extend(['-e', f'ROS_DOMAIN_ID={self.environment_config.ros_domain_id}'])
        env_args.extend(['-e', f'ROS_LOCALHOST_ONLY={self.environment_config.ros_localhost_only}'])
        
        # 显示环境变量
        display = self.environment_config.get_display()
        env_args.extend(['-e', f'DISPLAY={display}'])
        
        # PulseAudio 环境变量 (始终添加)
        pulse_server = self.environment_config.get_pulse_server()
        env_args.extend(['-e', f'PULSE_SERVER={pulse_server}'])
        
        return env_args
    
    def _setup_x11_access(self):
        """设置X11访问权限"""
        try:
            # 允许所有用户访问X11显示服务器
            subprocess.run(['xhost', '+local:'], capture_output=True, check=False)
            self.logger.info("已授权X11访问")
        except Exception as e:
            self.logger.warning(f"设置X11访问权限失败: {e}")
    
    def _build_docker_command(self, container_name: str, image_name: str) -> DockerCommand:
        """构建Docker命令"""
        cmd = DockerCommand(
            name=container_name,
            image=image_name,
            network=self.resource_config.network if self.resource_config.network != 'bridge' else None,
            privileged=self.resource_config.privileged,
            gpu_options=self.resource_config.gpu_options if self.resource_config.gpu_enabled else None,
            volumes=self._build_volume_args(),
            devices=self._build_device_args(),
            environment=self._build_environment_args()
        )
        return cmd
    
    def run(self, container_name: Optional[str] = None, image_name: Optional[str] = None, dry_run: bool = False) -> bool:
        """运行容器"""
        # 设置 dry-run 模式标志
        self._dry_run_mode = dry_run
        self.device_mapper._dry_run_mode = dry_run
        
        # 使用配置或参数中的值
        final_container_name = container_name or self.container_config.name
        final_image_name = image_name or f"{self.container_config.image_repository}:{self.container_config.image_tag}"
        
        if not final_container_name or not final_image_name:
            self.logger.error("容器名称和镜像名称不能为空")
            return False
        
        # 停止现有容器
        if not dry_run:
            if not self._stop_and_remove_container(final_container_name):
                return False
        
        # 授权X11访问
        if not dry_run:
            self._setup_x11_access()
        
        # 构建Docker命令
        docker_cmd = self._build_docker_command(final_container_name, final_image_name)
        command = docker_cmd.build()
        
        if dry_run:
            print()
            print(' '.join(command))
            return True
        
        # 执行Docker命令
        try:
            subprocess.run(command, check=True)
            self.logger.info(f"容器 {final_container_name} 启动成功")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"启动容器失败: {e}")
            return False
        except KeyboardInterrupt:
            self.logger.info("用户中断操作")
            return False


class DockerStarter:
    """Docker容器启动器 - 简化部署版本"""
    
    def __init__(self, config_file: str = "config.yaml", logger: Optional[logging.Logger] = None):
        self.script_dir = Path(__file__).parent.absolute()
        self.config_file = self.script_dir / config_file
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = self._load_config()
        self.runner = DockerRunner(config_file, self.logger)
        
        # 需要检查的USB设备配置 (Valve Software LHR devices)
        self.required_usb_devices = [
            {'vendor_id': '28de', 'product_id': '2300', 'name': 'Valve Software LHR', 'count': 2}
        ]
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"解析配置文件失败: {e}")
    
    def _get_config(self, path: str, default: str = "") -> str:
        """获取配置值"""
        try:
            keys = path.strip('.').split('.')
            return str(reduce(getitem, keys, self.config)) if self.config else default
        except (KeyError, TypeError):
            return default
    
    def _check_docker(self) -> bool:
        """检查Docker是否可用"""
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Docker 未安装。请运行: ./install.sh")
            return False
    
    def _check_usb_devices(self) -> bool:
        """检查必需的USB设备是否存在"""
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, check=True)
            lsusb_output = result.stdout
            
            for device_req in self.required_usb_devices:
                vendor_id = device_req['vendor_id']
                product_id = device_req['product_id']
                required_count = device_req['count']
                device_name = device_req['name']
                
                # 计算匹配的设备数量
                device_pattern = f"{vendor_id}:{product_id}"
                actual_count = lsusb_output.count(device_pattern)
                
                if actual_count != required_count:
                    self.logger.error(
                        f"USB设备检查失败: 需要 {required_count} 个 {device_name} 设备 "
                        f"(ID {vendor_id}:{product_id})，实际找到 {actual_count} 个"
                    )
                    self.logger.error("请确保所有必需的USB设备已连接")
                    
                    # 显示当前连接的设备信息
                    if actual_count > 0:
                        self.logger.info(f"当前找到的 {device_name} 设备:")
                        for line in lsusb_output.split('\n'):
                            if device_pattern in line:
                                self.logger.info(f"  {line.strip()}")
                    
                    return False
                
                self.logger.info(
                    f"✓ 找到 {actual_count} 个 {device_name} 设备 (ID {vendor_id}:{product_id})"
                )
            
            return True
            
        except subprocess.CalledProcessError:
            self.logger.error("无法执行 lsusb 命令，请确保 usbutils 已安装")
            return False
        except FileNotFoundError:
            self.logger.error("未找到 lsusb 命令，请安装 usbutils 包")
            return False
    
    def _check_image_exists(self, image_name: str) -> bool:
        """检查镜像是否存在"""
        try:
            result = subprocess.run(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}', image_name],
                capture_output=True, text=True, check=True
            )
            return image_name in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def _pull_image(self, image_name: str) -> bool:
        """拉取镜像"""
        self.logger.info(f"拉取镜像: {image_name}")
        try:
            subprocess.run(['docker', 'pull', image_name], check=True)
            self.logger.info(f"镜像拉取成功: {image_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"拉取镜像失败: {e}")
            return False
    
    def _enter_container(self, container_name: str):
        """进入容器"""
        command = self._get_config('.container.command', 'bash')
        
        # 检查是否需要通过bash执行命令
        exec_mode = self._get_config('.container.exec_mode', 'interactive')
        
        try:
            if exec_mode == 'command':
                # 命令执行模式：通过bash -c执行，可以加载环境变量
                self.logger.info(f"在容器中执行命令: {command}")
                subprocess.run(['docker', 'exec', '-it', container_name, 'bash', '-c', command], check=True)
            else:
                # 交互式模式：直接执行命令（通常是bash）
                self.logger.info(f"进入容器: {container_name}")
                subprocess.run(['docker', 'exec', '-it', container_name, command], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"进入容器失败: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.info("用户中断操作")
            sys.exit(0)
    
    def start(self, dry_run: bool = False, force: bool = False) -> bool:
        """启动容器 - 简化部署版本"""
        if not dry_run and not self._check_docker():
            return False
        
        # 检查必需的USB设备
        if not dry_run and not self._check_usb_devices():
            return False
        
        # 解析配置
        container_name = self._get_config('.container.name')
        image_repository = self._get_config('.container.image.repository')
        image_tag = self._get_config('.container.image.tag')
        
        if not container_name or not image_repository or not image_tag:
            self.logger.error("容器名称和镜像信息不能为空")
            return False
        
        image_name = f"{image_repository}:{image_tag}"
        
        if dry_run:
            # dry_run 模式下直接调用 runner.run()，会打印完整的 docker run 命令
            return self.runner.run(container_name, image_name, dry_run)
        
        # 始终拉取镜像以确保使用最新版本
        if not self._pull_image(image_name):
            self.logger.error("镜像拉取失败，无法继续")
            return False
        
        # 运行容器
        if not self.runner.run(container_name, image_name, dry_run):
            self.logger.error("容器启动失败")
            return False
        
        # 进入容器
        self._enter_container(container_name)
        return True


def main():
    parser = argparse.ArgumentParser(description='Docker 容器启动脚本 - 简化部署版本')
    parser.add_argument('--config','-c', default='config.yaml', help='配置文件路径')
    parser.add_argument('--dry-run', action='store_true', help='预览命令而不执行')
    parser.add_argument('--force', '-f', action='store_true', help='强制拉取镜像并重新创建容器')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        starter = DockerStarter(args.config)
        success = starter.start(args.dry_run, args.force)
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()