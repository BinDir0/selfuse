#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import yaml
except ImportError:
    print("错误: 缺少 yaml 模块，请运行: pip install PyYAML", file=sys.stderr)
    sys.exit(1)


@dataclass
class TTYDeviceInfo:
    """TTY设备信息"""
    device_path: str
    vendor_id: Optional[str] = None
    product_id: Optional[str] = None
    vendor_name: Optional[str] = None
    model_name: Optional[str] = None
    serial: Optional[str] = None
    interface_num: Optional[str] = None
    config_name: Optional[str] = None  # 配置文件中的设备名称
    
    def __str__(self) -> str:
        """格式化输出设备信息"""
        lines = [f"\n{'='*60}"]
        
        if self.config_name:
            lines.append(f"配置名称: {self.config_name}")
        
        lines.append(f"设备路径: {self.device_path}")
        
        if self.vendor_id and self.product_id:
            lines.append(f"USB ID: {self.vendor_id}:{self.product_id}")
        
        if self.vendor_name:
            lines.append(f"制造商: {self.vendor_name}")
        
        if self.model_name:
            lines.append(f"型号: {self.model_name}")
        
        if self.serial:
            lines.append(f"序列号: {self.serial}")
        
        if self.interface_num:
            lines.append(f"接口编号: {self.interface_num}")
        
        lines.append('='*60)
        return '\n'.join(lines)


class TTYDeviceFinder:
    """TTY设备查找器"""
    
    def __init__(self):
        self.tty_devices = self._scan_tty_devices()
    
    def _scan_tty_devices(self) -> List[str]:
        """扫描系统中的所有tty设备"""
        tty_devices = []
        for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ttyS*']:
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
        except FileNotFoundError:
            print("错误: 未找到 udevadm 命令。请确保系统已安装 udev。", file=sys.stderr)
            return {}
        except Exception as e:
            print(f"警告: 获取设备 {tty_device} 信息失败: {e}", file=sys.stderr)
            return {}
    
    def get_all_devices_info(self) -> List[TTYDeviceInfo]:
        """获取所有tty设备的详细信息"""
        devices_info = []
        
        for tty_device in self.tty_devices:
            device_info = self._get_device_info(tty_device)
            
            if not device_info:
                # 即使没有详细信息，也记录设备路径
                devices_info.append(TTYDeviceInfo(device_path=tty_device))
                continue
            
            info = TTYDeviceInfo(
                device_path=tty_device,
                vendor_id=device_info.get('ID_VENDOR_ID'),
                product_id=device_info.get('ID_MODEL_ID'),
                vendor_name=device_info.get('ID_VENDOR'),
                model_name=device_info.get('ID_MODEL'),
                serial=device_info.get('ID_SERIAL'),
                interface_num=device_info.get('ID_USB_INTERFACE_NUM')
            )
            devices_info.append(info)
        
        return devices_info
    
    def find_by_usb_id(self, vendor_id: str, product_id: str, 
                      interface_num: Optional[str] = None) -> List[TTYDeviceInfo]:
        """根据USB ID查找设备"""
        matched_devices = []
        
        for tty_device in self.tty_devices:
            device_info = self._get_device_info(tty_device)
            
            # 匹配 vendor 和 product ID
            if (device_info.get('ID_VENDOR_ID') == vendor_id and 
                device_info.get('ID_MODEL_ID') == product_id):
                
                # 如果指定了接口编号，进行精确匹配
                if interface_num is not None:
                    if device_info.get('ID_USB_INTERFACE_NUM') != interface_num:
                        continue
                
                info = TTYDeviceInfo(
                    device_path=tty_device,
                    vendor_id=device_info.get('ID_VENDOR_ID'),
                    product_id=device_info.get('ID_MODEL_ID'),
                    vendor_name=device_info.get('ID_VENDOR'),
                    model_name=device_info.get('ID_MODEL'),
                    serial=device_info.get('ID_SERIAL'),
                    interface_num=device_info.get('ID_USB_INTERFACE_NUM')
                )
                matched_devices.append(info)
        
        return matched_devices
    
    def find_by_keyword(self, keyword: str) -> List[TTYDeviceInfo]:
        """根据关键词搜索设备（在制造商、型号、序列号中搜索）"""
        keyword_lower = keyword.lower()
        matched_devices = []
        
        for device_info in self.get_all_devices_info():
            # 在各个字段中搜索关键词
            search_fields = [
                device_info.vendor_name or '',
                device_info.model_name or '',
                device_info.serial or '',
                device_info.device_path or ''
            ]
            
            if any(keyword_lower in field.lower() for field in search_fields):
                matched_devices.append(device_info)
        
        return matched_devices


@dataclass
class DeviceConfig:
    """设备配置"""
    name: str
    usb_vendor: Optional[str] = None
    usb_product: Optional[str] = None
    usb_interface: Optional[str] = None
    container_path: Optional[str] = None
    enabled: bool = True


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"解析配置文件失败: {e}")
    
    def get_device_configs(self) -> List[DeviceConfig]:
        """获取设备配置列表"""
        config = self.load_config()
        devices = config.get('devices', [])
        
        device_configs = []
        for dev in devices:
            if isinstance(dev, dict):
                # 只处理有 USB 配置的设备
                if 'usb_vendor' in dev and 'usb_product' in dev:
                    device_configs.append(DeviceConfig(
                        name=dev.get('name', 'Unknown'),
                        usb_vendor=dev.get('usb_vendor'),
                        usb_product=dev.get('usb_product'),
                        usb_interface=dev.get('usb_interface'),
                        container_path=dev.get('container_path'),
                        enabled=dev.get('enabled', True)
                    ))
        
        return device_configs


def find_devices_from_config(config_path: str = "config.yaml", verbose: bool = False) -> List[TTYDeviceInfo]:
    """从配置文件查找设备"""
    try:
        loader = ConfigLoader(config_path)
        device_configs = loader.get_device_configs()
        finder = TTYDeviceFinder()
        
        if verbose:
            print(f"\n配置文件中定义了 {len(device_configs)} 个设备:")
            for dev_config in device_configs:
                status = "✓" if dev_config.enabled else "✗"
                interface_info = f" [接口:{dev_config.usb_interface}]" if dev_config.usb_interface else ""
                print(f"  {status} {dev_config.name}: {dev_config.usb_vendor}:{dev_config.usb_product}{interface_info}")
            print()
        
        found_devices = []
        missing_devices = []
        
        for dev_config in device_configs:
            if not dev_config.enabled:
                continue
            
            devices = finder.find_by_usb_id(
                dev_config.usb_vendor,
                dev_config.usb_product,
                dev_config.usb_interface
            )
            
            if devices:
                # 为找到的设备添加配置名称
                for device in devices:
                    device.config_name = dev_config.name
                    found_devices.append(device)
            else:
                missing_devices.append(dev_config)
        
        # 显示未找到的设备
        if missing_devices and verbose:
            print(f"\n⚠ 未找到以下设备:")
            for dev_config in missing_devices:
                interface_info = f" [接口:{dev_config.usb_interface}]" if dev_config.usb_interface else ""
                print(f"  ✗ {dev_config.name}: {dev_config.usb_vendor}:{dev_config.usb_product}{interface_info}")
            print()
        
        return found_devices
    
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"错误: 加载配置失败: {e}", file=sys.stderr)
        return []


def print_device_summary(devices: List[TTYDeviceInfo], title: str = "设备列表"):
    """打印设备摘要"""
    print(f"\n{'='*60}")
    print(f"{title} (共 {len(devices)} 个设备)")
    print(f"{'='*60}")
    
    if not devices:
        print("未找到设备")
        return
    
    for device in devices:
        print(device)


def print_simple_list(devices: List[TTYDeviceInfo]):
    """打印简化的设备列表"""
    if not devices:
        print("未找到设备")
        return
    
    print(f"\n找到 {len(devices)} 个设备:")
    print("-" * 80)
    
    for device in devices:
        config_name = f"[{device.config_name}]" if device.config_name else ""
        usb_id = f"{device.vendor_id}:{device.product_id}" if device.vendor_id else "N/A"
        vendor = device.vendor_name or "Unknown"
        model = device.model_name or "Unknown"
        interface = f" [接口:{device.interface_num}]" if device.interface_num else ""
        
        print(f"{device.device_path:<20} {usb_id:<12} {config_name:<25} {vendor:<15} {model}{interface}")


def main():
    parser = argparse.ArgumentParser(
        description='查找并显示串口设备信息（手套和手部追踪设备）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          根据配置文件查找设备（默认）
  %(prog)s --config config.yaml     指定配置文件
  %(prog)s --all                    显示所有串口设备
  %(prog)s --usb 0483:5739          查找指定USB ID的设备
  %(prog)s --keyword valve          搜索包含关键词的设备
  %(prog)s --keyword glove          搜索手套设备
  %(prog)s --keyword hand           搜索手部设备
  %(prog)s --simple                 简化输出格式
        """
    )
    
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--all', '-a', action='store_true', 
                       help='显示所有串口设备')
    parser.add_argument('--usb', '-u', metavar='VENDOR:PRODUCT',
                       help='根据USB ID查找设备 (例如: 0483:5739)')
    parser.add_argument('--interface', '-i', metavar='NUM',
                       help='指定USB接口编号')
    parser.add_argument('--keyword', '-k', metavar='KEYWORD',
                       help='根据关键词搜索设备')
    parser.add_argument('--simple', '-s', action='store_true',
                       help='使用简化的输出格式')
    
    args = parser.parse_args()
    
    # 检查是否有root权限（某些情况下udevadm需要）
    if os.geteuid() != 0:
        print("提示: 某些设备信息可能需要root权限才能查看", file=sys.stderr)
        print("      如果输出不完整，请尝试使用 sudo 运行此脚本\n", file=sys.stderr)
    
    finder = TTYDeviceFinder()
    
    # 如果没有指定任何参数，从配置文件读取设备
    if not any([args.all, args.usb, args.keyword]):
        print(f"正在从配置文件读取设备信息: {args.config}")
        devices = find_devices_from_config(args.config, verbose=True)
        
        if devices:
            if args.simple:
                print_simple_list(devices)
            else:
                print_device_summary(devices, f"找到的设备")
        else:
            print("提示: 使用 --all 查看所有串口设备")
        return
    
    # 显示所有设备
    if args.all:
        devices = finder.get_all_devices_info()
        if args.simple:
            print_simple_list(devices)
        else:
            print_device_summary(devices, "所有串口设备")
    
    # 根据USB ID查找
    if args.usb:
        try:
            vendor_id, product_id = args.usb.split(':')
            devices = finder.find_by_usb_id(vendor_id, product_id, args.interface)
            title = f"USB设备 {vendor_id}:{product_id}"
            if args.interface:
                title += f" (接口 {args.interface})"
            
            if args.simple:
                print_simple_list(devices)
            else:
                print_device_summary(devices, title)
        except ValueError:
            print("错误: USB ID 格式应为 VENDOR:PRODUCT (例如: 28de:2300)", file=sys.stderr)
            sys.exit(1)
    
    # 根据关键词搜索
    if args.keyword:
        devices = finder.find_by_keyword(args.keyword)
        title = f"包含关键词 '{args.keyword}' 的设备"
        
        if args.simple:
            print_simple_list(devices)
        else:
            print_device_summary(devices, title)


if __name__ == '__main__':
    main()

