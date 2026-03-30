#!/usr/bin/env python3
"""
Zarr Dataset Episode Analysis
Zarr 数据集 Episode 分析脚本
Analyzes episode_name and episode_ends from multiple zarr datasets
分析来自多个 zarr 数据集的 episode_name 和 episode_ends，生成详细的统计报告 PDF。
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
import urllib.request
import seaborn as sns
from typing import Dict, List, Tuple
import re
import json
import argparse
import os
from collections import defaultdict

from episode_source import load_episode_bundle, normalize_data_format

zarrdir = os.environ.get("DEFAULT_ZARR_SCAN_DIR", "/share_data/yeyuyao/teleop-data-complete")
num = 0

def get_subdirs(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d))]

def get_subnames(directory):
    return [d for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d))]

def get_zarrs(zarrdir, num):
    '''从zarrdir中提取所有zarr_path，返回指定位置的10个zarr_path'''
    zarrlist = get_subdirs(zarrdir)
    zarrlist.sort()
    return zarrlist
    #return [zarrpath for i, zarrpath in enumerate(zarrlist) if i // 3 == num]

def get_names(zarrdir, num):
    '''从zarrdir中提取所有zarr_path，返回指定位置的10个zarr_path'''
    namelist = get_subnames(zarrdir)
    namelist.sort()
    return namelist
    #return [name for i, name in enumerate(namelist) if i // 3 == num]

def merge_datasets_info(dataset_info_dicts):
    """
    合并多个 dataset_info 字典。
    
    参数:
        dataset_info_dicts: 列表，包含多个 dataset_info 字典。
    
    返回:
        一个合并后的大字典，结构类似原字典，但每个值都是所有数据集拼接后的结果，
        并包含全局索引映射。
    """
    # 初始化合并结果容器
    merged = {
        'all_dataset_names': [],          # 记录所有数据集的名称
        'episode_names': [],       # 合并后的 episode_names
        'episode_ends': [],        # 合并后的 episode_ends (需要偏移)
        'episode_to_dataset': [],         # 记录每个episode属于哪个数据集
        'episode_offset_in_dataset': []   # 记录每个episode在原始数据集中的起始位置
    }
    
    # 用于存储需要沿第一维拼接的数组
    array_keys = [
        'instruction', 'instruction_num', 'intrinsic_h', 'intrinsic_b',
        'extrinsic_hl', 'extrinsic_hr', 'extrinsic_bl', 'extrinsic_br',
        'joint_state', 'joint_action', 'wrist_state_h', 'wrist_action_h',
        'wrist_state_b', 'wrist_action_b', 'fingertips_state_h', 
        'fingertips_action_h', 'fingertips_state_b', 'fingertips_action_b'
    ]
    
    # 初始化所有数组键的容器
    for key in array_keys:
        merged[key] = []
    
    # 初始化其他兼容性键的容器
    compatible_keys = ['intrinsic', 'extrinsic', 'wrist_state', 'wrist_action', 
                       'fingertips_state', 'fingertips_action']
    for key in compatible_keys:
        merged[key] = []
    
    # 初始化问题episodes记录
    merged['problematic_episodes_extrinsic'] = {}
    
    # 全局偏移量（用于调整episode_ends）
    global_frame_offset = 0
    global_episode_offset = 0
    
    # 遍历每个数据集进行合并
    for ds_idx, (ds_name, ds_info) in enumerate(dataset_info_dicts.items()):
        merged['all_dataset_names'].append(ds_name)
        
        # 1. 处理 episode_names 和 lengths
        num_episodes = len(ds_info['episode_names'])
        merged['episode_names'].extend(ds_info['episode_names'])
        
        # 2. 处理 episode_ends (需要加上全局偏移)
        adjusted_ends = ds_info['episode_ends'] + global_frame_offset
        merged['episode_ends'].extend(adjusted_ends)
        
        # 3. 记录索引映射信息
        for ep_idx in range(num_episodes):
            merged['episode_to_dataset'].append(ds_name)
            merged['episode_offset_in_dataset'].append(global_episode_offset + ep_idx)
        
        # 4. 合并数组数据（沿第一维）
        for key in array_keys:
            if key in ds_info:
                merged[key].append(ds_info[key])
        
        # 5. 合并兼容性键（使用胸部相机数据）
        for key in compatible_keys:
            if key in ds_info:
                merged[key].append(ds_info[key])
        
        # 6. 合并问题episodes（需要添加数据集前缀以避免冲突）
        if 'problematic_episodes_extrinsic' in ds_info:
            for ep_name, problem_desc in ds_info['problematic_episodes_extrinsic'].items():
                prefixed_name = f"{ds_name}::{ep_name}"
                merged['problematic_episodes_extrinsic'][prefixed_name] = {
                    'dataset': ds_name,
                    'original_name': ep_name,
                    'description': problem_desc
                }
        
        # 7. 更新偏移量
        if ds_info['episode_ends'].size > 0:
            global_frame_offset = adjusted_ends[-1]
        global_episode_offset += num_episodes
    
    # 将列表转换为numpy数组（对于需要拼接的数组）
    for key in array_keys + compatible_keys:
        if merged[key]:  # 确保列表不为空
            merged[key] = np.concatenate(merged[key], axis=0)
        else:
            merged[key] = np.array([])  # 或保留为None
    
    # 转换其他列表为数组
    merged['episode_names'] = np.array(merged['episode_names'])
    merged['episode_ends'] = np.array(merged['episode_ends'])
    merged['episode_to_dataset'] = np.array(merged['episode_to_dataset'])
    merged['episode_offset_in_dataset'] = np.array(merged['episode_offset_in_dataset'])
    
    # 添加便捷的统计信息
    merged['total_episodes'] = len(merged['episode_names'])
    merged['total_frames'] = merged['episode_ends'][-1] if merged['episode_ends'].size > 0 else 0
    merged['dataset_count'] = len(merged['all_dataset_names'])
    
    return merged

def plot_frame_change_rate(datasets_info: Dict, data_type: str = 'state') -> plt.Figure:
    """
    Analyze frame-to-frame change rates for STATE or ACTION with enhanced statistics, rotation analysis, and anomaly detection
    分析 STATE 或 ACTION 的帧间变化率，包括增强的统计数据、旋转分析和异常检测
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        data_type: 'state' or 'action' to specify which data to plot
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    # 3 rows: Translation, Rotation, Fingertips
    # 3行: 平移、旋转、指尖
    fig, axes = plt.subplots(3, num_datasets, figsize=(6*num_datasets, 12))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Determine type label
    type_label = 'State' if data_type == 'state' else 'Action'
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        wrist_state = info['wrist_state']
        wrist_action = info['wrist_action']
        fingertips_state = info['fingertips_state']
        fingertips_action = info['fingertips_action']
        episode_ends = info['episode_ends']
        
        # Create mask to exclude episode boundaries (shared for both state and action)
        # 创建掩码以排除 Episode 边界处的帧间差异
        num_diffs = len(wrist_state) - 1
        valid_mask = np.ones(num_diffs, dtype=bool)
        boundary_indices = episode_ends[:-1] - 1  # Exclude episode boundaries
        valid_mask[boundary_indices] = False
        
        # ========== STATE calculations ==========
        wrist_diff_state = np.diff(wrist_state, axis=0)
        fingertips_diff_state = np.diff(fingertips_state, axis=0)
        
        # Wrist translation magnitude (first 6: left+right xyz)
        wrist_translation_state = wrist_diff_state[:, :6]
        wrist_translation_mag_state = np.linalg.norm(wrist_translation_state, axis=1)
        wrist_translation_mag_state = wrist_translation_mag_state[valid_mask]
        
        # Wrist rotation magnitude (next 12: left+right rot6d) - use proper rotation angle calculation
        # Calculate frame-to-frame rotation angles correctly using rot6d representation
        left_rot6d_state = wrist_state[:, 6:12]   # Left hand rotation (rot6d, 6D)
        right_rot6d_state = wrist_state[:, 12:18]  # Right hand rotation (rot6d, 6D)
        left_rotation_mag_state = compute_rotation_angle_from_rot6d_diff(left_rot6d_state[:-1], left_rot6d_state[1:])
        right_rotation_mag_state = compute_rotation_angle_from_rot6d_diff(right_rot6d_state[:-1], right_rot6d_state[1:])
        # Apply valid mask to both hands separately, then concatenate
        left_rotation_mag_state = left_rotation_mag_state[valid_mask]
        right_rotation_mag_state = right_rotation_mag_state[valid_mask]
        wrist_rotation_mag_state = np.concatenate([left_rotation_mag_state, right_rotation_mag_state])
        
        # Fingertips magnitude: reshape to (frames-1, 10, 3) and calculate magnitude for each fingertip
        # Then take mean across all fingertips (same method as ACTION)
        fingertips_diff_3d_state = fingertips_diff_state.reshape(-1, 10, 3)
        fingertips_magnitude_state = np.linalg.norm(fingertips_diff_3d_state, axis=2)  # (frames-1, 10)
        fingertips_mag_state = np.mean(fingertips_magnitude_state, axis=1)  # (frames-1,): mean across all fingertips
        fingertips_mag_state = fingertips_mag_state[valid_mask]
        
        # ========== ACTION calculations ==========
        wrist_diff_action = np.diff(wrist_action, axis=0)
        fingertips_diff_action = np.diff(fingertips_action, axis=0)
        
        # Wrist translation magnitude (first 6: left+right xyz)
        wrist_translation_action = wrist_diff_action[:, :6]
        wrist_translation_mag_action = np.linalg.norm(wrist_translation_action, axis=1)
        wrist_translation_mag_action = wrist_translation_mag_action[valid_mask]
        
        # Wrist rotation magnitude (next 12: left+right rot6d) - use proper rotation angle calculation
        # Calculate frame-to-frame rotation angles correctly using rot6d representation
        left_rot6d_action = wrist_action[:, 6:12]   # Left hand rotation (rot6d, 6D)
        right_rot6d_action = wrist_action[:, 12:18]  # Right hand rotation (rot6d, 6D)
        left_rotation_mag_action = compute_rotation_angle_from_rot6d_diff(left_rot6d_action[:-1], left_rot6d_action[1:])
        right_rotation_mag_action = compute_rotation_angle_from_rot6d_diff(right_rot6d_action[:-1], right_rot6d_action[1:])
        # Apply valid mask to both hands separately, then concatenate
        left_rotation_mag_action = left_rotation_mag_action[valid_mask]
        right_rotation_mag_action = right_rotation_mag_action[valid_mask]
        wrist_rotation_mag_action = np.concatenate([left_rotation_mag_action, right_rotation_mag_action])
        
        # Fingertips magnitude: reshape to (frames-1, 10, 3) and calculate magnitude for each fingertip
        # Then take mean across all fingertips (same method as Outlier Distribution Analysis)
        fingertips_diff_3d_action = fingertips_diff_action.reshape(-1, 10, 3)
        fingertips_magnitude_action = np.linalg.norm(fingertips_diff_3d_action, axis=2)  # (frames-1, 10)
        fingertips_mag_action = np.mean(fingertips_magnitude_action, axis=1)  # (frames-1,): mean across all fingertips
        fingertips_mag_action = fingertips_mag_action[valid_mask]
        
        # Helper function to plot a single metric
        def plot_metric(ax, data, title, xlabel, color_idx):
            sample_size = min(100000, len(data))
            if sample_size < len(data):
                sample_indices = np.linspace(0, len(data)-1, sample_size, dtype=int)
                sample_data = data[sample_indices]
            else:
                sample_data = data
            
            sns.histplot(sample_data, bins=50, ax=ax,
                        kde=True, color=sns.color_palette("Set2")[color_idx], edgecolor='black', alpha=0.7)
            
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
            
            stats_text = (f'Mean: {mean_val:.4f}\n'
                         f'Median: {median_val:.4f}\n'
                         f'Std: {std_val:.4f}\n'
                         f'Min: {min_val:.4f}\n'
                         f'Max: {max_val:.4f}')
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
        
        # Select data based on data_type
        if data_type == 'state':
            wrist_translation_mag = wrist_translation_mag_state
            wrist_rotation_mag = wrist_rotation_mag_state
            fingertips_mag = fingertips_mag_state
        else:  # action
            wrist_translation_mag = wrist_translation_mag_action
            wrist_rotation_mag = wrist_rotation_mag_action
            fingertips_mag = fingertips_mag_action
        
        # Row 0: Wrist Translation Change Rate
        # 第一行: 手腕平移变化率
        plot_metric(axes[0, idx], wrist_translation_mag,
                   f'{name} - Wrist Translation Change Rate ({type_label})',
                   'Frame-to-Frame Wrist Translation Change (m)', idx)
        
        # Row 1: Wrist Rotation Change Rate
        # 第二行: 手腕旋转变化率
        plot_metric(axes[1, idx], wrist_rotation_mag,
                   f'{name} - Wrist Rotation Change Rate ({type_label})',
                   'Frame-to-Frame Wrist Rotation Change (rad)', idx)
        
        # Row 2: Fingertips Change Rate
        # 第三行: 指尖变化率
        plot_metric(axes[2, idx], fingertips_mag,
                   f'{name} - Fingertips Change Rate ({type_label})',
                   'Frame-to-Frame Fingertips Change (m)', idx)
    
    plt.suptitle(f'Frame-to-Frame Change Rate Analysis ({type_label})', fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(hspace=0.4, top=0.95)
    
    return fig

def plot_frame_change_rate_outliers(datasets_info: Dict, data_type: str = 'state') -> plt.Figure:
    """
    Analyze outliers in frame-to-frame change rates for STATE or ACTION using absolute threshold method
    使用绝对阈值方法分析 STATE 或 ACTION 的帧间变化率中的异常值
    Reference: plot_outlier_distribution_analysis for visualization style
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        data_type: 'state' or 'action' to specify which data to plot
        
    Returns:
        matplotlib Figure object
    """
    # Use same thresholds as threshold-based anomaly detection
    # 使用与基于阈值的异常检测相同的阈值
    THRESHOLD_WRIST_TRANSLATION = 0.05  # 5cm
    THRESHOLD_WRIST_ROTATION = 0.5      # ~28.6 degrees
    THRESHOLD_FINGERTIPS_DISPLACEMENT = 0.05  # 5cm
    
    num_datasets = len(datasets_info)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, num_datasets, hspace=0.6, wspace=0.3)
    
    # Determine type label
    type_label = 'State' if data_type == 'state' else 'Action'
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        wrist_state = info['wrist_state']
        wrist_action = info['wrist_action']
        fingertips_state = info['fingertips_state']
        fingertips_action = info['fingertips_action']
        episode_ends = info['episode_ends']
        
        # Create mask to exclude episode boundaries (shared for both state and action)
        num_diffs = len(wrist_state) - 1
        valid_mask = np.ones(num_diffs, dtype=bool)
        boundary_indices = episode_ends[:-1] - 1  # Exclude episode boundaries
        valid_mask[boundary_indices] = False
        
        # ========== STATE calculations ==========
        wrist_diff_state = np.diff(wrist_state, axis=0)
        fingertips_diff_state = np.diff(fingertips_state, axis=0)
        
        # Wrist translation magnitude (first 6: left+right xyz)
        wrist_translation_state = wrist_diff_state[:, :6]
        wrist_translation_mag_state = np.linalg.norm(wrist_translation_state, axis=1)
        wrist_translation_mag_state = wrist_translation_mag_state[valid_mask]
        
        # Wrist rotation magnitude (next 12: left+right rot6d) - use proper rotation angle calculation
        # Calculate frame-to-frame rotation angles correctly using rot6d representation
        left_rot6d_state = wrist_state[:, 6:12]   # Left hand rotation (rot6d, 6D)
        right_rot6d_state = wrist_state[:, 12:18]  # Right hand rotation (rot6d, 6D)
        left_rotation_mag_state = compute_rotation_angle_from_rot6d_diff(left_rot6d_state[:-1], left_rot6d_state[1:])
        right_rotation_mag_state = compute_rotation_angle_from_rot6d_diff(right_rot6d_state[:-1], right_rot6d_state[1:])
        # Apply valid mask to both hands separately, then concatenate
        left_rotation_mag_state = left_rotation_mag_state[valid_mask]
        right_rotation_mag_state = right_rotation_mag_state[valid_mask]
        wrist_rotation_mag_state = np.concatenate([left_rotation_mag_state, right_rotation_mag_state])
        
        # Fingertips magnitude: reshape to (frames-1, 10, 3) and calculate magnitude for each fingertip
        # Then take mean across all fingertips (same method as Outlier Distribution Analysis)
        fingertips_diff_3d_state = fingertips_diff_state.reshape(-1, 10, 3)
        fingertips_magnitude_state = np.linalg.norm(fingertips_diff_3d_state, axis=2)  # (frames-1, 10)
        fingertips_mag_state = np.mean(fingertips_magnitude_state, axis=1)  # (frames-1,): mean across all fingertips
        fingertips_mag_state = fingertips_mag_state[valid_mask]
        
        # ========== ACTION calculations ==========
        wrist_diff_action = np.diff(wrist_action, axis=0)
        fingertips_diff_action = np.diff(fingertips_action, axis=0)
        
        # Wrist translation magnitude (first 6: left+right xyz)
        wrist_translation_action = wrist_diff_action[:, :6]
        wrist_translation_mag_action = np.linalg.norm(wrist_translation_action, axis=1)
        wrist_translation_mag_action = wrist_translation_mag_action[valid_mask]
        
        # Wrist rotation magnitude (next 12: left+right rot6d) - use proper rotation angle calculation
        # Calculate frame-to-frame rotation angles correctly using rot6d representation
        left_rot6d_action = wrist_action[:, 6:12]   # Left hand rotation (rot6d, 6D)
        right_rot6d_action = wrist_action[:, 12:18]  # Right hand rotation (rot6d, 6D)
        left_rotation_mag_action = compute_rotation_angle_from_rot6d_diff(left_rot6d_action[:-1], left_rot6d_action[1:])
        right_rotation_mag_action = compute_rotation_angle_from_rot6d_diff(right_rot6d_action[:-1], right_rot6d_action[1:])
        # Apply valid mask to both hands separately, then concatenate
        left_rotation_mag_action = left_rotation_mag_action[valid_mask]
        right_rotation_mag_action = right_rotation_mag_action[valid_mask]
        wrist_rotation_mag_action = np.concatenate([left_rotation_mag_action, right_rotation_mag_action])
        
        # Fingertips magnitude: reshape to (frames-1, 10, 3) and calculate magnitude for each fingertip
        # Then take mean across all fingertips (same method as Outlier Distribution Analysis)
        fingertips_diff_3d_action = fingertips_diff_action.reshape(-1, 10, 3)
        fingertips_magnitude_action = np.linalg.norm(fingertips_diff_3d_action, axis=2)  # (frames-1, 10)
        fingertips_mag_action = np.mean(fingertips_magnitude_action, axis=1)  # (frames-1,): mean across all fingertips
        fingertips_mag_action = fingertips_mag_action[valid_mask]
        
        # Helper function to plot outlier distribution
        def plot_outlier_distribution(ax, data, threshold, title, xlabel, unit='m', is_rotation=False, num_frames=None):
            """
            Plot outlier distribution
            is_rotation: If True, data contains concatenated left+right hand data and we need to count unique frames
            num_frames: Number of frames (required if is_rotation=True)
            """
            outlier_mask = data > threshold
            outlier_values = data[outlier_mask]
            
            extreme_error_mask = data > (threshold * 5)
            extreme_values = data[extreme_error_mask] if np.sum(extreme_error_mask) > 0 else np.array([])
            
            # For rotation data, map to unique frames
            if is_rotation and num_frames is not None:
                outlier_indices = np.where(outlier_mask)[0]
                outlier_frames_unique = np.unique(outlier_indices % num_frames)
                outlier_count = len(outlier_frames_unique)
                
                extreme_indices = np.where(extreme_error_mask)[0]
                extreme_frames_unique = np.unique(extreme_indices % num_frames)
                extreme_error_count = len(extreme_frames_unique)
                
                total_count = num_frames
            else:
                outlier_count = len(outlier_values)
                extreme_error_count = len(extreme_values)
                total_count = len(data)
            
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            
            if len(outlier_values) > 0:
                max_val = np.nanmax(outlier_values)
                min_val = np.nanmin(outlier_values)
                bins = np.linspace(min_val, max(max_val * 1.1, threshold * 2), 50)
                ax.hist(outlier_values, bins=bins, alpha=0.8, color='#FF6B6B', label='Outliers', edgecolor='black')
            else:
                bins = np.linspace(threshold, threshold * 2, 50)
            
            threshold_label = f'{threshold:.4f}{unit}' if unit == 'm' else f'{threshold:.4f}{unit}'
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold: {threshold_label}')
            
            if extreme_error_count > 0 and len(extreme_values) > 0:
                severity = extreme_values / threshold
                sorted_indices = np.argsort(extreme_values)
                y_positions = np.linspace(0.05, 0.95, len(extreme_values))
                y_positions = y_positions[sorted_indices]
                y_min, y_max = ax.get_ylim()
                if y_max <= 0:
                    y_max = 1
                y_scaled = y_min + y_positions * (y_max - y_min) * 0.1
                scatter = ax.scatter(extreme_values, y_scaled, c=severity,
                                    cmap='Reds', s=150, marker='x', linewidths=2.5,
                                    label=f'Extreme Errors ({extreme_error_count})',
                                    zorder=5, vmin=5, vmax=np.max(severity))
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                cbar.set_label('Severity (× threshold)', fontsize=9, rotation=270, labelpad=15)
            
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{title}\n'
                         f'Outliers: {outlier_count} ({outlier_count/total_count*100:.2f}%)\n'
                         f'Extreme Errors: {extreme_error_count} | NaN: {nan_count} | Inf: {inf_count}',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)
        
        # Select data based on data_type
        if data_type == 'state':
            wrist_translation_mag = wrist_translation_mag_state
            wrist_rotation_mag = wrist_rotation_mag_state
            fingertips_mag = fingertips_mag_state
        else:  # action
            wrist_translation_mag = wrist_translation_mag_action
            wrist_rotation_mag = wrist_rotation_mag_action
            fingertips_mag = fingertips_mag_action
        
        # Number of valid frames (after excluding episode boundaries)
        num_valid_frames = np.sum(valid_mask)
        
        # Row 0: Wrist Translation Outliers
        # 第一行: 手腕平移异常
        plot_outlier_distribution(fig.add_subplot(gs[0, idx]), wrist_translation_mag,
                                 THRESHOLD_WRIST_TRANSLATION,
                                 f'{name} - Wrist Translation Outlier Distribution ({type_label})',
                                 'Frame-to-Frame Wrist Translation Change (m)', 'm')
        
        # Row 1: Wrist Rotation Outliers
        # 第二行: 手腕旋转异常（使用唯一帧数统计）
        plot_outlier_distribution(fig.add_subplot(gs[1, idx]), wrist_rotation_mag,
                                 THRESHOLD_WRIST_ROTATION,
                                 f'{name} - Wrist Rotation Outlier Distribution ({type_label})',
                                 'Frame-to-Frame Wrist Rotation Change (rad)', 'rad',
                                 is_rotation=True, num_frames=num_valid_frames)
        
        # Row 2: Fingertips Outliers
        # 第三行: 指尖异常
        plot_outlier_distribution(fig.add_subplot(gs[2, idx]), fingertips_mag,
                                 THRESHOLD_FINGERTIPS_DISPLACEMENT,
                                 f'{name} - Fingertips Outlier Distribution ({type_label})',
                                 'Frame-to-Frame Fingertips Change (m)', 'm')
    
    plt.suptitle(f'Frame-to-Frame Change Rate Outlier Analysis ({type_label}, Absolute Threshold Method)', 
                fontsize=16, fontweight='bold', y=0.995)
    # Use subplots_adjust instead of tight_layout to avoid warning with colorbars
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.6, wspace=0.3)
    
    return fig

def configure_chinese_font():
    """
    Configure matplotlib to support Chinese characters.
    If no suitable font is found, download SimHei.ttf automatically.
    配置 matplotlib 以支持中文字符。
    如果找不到合适的字体，自动下载 SimHei.ttf。
    """
    # Common Chinese fonts to check
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'PingFang SC', 'Heiti TC', 'Noto Sans CJK SC']
    
    # Check if any system font matches
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    available_fonts = [f for f in chinese_fonts if f in system_fonts]
    
    font_path = None
    
    if available_fonts:
        print(f"Using system Chinese font: {available_fonts[0]}")
        plt.rcParams['font.sans-serif'] = [available_fonts[0]] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return

    # If no system font, check/download SimHei.ttf
    font_dir = Path.cwd() / "fonts"
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / "SimHei.ttf"
    
    if not font_path.exists():
        print(f"No system Chinese font found. Downloading SimHei.ttf to {font_path}...")
        url = "https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf"
        try:
            # Add headers to avoid 403 Forbidden
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, font_path)
            print("Downloaded SimHei.ttf successfully.")
        except Exception as e:
            print(f"Failed to download font: {e}")
            print("Chinese characters may not display correctly.")
            return

    if font_path.exists():
        try:
            # Register the custom font
            fm.fontManager.addfont(str(font_path))
            # Set as the first sans-serif font
            plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Loaded custom font from {font_path}")
        except Exception as e:
            print(f"Failed to load font from {font_path}: {e}")

# Set seaborn style and theme
# 设置 seaborn 样式和主题，用于美化图表
sns.set_theme(style="whitegrid", palette="husl", context="notebook")
sns.set_palette("Set2")

# Configure matplotlib parameters for better looking plots
# 配置 matplotlib 参数以获得更好看的图表
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
# # Initialize font configuration
# configure_chinese_font()

# Default Zarr file paths (used if not specified via command line)
# 默认 Zarr 文件路径（如果未通过命令行指定则使用）
if os.path.isdir(zarrdir):
    DEFAULT_ZARR_PATHS = get_zarrs(zarrdir, num)
    DEFAULT_DATASET_NAMES = get_names(zarrdir, num)
else:
    DEFAULT_ZARR_PATHS = []
    DEFAULT_DATASET_NAMES = []


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert rot6d representation to full 3x3 rotation matrix.
    将 6D 旋转表示法 (rot6d) 转换为完整的 3x3 旋转矩阵。

    rot6d format: [r11, r21, r31, r12, r22, r32] (first two columns of rotation matrix flattened)
    rot6d 格式: 旋转矩阵的前两列扁平化形式。

    Args:
        rot6d: Array of shape (N, 6) or (6,) containing rot6d vectors.

    Returns:
        rotation_matrices: Array of shape (N, 3, 3) or (3, 3).
    """
    rot6d = np.asarray(rot6d)
    original_shape = rot6d.shape
    if rot6d.ndim == 1:
        rot6d = rot6d.reshape(1, -1)
    
    # Extract first two columns (vectorized) / 提取前两列
    col1 = rot6d[:, :3]  # [r11, r21, r31]
    col2 = rot6d[:, 3:6]  # [r12, r22, r32]
    
    # Normalize column 1 / 归一化第一列
    col1_norm = col1 / (np.linalg.norm(col1, axis=1, keepdims=True) + 1e-8)
    
    # Normalize column 2
    # Note: A strict 6D->Matrix conversion should project col2 to be orthogonal to col1 (Gram-Schmidt).
    # The current implementation performs simple normalization, assuming inputs are already close to orthogonal.
    # 注意：严格的 6D 转换应包含 Gram-Schmidt 正交化投影。此处仅做归一化，假设输入已近似正交。
    col2_norm = col2 / (np.linalg.norm(col2, axis=1, keepdims=True) + 1e-8)
    
    # Recover third column via cross product: col3 = col1 x col2
    # 通过叉积恢复第三列 (右手定则)
    col3 = np.cross(col1_norm, col2_norm)
    col3_norm = col3 / (np.linalg.norm(col3, axis=1, keepdims=True) + 1e-8)
    
    # Reconstruct matrices: (N, 3, 3)
    # 重构完整旋转矩阵，列向量为 [col1, col2, col3]
    rotation_matrices = np.stack([col1_norm, col2_norm, col3_norm], axis=2)
    
    if len(original_shape) == 1:
        return rotation_matrices[0]
    return rotation_matrices


def rotation_matrix_to_angle(R: np.ndarray) -> np.ndarray:
    """
    Extract rotation angle from rotation matrix using trace formula
    使用迹公式 (Trace formula) 从旋转矩阵中提取旋转角度
    Angle = arccos((tr(R) - 1) / 2)
    
    Args:
        R: Array of shape (N, 3, 3) or (3, 3) containing rotation matrices
        
    Returns:
        rotation_angles: Array of shape (N,) or scalar containing rotation angles in radians
    """
    R = np.asarray(R)
    original_shape = R.shape
    if R.ndim == 2:
        R = R.reshape(1, 3, 3)
    
    N = R.shape[0]
    
    # Vectorized trace calculation: trace of each matrix along last two axes
    # 向量化计算迹: 沿最后两个轴计算对角线之和
    # R shape: (N, 3, 3), trace along axis1 and axis2
    traces = np.trace(R, axis1=1, axis2=2)
    
    # Clamp trace to [-1, 3] to avoid numerical errors (vectorized)
    # 将迹限制在合理范围内，避免数值误差导致 arccos 越界
    # For a rotation matrix, trace is 1 + 2cos(theta), so trace is in [-1, 3]
    trace_clamped = np.clip(traces, -1.0, 3.0)
    
    # Compute angles (vectorized)
    # 计算角度 (弧度制)
    angles = np.arccos(np.clip((trace_clamped - 1.0) / 2.0, -1.0, 1.0))
    
    # Handle edge cases (vectorized)
    # 处理 NaN 或 Inf 的情况
    rotation_angles = np.where(np.isnan(angles) | np.isinf(angles), 0.0, angles)
    
    # Return in original shape
    if len(original_shape) == 2:
        return rotation_angles[0]
    return rotation_angles


def compute_rotation_angle_from_rot6d_diff(rot6d_state: np.ndarray, rot6d_action: np.ndarray) -> np.ndarray:
    """
    Compute rotation angle difference between two rot6d representations
    计算两个 rot6d 表示之间的旋转角度差
    
    This is the CORRECT way to compute rotation differences:
    这是计算旋转差异的正确方法：
    1. Convert rot6d_state and rot6d_action to rotation matrices (转为旋转矩阵)
    2. Compute relative rotation: R_diff = R_action @ R_state^T (计算相对旋转)
    3. Extract angle from R_diff (从相对旋转矩阵提取角度)
    
    Args:
        rot6d_state: Array of shape (N, 6) containing state rot6d vectors
        rot6d_action: Array of shape (N, 6) containing action rot6d vectors
        
    Returns:
        rotation_angles: Array of shape (N,) containing rotation angles in radians (弧度)
    """
    rot6d_state = np.asarray(rot6d_state)
    rot6d_action = np.asarray(rot6d_action)
    
    # Ensure same shape
    if rot6d_state.ndim == 1:
        rot6d_state = rot6d_state.reshape(1, -1)
    if rot6d_action.ndim == 1:
        rot6d_action = rot6d_action.reshape(1, -1)
    
    assert rot6d_state.shape == rot6d_action.shape, "State and action rot6d must have same shape"
    
    N = rot6d_state.shape[0]
    rotation_angles = np.zeros(N)
    
    # Convert to rotation matrices (vectorized)
    # 转换为旋转矩阵
    R_state = rot6d_to_rotation_matrix(rot6d_state)  # Shape: (N, 3, 3)
    R_action = rot6d_to_rotation_matrix(rot6d_action)  # Shape: (N, 3, 3)
    
    # Compute relative rotation (vectorized): R_diff = R_action @ R_state^T
    # 计算相对旋转矩阵。注意 R_state^T 是 R_state 的逆（对于旋转矩阵）
    # Use einsum for batch matrix multiplication: 'nij,njk->nik'
    # R_state.T has shape (N, 3, 3), we need to transpose last two dims: (N, 3, 3) -> (N, 3, 3)
    R_state_T = np.transpose(R_state, (0, 2, 1))  # Transpose last two dimensions
    R_diff = np.einsum('nij,njk->nik', R_action, R_state_T)  # Batch matrix multiplication
    
    # Extract angles (vectorized)
    # 提取角度
    rotation_angles = rotation_matrix_to_angle(R_diff)
    
    return rotation_angles


def calculate_episode_lengths(episode_ends: np.ndarray) -> np.ndarray:
    """
    Calculate length of each episode from episode_ends
    根据 episode_ends 计算每个 episode 的长度
    
    Args:
        episode_ends: Array of cumulative episode end indices (累积结束索引)
        
    Returns:
        episode_lengths: Array of episode lengths (每集长度)
    """
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts
    return episode_lengths


def get_episode_info_for_frames(frame_indices: np.ndarray, episode_ends: np.ndarray, episode_names: np.ndarray) -> List[Dict]:
    """
    Get episode information for given frame indices
    根据帧索引获取对应的 episode 信息
    
    Args:
        frame_indices: Array of frame indices (帧索引数组)
        episode_ends: Array of cumulative episode end indices (累积结束索引)
        episode_names: Array of episode names (episode 名称数组)
        
    Returns:
        List of dictionaries containing frame_idx and episode_name
        包含 frame_idx 和 episode_name 的字典列表
    """
    episode_info = []
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    for frame_idx in frame_indices:
        # Find which episode this frame belongs to
        # 找到该帧属于哪个 episode
        episode_idx = np.searchsorted(episode_ends, frame_idx, side='right')
        if episode_idx < len(episode_names):
            episode_name = str(episode_names[episode_idx])
            episode_start = episode_starts[episode_idx]
            episode_end = episode_ends[episode_idx]
            frame_in_episode = int(frame_idx - episode_start)
            
            episode_info.append({
                'frame_idx': int(frame_idx),
                'episode_idx': int(episode_idx),
                'episode_name': episode_name,
                'frame_in_episode': frame_in_episode,
                'episode_start': int(episode_start),
                'episode_end': int(episode_end)
            })
    
    return episode_info


def create_statistics_table(datasets_info: Dict) -> plt.Figure:
    """
    Create an enhanced table showing basic statistics for all datasets
    创建一个增强的统计表格，显示所有数据集的基本统计信息
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    # 准备表格数据: 数据集名称，Episode数量，总帧数，平均/中位/最小/最大长度，标准差
    headers = ['Dataset', 'Num Episodes', 'Total Frames', 'Avg Length', 
               'Median Length', 'Min Length', 'Max Length', 'Std Length']
    table_data = []
    
    for name, info in datasets_info.items():
        lengths = info['lengths']
        row = [
            name,
            f"{len(lengths):,}",
            f"{info['episode_ends'][-1]:,}",
            f"{np.mean(lengths):.2f}",
            f"{np.median(lengths):.2f}",
            f"{np.min(lengths):,}",
            f"{np.max(lengths):,}",
            f"{np.std(lengths):.2f}"
        ]
        table_data.append(row)
    
    # Create table
    # 创建表格对象
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.13, 0.12, 0.12, 0.11, 0.12, 0.11, 0.11, 0.11])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Use seaborn color palette for header
    # 设置表头颜色
    header_color = sns.color_palette("deep")[2]  # Green from deep palette
    
    # Style header
    # 样式化表头
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(2)
    
    # Style data rows with seaborn colors
    # 样式化数据行 (交替颜色)
    row_colors = sns.color_palette("pastel", n_colors=len(table_data))
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_colors[i-1])
            table[(i, j)].set_edgecolor('white')
            table[(i, j)].set_linewidth(1.5)
            table[(i, j)].set_text_props(fontsize=10)
            
            # Make dataset name column bold
            if j == 0:
                table[(i, j)].set_text_props(weight='bold', fontsize=11)
    
    plt.title('Episode Statistics Summary', 
             fontsize=17, fontweight='bold', pad=20)
    
    return fig


def plot_episode_length_distribution(datasets_info: Dict) -> plt.Figure:
    """
    Plot histogram with KDE of episode lengths for all datasets using seaborn
    使用 seaborn 绘制所有数据集 Episode 长度的直方图和核密度估计 (KDE)
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6*num_datasets, 5))
    
    # Use seaborn color palette
    colors = sns.color_palette("Set2", n_colors=num_datasets)
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = [axes]
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        lengths = info['lengths']
        ax = axes[idx]
        
        # Use seaborn histplot with KDE overlay
        # 绘制直方图并叠加 KDE 曲线
        sns.histplot(lengths, bins=50, kde=True, ax=ax, 
                    color=colors[idx], edgecolor='black', 
                    alpha=0.6, linewidth=1.5, line_kws={'linewidth': 2})
        
        # Add mean and median lines with better styling
        # 添加均值和中位数竖线
        mean_val = np.mean(lengths)
        median_val = np.median(lengths)
        
        ax.axvline(mean_val, color='#E74C3C', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {mean_val:.1f}', alpha=0.8)
        ax.axvline(median_val, color='#27AE60', linestyle='--', 
                   linewidth=2.5, label=f'Median: {median_val:.1f}', alpha=0.8)
        
        ax.set_xlabel('Episode Length (frames)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}\n({len(lengths):,} episodes)', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Remove top and right spines for cleaner look
        sns.despine(ax=ax)
    
    plt.suptitle('Episode Length Distribution with KDE', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


# 注释掉: 新数据集不包含 presence 字段
# def plot_presence_distribution(datasets_info: Dict) -> plt.Figure:
#     """
#     Plot presence distribution for all datasets using seaborn
#     绘制手部存在情况分布 (饼图)
#     Presence: 1=left hand, 2=right hand, 3=both hands
#     1=仅左手, 2=仅右手, 3=双手
#     
#     Args:
#         datasets_info: Dictionary containing episode data for each dataset
#         
#     Returns:
#         matplotlib Figure object
#     """
#     num_datasets = len(datasets_info)
#     fig, axes = plt.subplots(1, num_datasets, figsize=(6*num_datasets, 5))
#     
#     # Handle single dataset case
#     if num_datasets == 1:
#         axes = [axes]
#     
#     presence_labels = {1: 'Left Only', 2: 'Right Only', 3: 'Both Hands'}
#     palette = ['#FFB6C1', '#87CEEB', '#90EE90']
#     
#     for idx, (name, info) in enumerate(datasets_info.items()):
#         presence = info['presence']
#         
#         # Create DataFrame for seaborn
#         df = pd.DataFrame({'presence': presence})
#         df['presence_label'] = df['presence'].map(presence_labels)
#         
#         # Count occurrences for pie chart
#         value_counts = df['presence_label'].value_counts()
#         
#         # Pie charts
#         # 绘制饼图
#         ax_pie = axes[idx]
#         wedges, texts, autotexts = ax_pie.pie(
#             value_counts.values,
#             labels=value_counts.index,
#             colors=palette[:len(value_counts)],
#             autopct=lambda pct: f'{pct:.1f}%',
#             startangle=90,
#             textprops={'fontsize': 11, 'weight': 'bold'},
#             explode=[0.05] * len(value_counts)  # Slightly separate slices
#         )
#         
#         for autotext in autotexts:
#             autotext.set_color('white')
#             autotext.set_fontsize(10)
#         
#         ax_pie.set_title(f'{name}\n({len(presence):,} frames)', 
#                         fontsize=13, fontweight='bold', pad=10)
#     
#     plt.suptitle('Hand Presence Distribution Analysis', 
#                 fontsize=16, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     
#     return fig


def plot_instruction_num_distribution(datasets_info: Dict) -> plt.Figure:
    """
    Plot instruction_num distribution for all datasets using seaborn
    绘制每帧有效指令数量分布的直方图
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6*num_datasets, 5))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = [axes]
    
    # Use seaborn color palette
    palette = sns.color_palette("viridis", n_colors=6)
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        instruction_num = info['instruction_num']
        ax = axes[idx]
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({'instruction_num': instruction_num})
        
        # Use seaborn countplot with hue to avoid deprecation warning
        # 使用 seaborn 的 countplot 绘制频率分布
        unique_values = df['instruction_num'].unique()
        n_colors_needed = len(unique_values)
        palette_subset = palette[:n_colors_needed]
        
        sns.countplot(data=df, x='instruction_num', ax=ax, 
                     hue='instruction_num', palette=palette_subset, 
                     edgecolor='black', linewidth=1.5)
        
        # Remove legend (for compatibility with older matplotlib/seaborn versions)
        if ax.legend_ is not None:
            ax.legend_.remove()
        
        # Add value labels on bars with percentages
        # 在柱状图上方添加数值和百分比标签
        for container in ax.containers:
            labels = []
            for v in container:
                height = v.get_height()
                percentage = height / len(instruction_num) * 100
                labels.append(f'{int(height):,}\n({percentage:.1f}%)')
            ax.bar_label(container, labels=labels, fontsize=9, 
                        fontweight='bold', padding=3)
        
        ax.set_xlabel('Number of Instructions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frame Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}\n({len(instruction_num):,} frames)', 
                    fontsize=13, fontweight='bold', pad=20)
        
        # Add statistics text with better styling
        # 添加均值和中位数统计信息
        mean_val = np.mean(instruction_num)
        median_val = np.median(instruction_num)
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#FFF9C4', 
                        edgecolor='black', linewidth=1.5, alpha=0.9))
        
        # Remove top and right spines
        sns.despine(ax=ax)
    
    plt.suptitle('Instruction Number Distribution', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def clean_instruction_prefix(inst_str: str) -> str:
    """
    Remove number prefix from instruction (e.g., "1. ", "2. ", "3. ")
    移除指令字符串中的数字前缀 (例如 "1. ", "2. ")
    
    Args:
        inst_str: Instruction string
        
    Returns:
        Cleaned instruction string without number prefix
    """
    cleaned = inst_str.strip()
    # Remove pattern like "1. ", "2. ", "10. " etc.
    # Check if starts with digit followed by dot
    if len(cleaned) > 1 and cleaned[0].isdigit() and cleaned[1] == '.':
        # Remove "digit. " pattern
        if len(cleaned) > 2 and cleaned[2] == ' ':
            cleaned = cleaned[3:].strip()
        else:
            # Remove "digit." pattern (without space)
            cleaned = cleaned[2:].strip()
    return cleaned


def process_instructions_vectorized(instruction, instruction_num, filter_errors=True, clean_prefix=True):
    """
    Fully vectorized processing of instructions for maximum performance
    完全向量化的指令处理函数，以实现最大性能
    
    Args:
        instruction: Array of shape (frames, 5) containing instruction strings (指令字符串数组)
        instruction_num: Array of shape (frames,) containing number of valid instructions per frame (每帧有效指令数)
        filter_errors: Whether to filter out error messages (是否过滤错误信息)
        clean_prefix: Whether to clean number prefixes (是否清除数字前缀)
        
    Returns:
        all_instructions: List of processed instruction strings (处理后的指令列表)
        total_checked_slots: Total number of slots checked (总共检查的槽位数量)
        non_empty_count: Number of non-empty instructions (非空指令数量)
        unique_instructions: Set of unique instruction strings (唯一指令集合)
    """
    # Convert to numpy array for faster processing
    instruction = np.asarray(instruction, dtype=object)
    instruction_num = np.asarray(instruction_num, dtype=np.int64)
    
    num_frames = len(instruction)
    max_slots = instruction.shape[1] if len(instruction.shape) > 1 else 5
    
    # Create 2D mask for valid slots using vectorized operations (much faster than list comprehension)
    # 使用向量化操作创建有效槽位的 2D 掩码（比列表推导式快得多）
    # For each frame, mark slots 0 to min(instruction_num[frame_idx], max_slots) as valid
    # This matches the original logic: min(num_inst, len(frame_instructions))
    slot_indices = np.arange(max_slots)
    
    # Clip instruction_num to max_slots to match original logic exactly
    instruction_num_clipped = np.minimum(instruction_num, max_slots)
    
    # Broadcast comparison: (num_frames, max_slots) boolean array
    # 广播比较: (num_frames, max_slots) 的布尔数组
    # True where slot_idx < instruction_num_clipped[frame_idx]
    valid_slots_mask = slot_indices[None, :] < instruction_num_clipped[:, None]
    
    # Extract valid instructions using boolean indexing (fully vectorized)
    # 使用布尔索引提取有效指令（完全向量化）
    all_instructions_flat = instruction[valid_slots_mask]
    
    # Calculate total checked slots (vectorized)
    # Use instruction_num_clipped to match the mask calculation
    total_checked_slots = int(np.sum(instruction_num_clipped))
    
    # Convert to strings and strip, handling bytes and encoding issues
    # 转换为字符串并去除首尾空格，处理字节串和编码问题
    all_instructions_str = []
    for inst in all_instructions_flat:
        try:
            # Handle bytes objects
            if isinstance(inst, bytes):
                inst_str = inst.decode('utf-8', errors='ignore').strip()
            else:
                inst_str = str(inst).strip()
            
            # Check if it's printable and not just control characters
            # Fix: remove whitespace/newlines before checking isprintable to allow multi-line instructions
            check_str = "".join(inst_str.split())
            if inst_str and check_str.isprintable():
                all_instructions_str.append(inst_str)
            else:
                all_instructions_str.append('')
        except Exception:
            all_instructions_str.append('')
    
    # Filter out empty strings and 'nan' (simple operation, use list comprehension)
    # 过滤掉空字符串和 'nan'
    mask_valid = [(s != '') and (s != 'nan') for s in all_instructions_str]
    
    # Only use pandas for complex operations (regex, contains)
    # 仅在需要复杂操作（正则、包含匹配）时使用 pandas
    if filter_errors or clean_prefix:
        # Convert to pandas Series only when needed
        df_inst_str = pd.Series(all_instructions_str, dtype=object)
        mask_valid_pd = pd.Series(mask_valid, dtype=bool)
        
        if filter_errors:
            # Filter out error messages (vectorized, case-insensitive)
            # Use single regex pattern for better performance
            # 过滤包含错误信息的指令 (不区分大小写，向量化正则匹配)
            mask_no_error = ~df_inst_str.str.lower().str.contains('error generating description|error', na=False, regex=True)
            mask_valid_pd = mask_valid_pd & mask_no_error
        
        if clean_prefix:
            # Vectorized prefix cleaning using regex (single pass)
            # Match original clean_instruction_prefix behavior: only single digit prefix (e.g., "1. ", "2. ")
            # Pattern: start with single digit followed by dot and optional space
            # Note: Original function only matches single digit, not multi-digit (e.g., "10. " won't match)
            # 向量化前缀清洗
            df_inst_valid = df_inst_str[mask_valid_pd]
            df_inst_cleaned = df_inst_valid.str.replace(r'^\d\.\s*', '', regex=True).str.strip()
            # Filter out empty strings after cleaning
            mask_valid_cleaned = (df_inst_cleaned != '')
            all_instructions = df_inst_cleaned[mask_valid_cleaned].tolist()
        else:
            all_instructions = df_inst_str[mask_valid_pd].tolist()
    else:
        # Simple case: no complex operations needed
        all_instructions = [s for s, valid in zip(all_instructions_str, mask_valid) if valid]
    
    # Calculate statistics
    non_empty_count = len(all_instructions)
    unique_instructions = set(all_instructions)
    
    return all_instructions, total_checked_slots, non_empty_count, unique_instructions


def plot_top_instructions(datasets_info: Dict, top_n: int = 20) -> plt.Figure:
    """
    Plot top N most frequent instructions as horizontal bar chart using seaborn
    绘制出现频率最高的 Top N 指令的水平条形图
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        top_n: Number of top instructions to show (显示的 Top 指令数量)
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    fig, axes = plt.subplots(num_datasets, 1, figsize=(15, 5.5*num_datasets))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = [axes]
    
    # Use seaborn color palettes for each dataset
    # Extend palette list to support more datasets
    available_palettes = ['Reds_r', 'Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r', 'Greys_r', 'YlOrRd_r', 'YlGnBu_r']
    palettes = (available_palettes * ((num_datasets // len(available_palettes)) + 1))[:num_datasets]
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        instruction = info['instruction']
        instruction_num = info['instruction_num']
        ax = axes[idx]
        
        # Use vectorized processing for better performance
        # 使用向量化处理获取指令列表
        all_instructions, _, _, _ = process_instructions_vectorized(
            instruction, instruction_num, 
            filter_errors=True, clean_prefix=True
        )
        
        # Count frequency
        from collections import Counter
        instruction_counter = Counter(all_instructions)
        top_instructions = instruction_counter.most_common(top_n)
        
        # Debug output for top instructions to verify content (especially for encoding issues)
        print(f"\nTop {len(top_instructions)} instructions for {name}:")
        for i, (inst, count) in enumerate(top_instructions):
            # Print repr to show hidden characters/encoding issues
            print(f"  {i+1}. {repr(inst)}: {count}")
        
        if len(top_instructions) == 0:
            ax.text(0.5, 0.5, 'No valid instructions found', 
                   ha='center', va='center', fontsize=14, 
                   transform=ax.transAxes, fontweight='bold')
            ax.set_title(f'{name} - Top Instructions', 
                        fontsize=13, fontweight='bold')
            continue
        
        # Create DataFrame for seaborn
        df = pd.DataFrame(top_instructions, columns=['instruction', 'count'])
        df['instruction'] = df['instruction'].apply(
            lambda x: x[:60] + '...' if len(x) > 60 else x
        )
        df = df.sort_values('count', ascending=True)  # For horizontal bar
        
        # Use seaborn barplot with hue to avoid deprecation warning
        colors_gradient = sns.color_palette(palettes[idx], n_colors=len(df))
        sns.barplot(data=df, y='instruction', x='count', ax=ax,
                   hue='instruction', palette=colors_gradient,
                   edgecolor='black', linewidth=1.2)
        
        # Remove legend (for compatibility with older matplotlib/seaborn versions)
        if ax.legend_ is not None:
            ax.legend_.remove()
        
        # Add count labels (vectorized: use itertuples instead of iterrows for better performance)
        # 添加计数标签
        for i, row in enumerate(df.itertuples()):
            ax.text(row.count, i, f" {row.count:,}",
                   ha='left', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='none'))
        
        ax.set_ylabel('')
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} - Top {len(top_instructions)} Instructions\n'
                    f'(Total unique: {len(instruction_counter):,})', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=9)
        
        # Remove top and right spines
        sns.despine(ax=ax)
    
    plt.suptitle('Most Frequent Instructions', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def create_data_quality_table(datasets_info: Dict) -> plt.Figure:
    """
    Create an enhanced table showing data quality metrics
    创建数据质量指标表格
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Dataset', 'Total Frames', 'Episodes', 'Presence\nValid %', 
               'Instruction\nNon-Empty %', 'Inst Num\nRange', 
               'Unique\nInstructions', 'Avg Inst\nPer Frame']
    table_data = []
    
    for name, info in datasets_info.items():
        total_frames = len(info['presence'])
        num_episodes = len(info['episode_names'])
        
        # Presence analysis
        # 手部存在数据有效性分析 (1-3为有效值)
        presence = info['presence']
        valid_presence = np.sum((presence >= 1) & (presence <= 3))
        presence_valid_pct = valid_presence / len(presence) * 100
        
        # Instruction analysis
        instruction = info['instruction']
        instruction_num = info['instruction_num']
        
        # Use vectorized processing for better performance
        _, total_checked_slots, non_empty_count, unique_instructions = process_instructions_vectorized(
            instruction, instruction_num,
            filter_errors=False, clean_prefix=False
        )
        
        # Calculate percentage: non-empty slots / total checked slots
        non_empty_pct = (float(non_empty_count) / float(total_checked_slots) * 100) if total_checked_slots > 0 else 0.0
        
        inst_num_range = f"{np.min(instruction_num)}-{np.max(instruction_num)}"
        avg_inst_per_frame = np.mean(instruction_num)
        
        row = [
            name,
            f"{total_frames:,}",
            f"{num_episodes:,}",
            f"{presence_valid_pct:.2f}%",
            f"{non_empty_pct:.2f}%",
            inst_num_range,
            f"{len(unique_instructions):,}",
            f"{avg_inst_per_frame:.2f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.12, 0.10, 0.12, 0.14, 0.10, 0.14, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)
    
    # Use seaborn color for header
    header_color = sns.color_palette("deep")[0]  # Blue from deep palette
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(2)
    
    # Style data rows with seaborn colors
    row_colors = sns.color_palette("pastel", n_colors=len(table_data))
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_colors[i-1])
            table[(i, j)].set_edgecolor('white')
            table[(i, j)].set_linewidth(1.5)
            table[(i, j)].set_text_props(fontsize=10)
            
            # Make dataset name column bold
            if j == 0:
                table[(i, j)].set_text_props(weight='bold', fontsize=11)
    
    plt.title('Data Quality Analysis', 
             fontsize=17, fontweight='bold', pad=20)
    
    return fig


def plot_data_quality_and_presence(datasets_info: Dict) -> plt.Figure:
    """
    Plot data quality table (presence analysis removed for new dataset structure)
    显示数据质量表格 (新数据集结构中已移除 presence 分析)
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    fig = plt.figure(figsize=(16, 6))
    
    # Data quality table
    # 数据质量表格
    ax_table = fig.add_subplot(111)
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Prepare table data (removed Presence column for new dataset)
    headers = ['Dataset', 'Total Frames', 'Instruction\nNon-Empty %', 'Inst Num\nRange', 
               'Unique\nInstructions', 'Avg Inst\nPer Frame']
    table_data = []
    
    for name, info in datasets_info.items():
        # Instruction analysis
        instruction = info['instruction']
        instruction_num = info['instruction_num']
        total_frames = len(instruction)
        
        # Use vectorized processing for better performance
        _, total_checked_slots, non_empty_count, unique_instructions = process_instructions_vectorized(
            instruction, instruction_num,
            filter_errors=False, clean_prefix=False
        )
        
        non_empty_pct = (float(non_empty_count) / float(total_checked_slots) * 100) if total_checked_slots > 0 else 0.0
        inst_num_range = f"{np.min(instruction_num)}-{np.max(instruction_num)}"
        avg_inst_per_frame = np.mean(instruction_num)
        
        row = [
            name,
            f"{total_frames:,}",
            f"{non_empty_pct:.2f}%",
            inst_num_range,
            f"{len(unique_instructions):,}",
            f"{avg_inst_per_frame:.2f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax_table.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.18, 0.12, 0.20, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    header_color = sns.color_palette("deep")[0]
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(2)
    
    # Style data rows
    row_colors = sns.color_palette("pastel", n_colors=len(table_data))
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_colors[i-1])
            table[(i, j)].set_edgecolor('white')
            table[(i, j)].set_linewidth(1.5)
            table[(i, j)].set_text_props(fontsize=10)
            if j == 0:
                table[(i, j)].set_text_props(weight='bold', fontsize=11)
    
    ax_table.set_title('Data Quality Metrics', 
             fontsize=15, fontweight='bold', pad=15)
    
    # Add overall title
    fig.suptitle('Data Quality Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def plot_dataset_size_pie_charts(datasets_info: Dict) -> plt.Figure:
    """
    Plot enhanced pie charts showing dataset size comparison for episodes and frames
    绘制数据集大小对比饼图 (Episode 数量和帧数)
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    names = list(datasets_info.keys())
    num_episodes = [len(info['lengths']) for info in datasets_info.values()]
    total_frames = [info['episode_ends'][-1] for info in datasets_info.values()]
    
    # Use seaborn color palette
    colors = sns.color_palette("Set2", n_colors=len(names))
    explode = tuple([0.08] * len(names))  # Slightly separate slices
    
    # Pie chart for episodes
    # Episode 数量分布饼图
    wedges1, texts1, autotexts1 = ax1.pie(
        num_episodes, 
        labels=names, 
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(num_episodes)):,})',
        startangle=90,
        explode=explode,
        textprops={'fontsize': 12, 'weight': 'bold'},
        pctdistance=0.75
    )
    
    # Style percentage text
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    # Style labels
    for text in texts1:
        text.set_fontsize(12)
        text.set_weight('bold')
    
    ax1.set_title('Episode Count Distribution', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Pie chart for frames
    # 帧数分布饼图
    wedges2, texts2, autotexts2 = ax2.pie(
        total_frames, 
        labels=names, 
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(total_frames)):,})',
        startangle=90,
        explode=explode,
        textprops={'fontsize': 12, 'weight': 'bold'},
        pctdistance=0.75
    )
    
    # Style percentage text
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    # Style labels
    for text in texts2:
        text.set_fontsize(12)
        text.set_weight('bold')
    
    ax2.set_title('Frame Count Distribution', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.suptitle('Dataset Size Comparison', 
                fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


# ============================================================================
# Phase 1: Basic Extensions (Lightweight)
# ============================================================================

def plot_camera_parameters(datasets_info: Dict) -> plt.Figure:
    """
    Plot camera intrinsic parameters analysis for both head and breast cameras
    绘制相机内参分析图 (头部和胸部摄像机)
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    # 4 rows: Head Focal, Head Principal Point, Breast Focal, Breast Principal Point
    fig = plt.figure(figsize=(6*num_datasets, 16))
    gs = fig.add_gridspec(4, num_datasets, hspace=0.3, wspace=0.3)
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        intrinsic_h = info['intrinsic_h']  # (frames, 4): fx, fy, cx, cy for head camera
        intrinsic_b = info['intrinsic_b']  # (frames, 4): fx, fy, cx, cy for breast camera
        
        # Head camera parameters
        fx_h = intrinsic_h[:, 0]
        fy_h = intrinsic_h[:, 1]
        cx_h = intrinsic_h[:, 2]
        cy_h = intrinsic_h[:, 3]
        
        # Breast camera parameters
        fx_b = intrinsic_b[:, 0]
        fy_b = intrinsic_b[:, 1]
        cx_b = intrinsic_b[:, 2]
        cy_b = intrinsic_b[:, 3]
        
        # Row 0: Head Camera Focal lengths
        ax1 = fig.add_subplot(gs[0, idx])
        fx_unique_h = len(np.unique(fx_h))
        fy_unique_h = len(np.unique(fy_h))
        
        if fx_unique_h == 1 and fy_unique_h == 1:
            ax1.text(0.5, 0.5, f'fx = {fx_h[0]:.2f}\nfy = {fy_h[0]:.2f}\n(Constant)', 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
        else:
            sns.histplot(fx_h, bins=50, ax=ax1, label='fx', alpha=0.6, color='#FF6B6B', 
                        edgecolor='black', kde=True)
            sns.histplot(fy_h, bins=50, ax=ax1, label='fy', alpha=0.6, color='#4ECDC4', 
                        edgecolor='black', kde=True)
            ax1.set_xlabel('Focal Length', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax1.legend()
        ax1.set_title(f'{name} - Focal Lengths (Head)', fontsize=12, fontweight='bold')
        sns.despine(ax=ax1)
        
        # Row 1: Head Camera Principal point
        ax2 = fig.add_subplot(gs[1, idx])
        cx_unique_h = len(np.unique(cx_h))
        cy_unique_h = len(np.unique(cy_h))
        
        if cx_unique_h == 1 and cy_unique_h == 1:
            ax2.scatter(cx_h[0], cy_h[0], s=200, color='#45B7D1', edgecolor='black', 
                       linewidth=2, zorder=3, alpha=0.8)
            ax2.annotate(f'cx = {cx_h[0]:.2f}\ncy = {cy_h[0]:.2f}', 
                        xy=(cx_h[0], cy_h[0]), xytext=(10, 10),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            ax2.set_xlim(cx_h[0] - 10, cx_h[0] + 10)
            ax2.set_ylim(cy_h[0] - 10, cy_h[0] + 10)
        else:
            sample_size = min(10000, len(cx_h))
            if sample_size < len(cx_h):
                sample_indices = np.linspace(0, len(cx_h)-1, sample_size, dtype=int)
                cx_sample = cx_h[sample_indices]
                cy_sample = cy_h[sample_indices]
            else:
                cx_sample = cx_h
                cy_sample = cy_h
            ax2.scatter(cx_sample, cy_sample, alpha=0.1, s=1, color='#45B7D1')
        ax2.set_xlabel('Principal Point X (cx)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Principal Point Y (cy)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{name} - Principal Point (Head)', fontsize=12, fontweight='bold')
        sns.despine(ax=ax2)
        
        # Row 2: Breast Camera Focal lengths
        ax3 = fig.add_subplot(gs[2, idx])
        fx_unique_b = len(np.unique(fx_b))
        fy_unique_b = len(np.unique(fy_b))
        
        if fx_unique_b == 1 and fy_unique_b == 1:
            ax3.text(0.5, 0.5, f'fx = {fx_b[0]:.2f}\nfy = {fy_b[0]:.2f}\n(Constant)', 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        else:
            sns.histplot(fx_b, bins=50, ax=ax3, label='fx', alpha=0.6, color='#FF6B6B', 
                        edgecolor='black', kde=True)
            sns.histplot(fy_b, bins=50, ax=ax3, label='fy', alpha=0.6, color='#4ECDC4', 
                        edgecolor='black', kde=True)
            ax3.set_xlabel('Focal Length', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax3.legend()
        ax3.set_title(f'{name} - Focal Lengths (Breast)', fontsize=12, fontweight='bold')
        sns.despine(ax=ax3)
        
        # Row 3: Breast Camera Principal point
        ax4 = fig.add_subplot(gs[3, idx])
        cx_unique_b = len(np.unique(cx_b))
        cy_unique_b = len(np.unique(cy_b))
        
        if cx_unique_b == 1 and cy_unique_b == 1:
            ax4.scatter(cx_b[0], cy_b[0], s=200, color='#45B7D1', edgecolor='black', 
                       linewidth=2, zorder=3, alpha=0.8)
            ax4.annotate(f'cx = {cx_b[0]:.2f}\ncy = {cy_b[0]:.2f}', 
                        xy=(cx_b[0], cy_b[0]), xytext=(10, 10),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            ax4.set_xlim(cx_b[0] - 10, cx_b[0] + 10)
            ax4.set_ylim(cy_b[0] - 10, cy_b[0] + 10)
        else:
            sample_size = min(10000, len(cx_b))
            if sample_size < len(cx_b):
                sample_indices = np.linspace(0, len(cx_b)-1, sample_size, dtype=int)
                cx_sample = cx_b[sample_indices]
                cy_sample = cy_b[sample_indices]
            else:
                cx_sample = cx_b
                cy_sample = cy_b
            ax4.scatter(cx_sample, cy_sample, alpha=0.1, s=1, color='#45B7D1')
        ax4.set_xlabel('Principal Point X (cx)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Principal Point Y (cy)', fontsize=11, fontweight='bold')
        ax4.set_title(f'{name} - Principal Point (Breast)', fontsize=12, fontweight='bold')
        sns.despine(ax=ax4)
    
    plt.suptitle('Camera Intrinsic Parameters Analysis (Head & Breast Cameras)', fontsize=16, fontweight='bold', y=0.998)
    
    return fig


def plot_data_quality_anomalies(datasets_info: Dict) -> plt.Figure:
    """
    Detect and visualize data quality anomalies using IQR method for both head and breast cameras
    使用 IQR (四分位距) 方法检测并可视化数据异常 (头部和胸部相机的手腕和指尖数据)
    IQR method: Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    # 6 rows: Head Trans, Head Rot, Head Fingertips, Breast Trans, Breast Rot, Breast Fingertips
    fig, axes = plt.subplots(6, num_datasets, figsize=(6*num_datasets, 20))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Helper function to detect outliers using IQR method
    def detect_outliers_iqr(data):
        """
        Detect outliers using IQR method
        IQR 异常检测辅助函数
        Returns: (normal_count, outlier_count, lower_bound, upper_bound, outlier_pct)
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Outliers are values outside [lower_bound, upper_bound]
        # 异常值定义为超出 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 范围的值
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_count = np.sum(outlier_mask)
        normal_count = len(data) - outlier_count
        outlier_pct = outlier_count / len(data) * 100
        
        return normal_count, outlier_count, lower_bound, upper_bound, outlier_pct
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        # Load both head and breast camera data
        wrist_state_h = info['wrist_state_h']
        wrist_action_h = info['wrist_action_h']
        fingertips_state_h = info['fingertips_state_h']
        fingertips_action_h = info['fingertips_action_h']
        
        wrist_state_b = info['wrist_state_b']
        wrist_action_b = info['wrist_action_b']
        fingertips_state_b = info['fingertips_state_b']
        fingertips_action_b = info['fingertips_action_b']
        
        # Row 0: Head Camera - Wrist Translation Anomalies
        ax1 = axes[0, idx]
        wrist_diff_h = np.abs(wrist_action_h - wrist_state_h)
        translation_diff_h = wrist_diff_h[:, :6]
        translation_magnitude_h = np.linalg.norm(translation_diff_h, axis=1)
        normal_count_trans_h, outlier_count_trans_h, lb_trans_h, ub_trans_h, outlier_pct_trans_h = detect_outliers_iqr(translation_magnitude_h)
        
        categories = ['Normal', 'Outliers']
        counts_h = [normal_count_trans_h, outlier_count_trans_h]
        colors_bar = ['#96CEB4', '#FF6B6B']
        
        bars = ax1.bar(categories, counts_h, color=colors_bar, edgecolor='black', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(translation_magnitude_h)*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
        ax1.set_title(f'{name} - Wrist Translation (Head Camera)\n'
                     f'Outlier: {outlier_pct_trans_h:.2f}% | Range: [{lb_trans_h:.4f}, {ub_trans_h:.4f}]m', 
                     fontsize=12, fontweight='bold', pad=20)
        sns.despine(ax=ax1)
        
        # Row 1: Head Camera - Wrist Rotation Anomalies
        ax2 = axes[1, idx]
        left_rot6d_state_h = wrist_state_h[:, 6:12]
        right_rot6d_state_h = wrist_state_h[:, 12:18]
        left_rot6d_action_h = wrist_action_h[:, 6:12]
        right_rot6d_action_h = wrist_action_h[:, 12:18]
        
        left_rotation_mag_h = compute_rotation_angle_from_rot6d_diff(left_rot6d_state_h, left_rot6d_action_h)
        right_rotation_mag_h = compute_rotation_angle_from_rot6d_diff(right_rot6d_state_h, right_rot6d_action_h)
        rotation_magnitude_h = np.concatenate([left_rotation_mag_h, right_rotation_mag_h])
        
        Q1_h = np.percentile(rotation_magnitude_h, 25)
        Q3_h = np.percentile(rotation_magnitude_h, 75)
        IQR_h = Q3_h - Q1_h
        lower_bound_h = Q1_h - 1.5 * IQR_h
        upper_bound_h = Q3_h + 1.5 * IQR_h
        outlier_mask_h = (rotation_magnitude_h < lower_bound_h) | (rotation_magnitude_h > upper_bound_h)
        
        outlier_indices_h = np.where(outlier_mask_h)[0]
        outlier_frames_unique_h = np.unique(outlier_indices_h % len(wrist_state_h))
        outlier_count_rot_h = len(outlier_frames_unique_h)
        normal_count_rot_h = len(wrist_state_h) - outlier_count_rot_h
        outlier_pct_rot_h = outlier_count_rot_h / len(wrist_state_h) * 100
        
        counts_rot_h = [normal_count_rot_h, outlier_count_rot_h]
        bars = ax2.bar(categories, counts_rot_h, color=colors_bar, edgecolor='black', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(wrist_state_h)*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
        ax2.set_title(f'{name} - Wrist Rotation (Head Camera)\n'
                     f'Outlier: {outlier_pct_rot_h:.2f}% | Range: [{lower_bound_h:.4f}, {upper_bound_h:.4f}] rad', 
                     fontsize=12, fontweight='bold', pad=20)
        sns.despine(ax=ax2)
        
        # Row 2: Head Camera - Fingertips Displacement Anomalies
        ax3 = axes[2, idx]
        fingertips_diff_h = np.abs(fingertips_action_h - fingertips_state_h)
        fingertips_diff_3d_h = fingertips_diff_h.reshape(-1, 10, 3)
        fingertips_magnitude_h = np.linalg.norm(fingertips_diff_3d_h, axis=2)
        fingertips_avg_magnitude_h = np.mean(fingertips_magnitude_h, axis=1)
        normal_count_fing_h, outlier_count_fing_h, lb_fing_h, ub_fing_h, outlier_pct_fing_h = detect_outliers_iqr(fingertips_avg_magnitude_h)
        
        counts_fing_h = [normal_count_fing_h, outlier_count_fing_h]
        bars = ax3.bar(categories, counts_fing_h, color=colors_bar, edgecolor='black', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(fingertips_avg_magnitude_h)*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
        ax3.set_title(f'{name} - Fingertips Displacement (Head Camera)\n'
                     f'Outlier: {outlier_pct_fing_h:.2f}% | Range: [{lb_fing_h:.4f}, {ub_fing_h:.4f}]m', 
                     fontsize=12, fontweight='bold', pad=20)
        sns.despine(ax=ax3)
        
        # Row 3: Breast Camera - Wrist Translation Anomalies
        ax4 = axes[3, idx]
        wrist_diff_b = np.abs(wrist_action_b - wrist_state_b)
        translation_diff_b = wrist_diff_b[:, :6]
        translation_magnitude_b = np.linalg.norm(translation_diff_b, axis=1)
        normal_count_trans_b, outlier_count_trans_b, lb_trans_b, ub_trans_b, outlier_pct_trans_b = detect_outliers_iqr(translation_magnitude_b)
        
        counts_b = [normal_count_trans_b, outlier_count_trans_b]
        bars = ax4.bar(categories, counts_b, color=colors_bar, edgecolor='black', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(translation_magnitude_b)*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
        ax4.set_title(f'{name} - Wrist Translation (Breast Camera)\n'
                     f'Outlier: {outlier_pct_trans_b:.2f}% | Range: [{lb_trans_b:.4f}, {ub_trans_b:.4f}]m', 
                     fontsize=12, fontweight='bold', pad=20)
        sns.despine(ax=ax4)
        
        # Row 4: Breast Camera - Wrist Rotation Anomalies
        ax5 = axes[4, idx]
        left_rot6d_state_b = wrist_state_b[:, 6:12]
        right_rot6d_state_b = wrist_state_b[:, 12:18]
        left_rot6d_action_b = wrist_action_b[:, 6:12]
        right_rot6d_action_b = wrist_action_b[:, 12:18]
        
        left_rotation_mag_b = compute_rotation_angle_from_rot6d_diff(left_rot6d_state_b, left_rot6d_action_b)
        right_rotation_mag_b = compute_rotation_angle_from_rot6d_diff(right_rot6d_state_b, right_rot6d_action_b)
        rotation_magnitude_b = np.concatenate([left_rotation_mag_b, right_rotation_mag_b])
        
        Q1_b = np.percentile(rotation_magnitude_b, 25)
        Q3_b = np.percentile(rotation_magnitude_b, 75)
        IQR_b = Q3_b - Q1_b
        lower_bound_b = Q1_b - 1.5 * IQR_b
        upper_bound_b = Q3_b + 1.5 * IQR_b
        outlier_mask_b = (rotation_magnitude_b < lower_bound_b) | (rotation_magnitude_b > upper_bound_b)
        
        outlier_indices_b = np.where(outlier_mask_b)[0]
        outlier_frames_unique_b = np.unique(outlier_indices_b % len(wrist_state_b))
        outlier_count_rot_b = len(outlier_frames_unique_b)
        normal_count_rot_b = len(wrist_state_b) - outlier_count_rot_b
        outlier_pct_rot_b = outlier_count_rot_b / len(wrist_state_b) * 100
        
        counts_rot_b = [normal_count_rot_b, outlier_count_rot_b]
        bars = ax5.bar(categories, counts_rot_b, color=colors_bar, edgecolor='black', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(wrist_state_b)*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax5.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
        ax5.set_title(f'{name} - Wrist Rotation (Breast Camera)\n'
                     f'Outlier: {outlier_pct_rot_b:.2f}% | Range: [{lower_bound_b:.4f}, {upper_bound_b:.4f}] rad', 
                     fontsize=12, fontweight='bold', pad=20)
        sns.despine(ax=ax5)
        
        # Row 5: Breast Camera - Fingertips Displacement Anomalies
        ax6 = axes[5, idx]
        fingertips_diff_b = np.abs(fingertips_action_b - fingertips_state_b)
        fingertips_diff_3d_b = fingertips_diff_b.reshape(-1, 10, 3)
        fingertips_magnitude_b = np.linalg.norm(fingertips_diff_3d_b, axis=2)
        fingertips_avg_magnitude_b = np.mean(fingertips_magnitude_b, axis=1)
        normal_count_fing_b, outlier_count_fing_b, lb_fing_b, ub_fing_b, outlier_pct_fing_b = detect_outliers_iqr(fingertips_avg_magnitude_b)
        
        counts_fing_b = [normal_count_fing_b, outlier_count_fing_b]
        bars = ax6.bar(categories, counts_fing_b, color=colors_bar, edgecolor='black', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(fingertips_avg_magnitude_b)*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax6.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
        ax6.set_title(f'{name} - Fingertips Displacement (Breast Camera)\n'
                     f'Outlier: {outlier_pct_fing_b:.2f}% | Range: [{lb_fing_b:.4f}, {ub_fing_b:.4f}]m', 
                     fontsize=12, fontweight='bold', pad=20)
        sns.despine(ax=ax6)
    
    plt.suptitle('Data Quality Anomaly Detection (IQR Method) - Head & Breast Cameras', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    return fig


def plot_outlier_distribution_analysis(datasets_info: Dict) -> plt.Figure:
    """
    Analyze outlier distribution and detect obvious data errors from threshold-based anomaly detection
    分析异常值分布并通过基于阈值的异常检测发现明显的数据错误
    Provides detailed analysis of outliers including severity, temporal distribution, and error detection
    提供包括严重程度、分布和错误检测在内的详细异常值分析
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    # Use same thresholds as threshold-based anomaly detection
    # 使用与基于阈值的异常检测相同的阈值
    THRESHOLD_WRIST_TRANSLATION = 0.08  # 5cm (手腕平移阈值)
    THRESHOLD_WRIST_ROTATION = 0.5      # ~28.6 degrees (手腕旋转阈值，弧度)
    THRESHOLD_FINGERTIPS_DISPLACEMENT = 0.08  # 5cm (指尖位移阈值)
    
    cameras = [('Head', '_h'), ('Chest', '_b')]
    num_datasets = len(datasets_info)
    num_cameras = len(cameras)
    
    # Adjust figure size based on number of columns (datasets * cameras)
    # 调整图像大小以适应更多列
    fig = plt.figure(figsize=(8 * num_datasets * num_cameras, 12))
    gs = fig.add_gridspec(3, num_datasets * num_cameras, hspace=0.6, wspace=0.3)
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        for cam_idx, (cam_label, cam_suffix) in enumerate(cameras):
            col = idx * num_cameras + cam_idx
            
            wrist_state = info.get(f'wrist_state{cam_suffix}')
            wrist_action = info.get(f'wrist_action{cam_suffix}')
            fingertips_state = info.get(f'fingertips_state{cam_suffix}')
            fingertips_action = info.get(f'fingertips_action{cam_suffix}')
            
            # If specific camera data is missing, try fallback for Chest (which is default) or skip
            if wrist_state is None:
                if cam_suffix == '_b' and 'wrist_state' in info:
                     wrist_state = info['wrist_state']
                     wrist_action = info['wrist_action']
                     fingertips_state = info['fingertips_state']
                     fingertips_action = info['fingertips_action']
                else:
                    # Skip if data not available
                    continue

            # Calculate wrist displacement
            # Note: For translation, we use absolute difference
            # For rotation, we need to compute relative rotation correctly (not simple difference)
            translation_diff = np.abs(wrist_action - wrist_state)[:, :6]
            translation_magnitude = np.linalg.norm(translation_diff, axis=1)
            
            # 1. Wrist Translation Analysis (Top row)
            # 第一行: 手腕平移分析
            ax1 = fig.add_subplot(gs[0, col])
            
            # Detect outliers
            # 检测异常值 (超过阈值)
            outlier_mask_trans = translation_magnitude > THRESHOLD_WRIST_TRANSLATION
            outlier_values_trans = translation_magnitude[outlier_mask_trans]
            
            # Check for extreme errors (more than 5x threshold)
            # 检测极端错误 (超过 5 倍阈值)
            extreme_error_mask_trans = translation_magnitude > (THRESHOLD_WRIST_TRANSLATION * 5)
            extreme_error_count_trans = np.sum(extreme_error_mask_trans)
            extreme_values_trans = translation_magnitude[extreme_error_mask_trans] if extreme_error_count_trans > 0 else np.array([])
            
            # Check for NaN/Inf
            # 检查无效值 (NaN/Inf)
            nan_count_trans = np.sum(np.isnan(translation_magnitude))
            inf_count_trans = np.sum(np.isinf(translation_magnitude))
            
            # Plot histogram - only outliers
            # 绘制直方图 - 仅显示异常值
            if len(outlier_values_trans) > 0:
                max_val = np.nanmax(outlier_values_trans)
                min_val = np.nanmin(outlier_values_trans)
                bins = np.linspace(min_val, max(max_val * 1.1, THRESHOLD_WRIST_TRANSLATION * 2), 50)
                ax1.hist(outlier_values_trans, bins=bins, alpha=0.8, color='#FF6B6B', label='Outliers', edgecolor='black')
            else:
                bins = np.linspace(THRESHOLD_WRIST_TRANSLATION, THRESHOLD_WRIST_TRANSLATION * 2, 50)
            
            ax1.axvline(THRESHOLD_WRIST_TRANSLATION, color='red', linestyle='--', linewidth=2, label=f'Threshold: {THRESHOLD_WRIST_TRANSLATION}m')
            if extreme_error_count_trans > 0:
                # Calculate severity: how many times over the threshold
                # 计算严重程度: 超过阈值的倍数
                severity_trans = extreme_values_trans / THRESHOLD_WRIST_TRANSLATION
                # Distribute y coordinates vertically to avoid overlap
                # 垂直分布 Y 坐标以避免重叠
                # Use sorted values to create a more organized vertical distribution
                sorted_indices = np.argsort(extreme_values_trans)
                y_positions = np.linspace(0.05, 0.95, len(extreme_values_trans))
                y_positions = y_positions[sorted_indices]
                # Get y-axis limits for proper scaling (after histogram is drawn)
                y_min, y_max = ax1.get_ylim()
                if y_max <= 0:
                    y_max = 1
                # Scale y positions to fit within the plot (use top 10% of y-axis range)
                y_scaled = y_min + y_positions * (y_max - y_min) * 0.1
                scatter = ax1.scatter(extreme_values_trans, y_scaled, c=severity_trans, 
                                     cmap='Reds', s=150, marker='x', linewidths=2.5,
                                     label=f'Extreme Errors ({extreme_error_count_trans})', 
                                     zorder=5, vmin=5, vmax=np.max(severity_trans))
                # Add colorbar to show severity
                cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
                cbar.set_label('Severity (× threshold)', fontsize=9, rotation=270, labelpad=15)
            
            ax1.set_xlabel('Wrist Translation Magnitude (m)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax1.set_title(f'{name} ({cam_label}) - Wrist Translation Outlier Distribution\n'
                          f'Outliers: {len(outlier_values_trans)} ({len(outlier_values_trans)/len(translation_magnitude)*100:.2f}%)\n'
                          f'Extreme Errors: {extreme_error_count_trans} | NaN: {nan_count_trans} | Inf: {inf_count_trans}',
                          fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            sns.despine(ax=ax1)
            
            # 2. Wrist Rotation Analysis (Middle row)
            # 第二行: 手腕旋转分析
            ax2 = fig.add_subplot(gs[1, col])
            # Calculate rotation angles correctly: compute relative rotation between state and action
            # Extract rot6d from state and action (not from difference!)
            left_rot6d_state = wrist_state[:, 6:12]   # Left hand rotation (rot6d, 6D) from state
            right_rot6d_state = wrist_state[:, 12:18]  # Right hand rotation (rot6d, 6D) from state
            left_rot6d_action = wrist_action[:, 6:12]   # Left hand rotation (rot6d, 6D) from action
            right_rot6d_action = wrist_action[:, 12:18]  # Right hand rotation (rot6d, 6D) from action
            
            # Compute relative rotation angles correctly
            left_rotation_mag = compute_rotation_angle_from_rot6d_diff(left_rot6d_state, left_rot6d_action)
            right_rotation_mag = compute_rotation_angle_from_rot6d_diff(right_rot6d_state, right_rot6d_action)
            rotation_magnitude = np.concatenate([left_rotation_mag, right_rotation_mag])
            
            # Detect outliers - map back to unique frame indices (same as JSON method)
            # 检测异常值 - 映射回唯一帧索引（与JSON方法一致）
            outlier_mask_rot = rotation_magnitude > THRESHOLD_WRIST_ROTATION
            outlier_indices_rot = np.where(outlier_mask_rot)[0]
            outlier_frames_rot_unique = np.unique(outlier_indices_rot % len(wrist_state))
            outlier_count_rot_unique = len(outlier_frames_rot_unique)
            outlier_values_rot = rotation_magnitude[outlier_mask_rot]
            
            # Check for extreme errors - map to unique frames
            extreme_error_mask_rot = rotation_magnitude > (THRESHOLD_WRIST_ROTATION * 5)
            extreme_indices_rot = np.where(extreme_error_mask_rot)[0]
            extreme_frames_rot_unique = np.unique(extreme_indices_rot % len(wrist_state))
            extreme_error_count_rot = len(extreme_frames_rot_unique)
            extreme_values_rot = rotation_magnitude[extreme_error_mask_rot] if len(extreme_frames_rot_unique) > 0 else np.array([])
            
            # Check for NaN/Inf
            nan_count_rot = np.sum(np.isnan(rotation_magnitude))
            inf_count_rot = np.sum(np.isinf(rotation_magnitude))
            
            # Plot histogram - only outliers
            if len(outlier_values_rot) > 0:
                max_val = np.nanmax(outlier_values_rot)
                min_val = np.nanmin(outlier_values_rot)
                bins = np.linspace(min_val, max(max_val * 1.1, THRESHOLD_WRIST_ROTATION * 2), 50)
                ax2.hist(outlier_values_rot, bins=bins, alpha=0.8, color='#FF6B6B', label='Outliers', edgecolor='black')
            else:
                bins = np.linspace(THRESHOLD_WRIST_ROTATION, THRESHOLD_WRIST_ROTATION * 2, 50)
            
            ax2.axvline(THRESHOLD_WRIST_ROTATION, color='red', linestyle='--', linewidth=2, label=f'Threshold: {THRESHOLD_WRIST_ROTATION}rad')
            if extreme_error_count_rot > 0:
                # Calculate severity: how many times over the threshold
                severity_rot = extreme_values_rot / THRESHOLD_WRIST_ROTATION
                # Distribute y coordinates vertically to avoid overlap
                sorted_indices = np.argsort(extreme_values_rot)
                y_positions = np.linspace(0.05, 0.95, len(extreme_values_rot))
                y_positions = y_positions[sorted_indices]
                # Get y-axis limits for proper scaling (after histogram is drawn)
                y_min, y_max = ax2.get_ylim()
                if y_max <= 0:
                    y_max = 1
                # Scale y positions to fit within the plot (use top 10% of y-axis range)
                y_scaled = y_min + y_positions * (y_max - y_min) * 0.1
                scatter = ax2.scatter(extreme_values_rot, y_scaled, c=severity_rot,
                                     cmap='Reds', s=150, marker='x', linewidths=2.5,
                                     label=f'Extreme Errors ({extreme_error_count_rot})',
                                     zorder=5, vmin=5, vmax=np.max(severity_rot))
                # Add colorbar to show severity
                cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
                cbar.set_label('Severity (× threshold)', fontsize=9, rotation=270, labelpad=15)
            
            ax2.set_xlabel('Wrist Rotation Magnitude (rad)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax2.set_title(f'{name} ({cam_label}) - Wrist Rotation Outlier Distribution\n'
                          f'Outliers: {outlier_count_rot_unique} ({outlier_count_rot_unique/len(wrist_state)*100:.2f}%)\n'
                          f'Extreme Errors: {extreme_error_count_rot} | NaN: {nan_count_rot} | Inf: {inf_count_rot}',
                          fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            sns.despine(ax=ax2)
            
            # 3. Fingertips Displacement Analysis (Bottom row)
            # 第三行: 指尖位移分析
            ax3 = fig.add_subplot(gs[2, col])
            fingertips_diff = np.abs(fingertips_action - fingertips_state)
            fingertips_diff_3d = fingertips_diff.reshape(-1, 10, 3)
            fingertips_magnitude = np.linalg.norm(fingertips_diff_3d, axis=2)
            fingertips_mean_mag = np.mean(fingertips_magnitude, axis=1)
            
            # Detect outliers
            outlier_mask_fing = fingertips_mean_mag > THRESHOLD_FINGERTIPS_DISPLACEMENT
            outlier_values_fing = fingertips_mean_mag[outlier_mask_fing]
            
            # Check for extreme errors
            extreme_error_mask_fing = fingertips_mean_mag > (THRESHOLD_FINGERTIPS_DISPLACEMENT * 5)
            extreme_error_count_fing = np.sum(extreme_error_mask_fing)
            extreme_values_fing = fingertips_mean_mag[extreme_error_mask_fing] if extreme_error_count_fing > 0 else np.array([])
            
            # Check for NaN/Inf
            nan_count_fing = np.sum(np.isnan(fingertips_mean_mag))
            inf_count_fing = np.sum(np.isinf(fingertips_mean_mag))
            
            # Plot histogram - only outliers
            if len(outlier_values_fing) > 0:
                max_val = np.nanmax(outlier_values_fing)
                min_val = np.nanmin(outlier_values_fing)
                bins = np.linspace(min_val, max(max_val * 1.1, THRESHOLD_FINGERTIPS_DISPLACEMENT * 2), 50)
                ax3.hist(outlier_values_fing, bins=bins, alpha=0.8, color='#FF6B6B', label='Outliers', edgecolor='black')
            else:
                bins = np.linspace(THRESHOLD_FINGERTIPS_DISPLACEMENT, THRESHOLD_FINGERTIPS_DISPLACEMENT * 2, 50)
            
            ax3.axvline(THRESHOLD_FINGERTIPS_DISPLACEMENT, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {THRESHOLD_FINGERTIPS_DISPLACEMENT}m')
            if extreme_error_count_fing > 0:
                # Calculate severity: how many times over the threshold
                severity_fing = extreme_values_fing / THRESHOLD_FINGERTIPS_DISPLACEMENT
                # Distribute y coordinates vertically to avoid overlap
                sorted_indices = np.argsort(extreme_values_fing)
                y_positions = np.linspace(0.05, 0.95, len(extreme_values_fing))
                y_positions = y_positions[sorted_indices]
                # Get y-axis limits for proper scaling (after histogram is drawn)
                y_min, y_max = ax3.get_ylim()
                if y_max <= 0:
                    y_max = 1
                # Scale y positions to fit within the plot (use top 10% of y-axis range)
                y_scaled = y_min + y_positions * (y_max - y_min) * 0.1
                scatter = ax3.scatter(extreme_values_fing, y_scaled, c=severity_fing,
                                     cmap='Reds', s=150, marker='x', linewidths=2.5,
                                     label=f'Extreme Errors ({extreme_error_count_fing})',
                                     zorder=5, vmin=5, vmax=np.max(severity_fing))
                # Add colorbar to show severity
                cbar = plt.colorbar(scatter, ax=ax3, pad=0.02)
                cbar.set_label('Severity (× threshold)', fontsize=9, rotation=270, labelpad=15)
            
            ax3.set_xlabel('Fingertips Displacement Magnitude (m)', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax3.set_title(f'{name} ({cam_label}) - Fingertips Displacement Outlier Distribution\n'
                          f'Outliers: {len(outlier_values_fing)} ({len(outlier_values_fing)/len(fingertips_mean_mag)*100:.2f}%)\n'
                          f'Extreme Errors: {extreme_error_count_fing} | NaN: {nan_count_fing} | Inf: {inf_count_fing}',
                          fontsize=11, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            sns.despine(ax=ax3)
    
    plt.suptitle('Outlier Distribution Analysis & Error Detection (Absolute Threshold Method)', 
                fontsize=16, fontweight='bold', y=0.995)
    # Use subplots_adjust instead of tight_layout to avoid warning with colorbars
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.6, wspace=0.3)
    
    return fig


# ============================================================================
# Phase 2: Motion Analysis (Medium Complexity)
# ============================================================================

def plot_wrist_displacement(datasets_info: Dict) -> plt.Figure:
    """
    Analyze wrist displacement statistics for both head and breast cameras
    分析手腕位移统计信息 (头部和胸部摄像机，Translation & Rotation)
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    # 4 rows: Translation Head, Translation Breast, Rotation Head, Rotation Breast
    fig, axes = plt.subplots(4, num_datasets, figsize=(6*num_datasets, 18))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        # 使用头部和胸部两套数据
        wrist_state_h = info['wrist_state_h']  # (frames, 18): left_xyz(3), right_xyz(3), left_rot6d(6), right_rot6d(6)
        wrist_action_h = info['wrist_action_h']
        wrist_state_b = info['wrist_state_b']
        wrist_action_b = info['wrist_action_b']
        
        # 处理头部相机数据
        # Row 0: Head Camera Translation
        wrist_diff_translation_h = wrist_action_h - wrist_state_h
        left_translation_h = wrist_diff_translation_h[:, :3]
        left_translation_mag_h = np.linalg.norm(left_translation_h, axis=1)
        right_translation_h = wrist_diff_translation_h[:, 3:6]
        right_translation_mag_h = np.linalg.norm(right_translation_h, axis=1)
        
        ax1 = axes[0, idx]
        sample_size = min(50000, len(left_translation_mag_h))
        if sample_size < len(left_translation_mag_h):
            sample_indices = np.linspace(0, len(left_translation_mag_h)-1, sample_size, dtype=int)
            left_sample_h = left_translation_mag_h[sample_indices]
            right_sample_h = right_translation_mag_h[sample_indices]
        else:
            left_sample_h = left_translation_mag_h
            right_sample_h = right_translation_mag_h
        
        sns.histplot(left_sample_h, bins=50, ax=ax1, label='Left', alpha=0.6, 
                    color='#FF6B6B', edgecolor='black', kde=True)
        sns.histplot(right_sample_h, bins=50, ax=ax1, label='Right', alpha=0.6, 
                    color='#4ECDC4', edgecolor='black', kde=True)
        
        stats_text = (f'Left: Mean={np.mean(left_translation_mag_h):.4f}, Med={np.median(left_translation_mag_h):.4f}\n'
                     f'Right: Mean={np.mean(right_translation_mag_h):.4f}, Med={np.median(right_translation_mag_h):.4f}')
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.set_xlabel('Translation Magnitude (m)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'{name} - Wrist Translation (Head Camera)', fontsize=12, fontweight='bold')
        ax1.legend()
        sns.despine(ax=ax1)
        
        # Row 1: Breast Camera Translation
        wrist_diff_translation_b = wrist_action_b - wrist_state_b
        left_translation_b = wrist_diff_translation_b[:, :3]
        left_translation_mag_b = np.linalg.norm(left_translation_b, axis=1)
        right_translation_b = wrist_diff_translation_b[:, 3:6]
        right_translation_mag_b = np.linalg.norm(right_translation_b, axis=1)
        
        ax2 = axes[1, idx]
        sample_size_b = min(50000, len(left_translation_mag_b))
        if sample_size_b < len(left_translation_mag_b):
            sample_indices_b = np.linspace(0, len(left_translation_mag_b)-1, sample_size_b, dtype=int)
            left_sample_b = left_translation_mag_b[sample_indices_b]
            right_sample_b = right_translation_mag_b[sample_indices_b]
        else:
            left_sample_b = left_translation_mag_b
            right_sample_b = right_translation_mag_b
        
        sns.histplot(left_sample_b, bins=50, ax=ax2, label='Left', alpha=0.6, 
                    color='#FF6B6B', edgecolor='black', kde=True)
        sns.histplot(right_sample_b, bins=50, ax=ax2, label='Right', alpha=0.6, 
                    color='#4ECDC4', edgecolor='black', kde=True)
        
        stats_text = (f'Left: Mean={np.mean(left_translation_mag_b):.4f}, Med={np.median(left_translation_mag_b):.4f}\n'
                     f'Right: Mean={np.mean(right_translation_mag_b):.4f}, Med={np.median(right_translation_mag_b):.4f}')
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.set_xlabel('Translation Magnitude (m)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'{name} - Wrist Translation (Breast Camera)', fontsize=12, fontweight='bold')
        ax2.legend()
        sns.despine(ax=ax2)
        
        # Row 2: Head Camera Rotation
        left_rot6d_state_h = wrist_state_h[:, 6:12]
        left_rot6d_action_h = wrist_action_h[:, 6:12]
        left_rotation_mag_h = compute_rotation_angle_from_rot6d_diff(left_rot6d_state_h, left_rot6d_action_h)
        right_rot6d_state_h = wrist_state_h[:, 12:18]
        right_rot6d_action_h = wrist_action_h[:, 12:18]
        right_rotation_mag_h = compute_rotation_angle_from_rot6d_diff(right_rot6d_state_h, right_rot6d_action_h)
        
        ax3 = axes[2, idx]
        if sample_size < len(left_rotation_mag_h):
            left_rot_sample_h = left_rotation_mag_h[sample_indices]
            right_rot_sample_h = right_rotation_mag_h[sample_indices]
        else:
            left_rot_sample_h = left_rotation_mag_h
            right_rot_sample_h = right_rotation_mag_h
        
        sns.histplot(left_rot_sample_h, bins=50, ax=ax3, label='Left', alpha=0.6, 
                    color='#FF6B6B', edgecolor='black', kde=True)
        sns.histplot(right_rot_sample_h, bins=50, ax=ax3, label='Right', alpha=0.6, 
                    color='#4ECDC4', edgecolor='black', kde=True)
        
        stats_text = (f'Left: Mean={np.mean(left_rotation_mag_h):.4f}, Med={np.median(left_rotation_mag_h):.4f}\n'
                     f'Right: Mean={np.mean(right_rotation_mag_h):.4f}, Med={np.median(right_rotation_mag_h):.4f}')
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_xlabel('Rotation Magnitude (rad)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title(f'{name} - Wrist Rotation (Head Camera)', fontsize=12, fontweight='bold')
        ax3.legend()
        sns.despine(ax=ax3)
        
        # Row 3: Breast Camera Rotation
        left_rot6d_state_b = wrist_state_b[:, 6:12]
        left_rot6d_action_b = wrist_action_b[:, 6:12]
        left_rotation_mag_b = compute_rotation_angle_from_rot6d_diff(left_rot6d_state_b, left_rot6d_action_b)
        right_rot6d_state_b = wrist_state_b[:, 12:18]
        right_rot6d_action_b = wrist_action_b[:, 12:18]
        right_rotation_mag_b = compute_rotation_angle_from_rot6d_diff(right_rot6d_state_b, right_rot6d_action_b)
        
        ax4 = axes[3, idx]
        if sample_size_b < len(left_rotation_mag_b):
            left_rot_sample_b = left_rotation_mag_b[sample_indices_b]
            right_rot_sample_b = right_rotation_mag_b[sample_indices_b]
        else:
            left_rot_sample_b = left_rotation_mag_b
            right_rot_sample_b = right_rotation_mag_b
        
        sns.histplot(left_rot_sample_b, bins=50, ax=ax4, label='Left', alpha=0.6, 
                    color='#FF6B6B', edgecolor='black', kde=True)
        sns.histplot(right_rot_sample_b, bins=50, ax=ax4, label='Right', alpha=0.6, 
                    color='#4ECDC4', edgecolor='black', kde=True)
        
        stats_text = (f'Left: Mean={np.mean(left_rotation_mag_b):.4f}, Med={np.median(left_rotation_mag_b):.4f}\n'
                     f'Right: Mean={np.mean(right_rotation_mag_b):.4f}, Med={np.median(right_rotation_mag_b):.4f}')
        ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_xlabel('Rotation Magnitude (rad)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title(f'{name} - Wrist Rotation (Breast Camera)', fontsize=12, fontweight='bold')
        ax4.legend()
        sns.despine(ax=ax4)
    
    plt.suptitle('Wrist Displacement Statistics (Head & Breast Cameras)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_fingertips_movement(datasets_info: Dict) -> plt.Figure:
    """
    Analyze fingertips movement range and statistics for both head and breast cameras
    分析指尖移动范围和统计信息 (头部和胸部摄像机)
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    num_datasets = len(datasets_info)
    # 2 rows: Head Camera, Breast Camera
    fig, axes = plt.subplots(2, num_datasets, figsize=(6*num_datasets, 10))
    
    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        # 头部和胸部相机数据
        fingertips_state_h = info['fingertips_state_h']  # (frames, 30): 10 fingertips * 3D
        fingertips_action_h = info['fingertips_action_h']
        fingertips_state_b = info['fingertips_state_b']
        fingertips_action_b = info['fingertips_action_b']
        
        # Row 0: Head Camera
        fingertips_diff_h = fingertips_action_h - fingertips_state_h
        fingertips_diff_3d_h = fingertips_diff_h.reshape(-1, 10, 3)
        fingertips_magnitude_h = np.linalg.norm(fingertips_diff_3d_h, axis=2)
        left_fingertips_h = fingertips_magnitude_h[:, :5]
        left_mean_h = np.mean(left_fingertips_h, axis=1)
        right_fingertips_h = fingertips_magnitude_h[:, 5:]
        right_mean_h = np.mean(right_fingertips_h, axis=1)
        
        ax1 = axes[0, idx]
        sample_size = min(50000, len(left_mean_h))
        if sample_size < len(left_mean_h):
            sample_indices = np.linspace(0, len(left_mean_h)-1, sample_size, dtype=int)
            left_sample_h = left_mean_h[sample_indices]
            right_sample_h = right_mean_h[sample_indices]
        else:
            left_sample_h = left_mean_h
            right_sample_h = right_mean_h
        
        sns.histplot(left_sample_h, bins=50, ax=ax1, label='Left Hand', alpha=0.6, 
                    color='#FF6B6B', edgecolor='black', kde=True)
        sns.histplot(right_sample_h, bins=50, ax=ax1, label='Right Hand', alpha=0.6, 
                    color='#4ECDC4', edgecolor='black', kde=True)
        
        stats_text = (f'Left: Mean={np.mean(left_mean_h):.4f}, Med={np.median(left_mean_h):.4f}\n'
                     f'Right: Mean={np.mean(right_mean_h):.4f}, Med={np.median(right_mean_h):.4f}')
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.set_xlabel('Average Fingertip Movement (m)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'{name} - Fingertips Movement (Head Camera)', fontsize=12, fontweight='bold')
        ax1.legend()
        sns.despine(ax=ax1)
        
        # Row 1: Breast Camera
        fingertips_diff_b = fingertips_action_b - fingertips_state_b
        fingertips_diff_3d_b = fingertips_diff_b.reshape(-1, 10, 3)
        fingertips_magnitude_b = np.linalg.norm(fingertips_diff_3d_b, axis=2)
        left_fingertips_b = fingertips_magnitude_b[:, :5]
        left_mean_b = np.mean(left_fingertips_b, axis=1)
        right_fingertips_b = fingertips_magnitude_b[:, 5:]
        right_mean_b = np.mean(right_fingertips_b, axis=1)
        
        ax2 = axes[1, idx]
        sample_size_b = min(50000, len(left_mean_b))
        if sample_size_b < len(left_mean_b):
            sample_indices_b = np.linspace(0, len(left_mean_b)-1, sample_size_b, dtype=int)
            left_sample_b = left_mean_b[sample_indices_b]
            right_sample_b = right_mean_b[sample_indices_b]
        else:
            left_sample_b = left_mean_b
            right_sample_b = right_mean_b
        
        sns.histplot(left_sample_b, bins=50, ax=ax2, label='Left Hand', alpha=0.6, 
                    color='#FF6B6B', edgecolor='black', kde=True)
        sns.histplot(right_sample_b, bins=50, ax=ax2, label='Right Hand', alpha=0.6, 
                    color='#4ECDC4', edgecolor='black', kde=True)
        
        stats_text = (f'Left: Mean={np.mean(left_mean_b):.4f}, Med={np.median(left_mean_b):.4f}\n'
                     f'Right: Mean={np.mean(right_mean_b):.4f}, Med={np.median(right_mean_b):.4f}')
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.set_xlabel('Average Fingertip Movement (m)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'{name} - Fingertips Movement (Breast Camera)', fontsize=12, fontweight='bold')
        ax2.legend()
        sns.despine(ax=ax2)
    
    plt.suptitle('Fingertips Movement Range Analysis (Head & Breast Cameras)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def detect_problematic_episodes_extrinsic(extrinsic: np.ndarray, episode_ends: np.ndarray, 
                                          episode_names: np.ndarray,
                                          translation_threshold: float = 0.5,
                                          rotation_threshold: float = 1) -> Dict:
    """
    Detect problematic episodes based on maximum extrinsic difference across all frame pairs
    基于所有帧对的最大外参差异检测有问题的 episodes
    
    For each episode, enumerate all frame pairs and find the maximum difference:
    - Find max translation distance across all frame pairs
    - Find max rotation matrix distance across all frame pairs
    - If either exceeds threshold, mark episode as problematic
    
    对每个 episode 枚举所有帧对，找到最大差异：
    - 找到所有帧对中的最大平移距离
    - 找到所有帧对中的最大旋转矩阵距离
    - 如果任一超过阈值，标记 episode 为有问题
    
    Extrinsic format (flattened 4x4 matrix):
    [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]
    
    Args:
        extrinsic: Array of shape (frames, 16) containing flattened 4x4 extrinsic matrices
        episode_ends: Array of cumulative episode end indices
        episode_names: Array of episode names
        translation_threshold: Threshold for translation distance (meters)
        rotation_threshold: Threshold for rotation matrix Frobenius norm distance
        
    Returns:
        Dictionary containing problematic episode information
    """
    print(f"  Detecting problematic episodes based on max extrinsic differences...")
    print(f"    Translation threshold: {translation_threshold}m")
    print(f"    Rotation threshold: {rotation_threshold}")
    
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    num_episodes = len(episode_names)
    
    problematic_episodes = []
    
    for episode_idx in range(num_episodes):
        episode_start = int(episode_starts[episode_idx])
        episode_end = int(episode_ends[episode_idx])
        episode_length = episode_end - episode_start
        episode_name = str(episode_names[episode_idx])
        
        # Skip episodes with less than 2 frames
        if episode_length < 2:
            continue
        
        # Get extrinsic data for this episode
        episode_extrinsic = extrinsic[episode_start:episode_end]
        
        # Extract translation vectors (tx, ty, tz)
        tx = episode_extrinsic[:, 3]   # X translation
        ty = episode_extrinsic[:, 7]   # Y translation
        tz = episode_extrinsic[:, 11]  # Z translation
        translation = np.stack([tx, ty, tz], axis=1)  # (episode_length, 3)
        
        # Extract rotation matrices (3x3)
        # Reshape flattened extrinsic to 4x4 matrices
        extrinsic_4x4 = episode_extrinsic.reshape(-1, 4, 4)
        rotation_matrices = extrinsic_4x4[:, :3, :3]  # (episode_length, 3, 3)
        
        # Find maximum translation difference across all frame pairs
        # Use broadcasting to compute all pairwise distances efficiently
        # translation shape: (episode_length, 3)
        # translation[:, None, :] - translation[None, :, :] gives (episode_length, episode_length, 3)
        translation_diffs = np.linalg.norm(
            translation[:, None, :] - translation[None, :, :], 
            axis=2
        )  # (episode_length, episode_length)
        
        # Find maximum translation difference and its frame pair
        max_translation_diff = np.max(translation_diffs)
        max_trans_idx = np.unravel_index(np.argmax(translation_diffs), translation_diffs.shape)
        max_trans_frame1, max_trans_frame2 = max_trans_idx
        
        # Find maximum rotation difference across all frame pairs
        # Use broadcasting to compute all pairwise Frobenius norms efficiently
        # rotation_matrices shape: (episode_length, 3, 3)
        # Compute all pairwise differences: (episode_length, episode_length, 3, 3)
        rotation_diffs_matrices = rotation_matrices[:, None, :, :] - rotation_matrices[None, :, :, :]
        
        # Compute Frobenius norm for each pair
        # Frobenius norm = sqrt(sum of squares of all elements)
        # Reshape to (episode_length, episode_length, 9) and compute norm along last axis
        rotation_diffs = np.linalg.norm(
            rotation_diffs_matrices.reshape(episode_length, episode_length, -1), 
            axis=2
        )  # (episode_length, episode_length)
        
        # Find maximum rotation difference and its frame pair
        max_rotation_diff = np.max(rotation_diffs)
        max_rot_idx = np.unravel_index(np.argmax(rotation_diffs), rotation_diffs.shape)
        max_rot_frame1, max_rot_frame2 = max_rot_idx
        
        # Check if episode is problematic (either threshold exceeded)
        if max_translation_diff > translation_threshold or max_rotation_diff > rotation_threshold:
            problematic_episodes.append({
                'episode_idx': int(episode_idx),
                'episode_name': episode_name,
                'episode_start': int(episode_start),
                'episode_end': int(episode_end),
                'episode_length': int(episode_length),
                'max_translation_diff': float(max_translation_diff),
                'max_translation_frame_pair': {
                    'frame1_in_episode': int(max_trans_frame1),
                    'frame2_in_episode': int(max_trans_frame2),
                    'frame1_global': int(episode_start + max_trans_frame1),
                    'frame2_global': int(episode_start + max_trans_frame2)
                },
                'max_rotation_diff': float(max_rotation_diff),
                'max_rotation_frame_pair': {
                    'frame1_in_episode': int(max_rot_frame1),
                    'frame2_in_episode': int(max_rot_frame2),
                    'frame1_global': int(episode_start + max_rot_frame1),
                    'frame2_global': int(episode_start + max_rot_frame2)
                },
                'translation_exceeds': bool(max_translation_diff > translation_threshold),
                'rotation_exceeds': bool(max_rotation_diff > rotation_threshold)
            })
    
    return {
        'translation_threshold': float(translation_threshold),
        'rotation_threshold': float(rotation_threshold),
        'total_episodes': int(num_episodes),
        'num_problematic_episodes': len(problematic_episodes),
        'problematic_episodes': problematic_episodes
    }


def plot_stationary_frame_detection(datasets_info: Dict) -> plt.Figure:
    """
    Detect stationary frames (frames with minimal change between adjacent frames) for both STATE and ACTION
    检测 STATE 和 ACTION 中的静止帧（相邻帧之间变化极小的帧）
    Requires 150 consecutive frames to satisfy thresholds to be counted as stationary
    需要连续150帧都满足阈值才能被统计为停顿帧
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        matplotlib Figure object
    """
    # Thresholds for stationary frame detection (very small values)
    # 静止帧检测的阈值（非常小的值）
    STATIONARY_THRESHOLD_WRIST_TRANSLATION = 0.001  # 1mm (手腕平移阈值)
    STATIONARY_THRESHOLD_WRIST_ROTATION = 0.01      # ~0.57 degrees (手腕旋转阈值)
    STATIONARY_THRESHOLD_FINGERTIPS = 0.001         # 1mm (指尖位移阈值)
    
    num_datasets = len(datasets_info)
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, num_datasets, hspace=0.5, wspace=0.3)
    
    for idx, (name, info) in enumerate(datasets_info.items()):
        wrist_state = info['wrist_state']
        wrist_action = info['wrist_action']
        fingertips_state = info['fingertips_state']
        fingertips_action = info['fingertips_action']
        episode_ends = info['episode_ends']
        
        # Create mask to exclude episode boundaries
        # 创建掩码以排除 Episode 边界
        num_diffs = len(wrist_state) - 1
        valid_mask = np.ones(num_diffs, dtype=bool)
        boundary_indices = episode_ends[:-1] - 1
        valid_mask[boundary_indices] = False
        
        # Helper function to detect stationary frames
        def detect_stationary_frames(wrist_data, fingertips_data, valid_mask, data_type_label):
            """
            Internal helper to detect stationary frames based on thresholds
            内部辅助函数: 基于阈值检测静止帧
            """
            # Calculate frame-to-frame differences
            wrist_diff = np.diff(wrist_data, axis=0)
            fingertips_diff = np.diff(fingertips_data, axis=0)
            
            # Wrist translation magnitude
            wrist_translation = wrist_diff[:, :6]
            wrist_translation_mag = np.linalg.norm(wrist_translation, axis=1)
            
            # Wrist rotation magnitude - use proper rotation angle calculation
            # Calculate frame-to-frame rotation angles correctly using rot6d representation
            left_rot6d = wrist_data[:, 6:12]   # Left hand rotation (rot6d, 6D)
            right_rot6d = wrist_data[:, 12:18]  # Right hand rotation (rot6d, 6D)
            left_rotation_mag = compute_rotation_angle_from_rot6d_diff(left_rot6d[:-1], left_rot6d[1:])
            right_rotation_mag = compute_rotation_angle_from_rot6d_diff(right_rot6d[:-1], right_rot6d[1:])
            
            # For stationary detection, use maximum of left and right rotation
            # 对于静止帧检测，使用左右手旋转的最大值
            wrist_rotation_mag = np.maximum(left_rotation_mag, right_rotation_mag)
            
            # Fingertips magnitude: reshape to (frames-1, 10, 3) and calculate magnitude for each fingertip
            # Then take mean across all fingertips
            fingertips_diff_3d = fingertips_diff.reshape(-1, 10, 3)
            fingertips_magnitude = np.linalg.norm(fingertips_diff_3d, axis=2)  # (frames-1, 10)
            fingertips_mag = np.mean(fingertips_magnitude, axis=1)  # (frames-1,)
            
            # Detect stationary frames: all three metrics below thresholds
            # 检测静止帧: 所有三个指标都必须低于各自的阈值
            # Apply valid_mask to all arrays before comparison
            wrist_translation_mag_valid = wrist_translation_mag[valid_mask]
            wrist_rotation_mag_valid = wrist_rotation_mag[valid_mask]
            fingertips_mag_valid = fingertips_mag[valid_mask]
            
            # First, find frames that satisfy all three thresholds individually
            # 首先，找到满足所有三个阈值的帧（单帧判断）
            single_frame_mask = (
                (wrist_translation_mag_valid < STATIONARY_THRESHOLD_WRIST_TRANSLATION) &
                (wrist_rotation_mag_valid < STATIONARY_THRESHOLD_WRIST_ROTATION) &
                (fingertips_mag_valid < STATIONARY_THRESHOLD_FINGERTIPS)
            )
            
            # Require 150 consecutive frames to be stationary
            # 要求连续150帧都满足条件才能被统计为停顿帧
            MIN_CONSECUTIVE_FRAMES = 150
            num_frames = len(single_frame_mask)
            stationary_mask = np.zeros(num_frames, dtype=bool)
            
            # Use sliding window to find consecutive sequences of at least 150 frames
            # 使用滑动窗口找到至少连续150帧的序列
            for i in range(num_frames - MIN_CONSECUTIVE_FRAMES + 1):
                # Check if next 150 frames all satisfy the condition
                # 检查接下来150帧是否都满足条件
                if np.all(single_frame_mask[i:i+MIN_CONSECUTIVE_FRAMES]):
                    # Mark all 150 frames as stationary
                    # 将这150帧都标记为停顿帧
                    stationary_mask[i:i+MIN_CONSECUTIVE_FRAMES] = True
            
            stationary_count = np.sum(stationary_mask)
            total_frames = len(wrist_translation_mag_valid)  # Count only valid frames (excluding episode boundaries)
            stationary_ratio = stationary_count / total_frames if total_frames > 0 else 0
            
            # Find consecutive stationary frame sequences (for statistics)
            # 寻找连续的静止帧序列（用于统计）
            stationary_indices = np.where(stationary_mask)[0]
            if len(stationary_indices) == 0:
                consecutive_lengths = np.array([])
            else:
                # Find consecutive sequences
                consecutive_lengths = []
                current_length = 1
                for i in range(1, len(stationary_indices)):
                    if stationary_indices[i] == stationary_indices[i-1] + 1:
                        current_length += 1
                    else:
                        if current_length >= MIN_CONSECUTIVE_FRAMES:  # Only count sequences >= 150 frames
                            consecutive_lengths.append(current_length)
                        current_length = 1
                if current_length >= MIN_CONSECUTIVE_FRAMES:  # Only count sequences >= 150 frames
                    consecutive_lengths.append(current_length)
                consecutive_lengths = np.array(consecutive_lengths)
            
            return {
                'stationary_mask': stationary_mask,
                'stationary_count': stationary_count,
                'total_frames': total_frames,
                'stationary_ratio': stationary_ratio,
                'consecutive_lengths': consecutive_lengths,
                'wrist_translation_mag': wrist_translation_mag_valid,
                'wrist_rotation_mag': wrist_rotation_mag_valid,
                'fingertips_mag': fingertips_mag_valid
            }
        
        # Detect stationary frames for STATE
        state_stats = detect_stationary_frames(wrist_state, fingertips_state, valid_mask, 'State')
        
        # Detect stationary frames for ACTION
        action_stats = detect_stationary_frames(wrist_action, fingertips_action, valid_mask, 'Action')
        
        # Row 0: Stationary Frame Ratio Comparison (State vs Action)
        # 第一行: 静止帧比例对比 (State vs Action)
        ax0 = fig.add_subplot(gs[0, idx])
        categories = ['State', 'Action']
        ratios = [state_stats['stationary_ratio'] * 100, action_stats['stationary_ratio'] * 100]
        colors = ['#4CAF50', '#2196F3']
        bars = ax0.bar(categories, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax0.set_ylabel('Stationary Frame Ratio (%)', fontsize=11, fontweight='bold')
        ax0.set_title(f'{name} - Stationary Frame Ratio Comparison', fontsize=12, fontweight='bold')
        ax0.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}%\n({state_stats["stationary_count"] if bar == bars[0] else action_stats["stationary_count"]} frames)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        sns.despine(ax=ax0)
        
        # Row 1: Consecutive Stationary Frame Length Distribution (State)
        # 第二行: 连续静止帧长度分布 (State)
        ax1 = fig.add_subplot(gs[1, idx])
        if len(state_stats['consecutive_lengths']) > 0:
            max_length = np.max(state_stats['consecutive_lengths'])
            bins = np.arange(1, max_length + 2) - 0.5
            ax1.hist(state_stats['consecutive_lengths'], bins=bins, alpha=0.7, color='#4CAF50', 
                    edgecolor='black', linewidth=1.5)
            ax1.axvline(np.mean(state_stats['consecutive_lengths']), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(state_stats["consecutive_lengths"]):.2f}')
            ax1.axvline(np.median(state_stats['consecutive_lengths']), color='blue', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(state_stats["consecutive_lengths"]):.2f}')
        else:
            ax1.text(0.5, 0.5, 'No consecutive stationary frames', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_xlabel('Consecutive Stationary Frame Length', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title(f'{name} - Consecutive Stationary Frame Length Distribution (State)', 
                     fontsize=12, fontweight='bold')
        if len(state_stats['consecutive_lengths']) > 0:
            ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        sns.despine(ax=ax1)
        
        # Row 2: Consecutive Stationary Frame Length Distribution (Action)
        # 第三行: 连续静止帧长度分布 (Action)
        ax2 = fig.add_subplot(gs[2, idx])
        if len(action_stats['consecutive_lengths']) > 0:
            max_length = np.max(action_stats['consecutive_lengths'])
            bins = np.arange(1, max_length + 2) - 0.5
            ax2.hist(action_stats['consecutive_lengths'], bins=bins, alpha=0.7, color='#2196F3', 
                    edgecolor='black', linewidth=1.5)
            ax2.axvline(np.mean(action_stats['consecutive_lengths']), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(action_stats["consecutive_lengths"]):.2f}')
            ax2.axvline(np.median(action_stats['consecutive_lengths']), color='blue', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(action_stats["consecutive_lengths"]):.2f}')
        else:
            ax2.text(0.5, 0.5, 'No consecutive stationary frames', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlabel('Consecutive Stationary Frame Length', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(f'{name} - Consecutive Stationary Frame Length Distribution (Action)', 
                     fontsize=12, fontweight='bold')
        if len(action_stats['consecutive_lengths']) > 0:
            ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        sns.despine(ax=ax2)
        
        # Row 3: Stationary Frame Statistics Table
        # 第四行: 静止帧统计数据表
        ax3 = fig.add_subplot(gs[3, idx])
        ax3.axis('off')
        
        # Create statistics table
        stats_data = {
            'Metric': [
                'Total Valid Frames',
                'Stationary Frames (State)',
                'Stationary Ratio (State)',
                'Max Consecutive (State)',
                'Mean Consecutive (State)',
                'Stationary Frames (Action)',
                'Stationary Ratio (Action)',
                'Max Consecutive (Action)',
                'Mean Consecutive (Action)'
            ],
            'Value': [
                f'{state_stats["total_frames"]}',
                f'{state_stats["stationary_count"]}',
                f'{state_stats["stationary_ratio"]*100:.2f}%',
                f'{np.max(state_stats["consecutive_lengths"]) if len(state_stats["consecutive_lengths"]) > 0 else 0}',
                f'{np.mean(state_stats["consecutive_lengths"]):.2f}' if len(state_stats["consecutive_lengths"]) > 0 else '0.00',
                f'{action_stats["stationary_count"]}',
                f'{action_stats["stationary_ratio"]*100:.2f}%',
                f'{np.max(action_stats["consecutive_lengths"]) if len(action_stats["consecutive_lengths"]) > 0 else 0}',
                f'{np.mean(action_stats["consecutive_lengths"]):.2f}' if len(action_stats["consecutive_lengths"]) > 0 else '0.00'
            ]
        }
        
        table = ax3.table(cellText=[[stats_data['Metric'][i], stats_data['Value'][i]] 
                                     for i in range(len(stats_data['Metric']))],
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(stats_data['Metric']) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#F2F2F2')
                    else:
                        cell.set_facecolor('white')
        
        ax3.set_title(f'{name} - Stationary Frame Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Stationary Frame Detection (State & Action)', fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.5, wspace=0.3)
    
    return fig


def collect_anomaly_data(datasets_info: Dict) -> Dict:
    """
    Collect all anomaly data points with frame indices and episode names
    收集所有异常数据点的帧索引和 episode 名称
    Only includes state-action difference anomalies and stationary frame detection
    仅包含state-action差异异常和停顿帧检测
    
    Args:
        datasets_info: Dictionary containing episode data for each dataset
        
    Returns:
        Dictionary containing anomaly information for all datasets
        包含所有数据集异常信息的字典

    注：经验证头部和胸部同分布，硬编码使用头部相机，视野更大便于.mp4可视化分析
    """
    # Thresholds (same as in visualization functions)
    # 阈值设置（与可视化函数保持一致）
    THRESHOLD_WRIST_TRANSLATION = 0.08  # 5cm
    THRESHOLD_WRIST_ROTATION = 0.5      # ~28.6 degrees
    THRESHOLD_FINGERTIPS_DISPLACEMENT = 0.08  # 5cm
    STATIONARY_THRESHOLD_WRIST_TRANSLATION = 0.001  # 1mm
    STATIONARY_THRESHOLD_WRIST_ROTATION = 0.01      # ~0.57 degrees
    STATIONARY_THRESHOLD_FINGERTIPS = 0.001         # 1mm
    
    anomaly_report = {}
    
    for name, info in datasets_info.items():
        print(f"\nCollecting anomaly data for {name}...")
        
        wrist_state = info['wrist_state_h']
        wrist_action = info['wrist_action_h']
        fingertips_state = info['fingertips_state_h']
        fingertips_action = info['fingertips_action_h']
        episode_ends = info['episode_ends']
        episode_names = info['episode_names']
        
        dataset_anomalies = {
            'dataset_name': name,
            'total_frames': len(wrist_state),
            'total_episodes': len(episode_names),
            'anomalies': {}
        }
        
        # ===== 1. Absolute Threshold Anomalies (State-Action Difference) =====
        # 绝对阈值异常检测（State-Action差异）
        print(f"  - Detecting absolute threshold anomalies (State-Action difference)...")
        
        # Calculate state-action differences for threshold-based detection
        translation_diff = np.abs(wrist_action - wrist_state)[:, :6]
        translation_magnitude = np.linalg.norm(translation_diff, axis=1)
        
        left_rot6d_state = wrist_state[:, 6:12]
        right_rot6d_state = wrist_state[:, 12:18]
        left_rot6d_action = wrist_action[:, 6:12]
        right_rot6d_action = wrist_action[:, 12:18]
        left_rotation_mag = compute_rotation_angle_from_rot6d_diff(left_rot6d_state, left_rot6d_action)
        right_rotation_mag = compute_rotation_angle_from_rot6d_diff(right_rot6d_state, right_rot6d_action)
        rotation_magnitude = np.concatenate([left_rotation_mag, right_rotation_mag])
        
        fingertips_diff = np.abs(fingertips_action - fingertips_state)
        fingertips_diff_3d = fingertips_diff.reshape(-1, 10, 3)
        fingertips_magnitude = np.linalg.norm(fingertips_diff_3d, axis=2)
        fingertips_mean_mag = np.mean(fingertips_magnitude, axis=1)
        
        # Extreme errors (5x threshold)
        extreme_mask_trans = translation_magnitude > (THRESHOLD_WRIST_TRANSLATION * 5)
        extreme_frames_trans = np.where(extreme_mask_trans)[0]
        
        extreme_mask_rot = rotation_magnitude > (THRESHOLD_WRIST_ROTATION * 5)
        extreme_indices_rot = np.where(extreme_mask_rot)[0]
        extreme_frames_rot = np.unique(extreme_indices_rot % len(wrist_state))
        
        extreme_mask_fing = fingertips_mean_mag > (THRESHOLD_FINGERTIPS_DISPLACEMENT * 5)
        extreme_frames_fing = np.where(extreme_mask_fing)[0]
        
        # All outliers (above threshold)
        outlier_mask_abs_trans = translation_magnitude > THRESHOLD_WRIST_TRANSLATION
        outlier_frames_abs_trans = np.where(outlier_mask_abs_trans)[0]
        
        outlier_mask_abs_rot = rotation_magnitude > THRESHOLD_WRIST_ROTATION
        outlier_indices_abs_rot = np.where(outlier_mask_abs_rot)[0]
        outlier_frames_abs_rot = np.unique(outlier_indices_abs_rot % len(wrist_state))
        
        outlier_mask_abs_fing = fingertips_mean_mag > THRESHOLD_FINGERTIPS_DISPLACEMENT
        outlier_frames_abs_fing = np.where(outlier_mask_abs_fing)[0]

        # ===== 新增代码：计算每个episode的最大偏移量 =====
        # Calculate max exceed value for each episode with offset frames
        
        # Function to calculate max exceed per episode
        def calculate_episode_max_exceed(anomaly_frames, all_exceed_values, episode_ends, episode_names):
            episode_dict = {}
            current_episode_idx = 0
            episode_start = 0
            
            for ep_idx, ep_end in enumerate(episode_ends):
                ep_name = episode_names[ep_idx]
                
                # Get anomaly frames within this episode
                ep_anomaly_mask = (anomaly_frames >= episode_start) & (anomaly_frames < ep_end)
                ep_anomaly_frames = anomaly_frames[ep_anomaly_mask]
                
                if len(ep_anomaly_frames) > 0:
                    # Get exceed values for these frames using the global exceed array
                    ep_exceed_values = all_exceed_values[ep_anomaly_frames]
                    max_exceed = np.max(ep_exceed_values) if len(ep_exceed_values) > 0 else 0
                    
                    episode_dict[ep_name] = max_exceed
                
                episode_start = ep_end
            
            return episode_dict
        
        # Calculate exceed values for ALL frames (not just anomaly frames)
        # For translation: 计算所有帧的偏移值，超过阈值则为正，否则为0
        translation_exceed_all = np.maximum(translation_magnitude - THRESHOLD_WRIST_TRANSLATION, 0)
        translation_episode_max_exceed = calculate_episode_max_exceed(
            outlier_frames_abs_trans, translation_exceed_all, episode_ends, episode_names
        )
        
        # For rotation: 计算每帧的最大旋转偏移
        rotation_exceed_per_frame = np.zeros(len(wrist_state))
        for i in range(len(wrist_state)):
            left_val = left_rotation_mag[i]
            right_val = right_rotation_mag[i]
            max_rot = max(left_val, right_val)
            rotation_exceed_per_frame[i] = max(max_rot - THRESHOLD_WRIST_ROTATION, 0)
        
        rotation_episode_max_exceed = calculate_episode_max_exceed(
            outlier_frames_abs_rot, rotation_exceed_per_frame, episode_ends, episode_names
        )
        
        # For fingertips: 计算所有帧的指尖偏移值
        fingertips_exceed_all = np.maximum(fingertips_mean_mag - THRESHOLD_FINGERTIPS_DISPLACEMENT, 0)
        fingertips_episode_max_exceed = calculate_episode_max_exceed(
            outlier_frames_abs_fing, fingertips_exceed_all, episode_ends, episode_names
        )
        
        # Convert to list format for JSON output
        def convert_to_episodes_list(episode_dict):
            return [{"episode_name": ep_name, "exceed": float(exceed)} 
                   for ep_name, exceed in episode_dict.items()]
        
        translation_episodes_list = convert_to_episodes_list(translation_episode_max_exceed)
        rotation_episodes_list = convert_to_episodes_list(rotation_episode_max_exceed)
        fingertips_episodes_list = convert_to_episodes_list(fingertips_episode_max_exceed)
        # ===== 新增代码结束 =====
        
        # Normal threshold outliers (above threshold)
        # 普通阈值异常（超过阈值）
        dataset_anomalies['anomalies']['diff_state_action_wrist_translation'] = {
            'description': 'State-Action difference wrist translation outliers (above threshold)',
            'threshold': f'{THRESHOLD_WRIST_TRANSLATION:.4f}m',
            'count': len(outlier_frames_abs_trans),
            'frames': get_episode_info_for_frames(outlier_frames_abs_trans, episode_ends, episode_names),
            'episodes': translation_episodes_list  # 新增的episodes标签
        }
        
        dataset_anomalies['anomalies']['diff_state_action_wrist_rotation'] = {
            'description': 'State-Action difference wrist rotation outliers (above threshold)',
            'threshold': f'{THRESHOLD_WRIST_ROTATION:.4f}rad',
            'count': len(outlier_frames_abs_rot),
            'frames': get_episode_info_for_frames(outlier_frames_abs_rot, episode_ends, episode_names),
            'episodes': rotation_episodes_list  # 新增的episodes标签
        }
        
        dataset_anomalies['anomalies']['diff_state_action_fingertips'] = {
            'description': 'State-Action difference fingertips displacement outliers (above threshold)',
            'threshold': f'{THRESHOLD_FINGERTIPS_DISPLACEMENT:.4f}m',
            'count': len(outlier_frames_abs_fing),
            'frames': get_episode_info_for_frames(outlier_frames_abs_fing, episode_ends, episode_names),
            'episodes': fingertips_episodes_list  # 新增的episodes标签
        }
        
        # # Normal threshold outliers (above threshold)
        # # 普通阈值异常（超过阈值）
        # dataset_anomalies['anomalies']['diff_state_action_wrist_translation'] = {
        #     'description': 'State-Action difference wrist translation outliers (above threshold)',
        #     'threshold': f'{THRESHOLD_WRIST_TRANSLATION:.4f}m',
        #     'count': len(outlier_frames_abs_trans),
        #     'frames': get_episode_info_for_frames(outlier_frames_abs_trans, episode_ends, episode_names)
        # }
        
        # dataset_anomalies['anomalies']['diff_state_action_wrist_rotation'] = {
        #     'description': 'State-Action difference wrist rotation outliers (above threshold)',
        #     'threshold': f'{THRESHOLD_WRIST_ROTATION:.4f}rad',
        #     'count': len(outlier_frames_abs_rot),
        #     'frames': get_episode_info_for_frames(outlier_frames_abs_rot, episode_ends, episode_names)
        # }
        
        # dataset_anomalies['anomalies']['diff_state_action_fingertips'] = {
        #     'description': 'State-Action difference fingertips displacement outliers (above threshold)',
        #     'threshold': f'{THRESHOLD_FINGERTIPS_DISPLACEMENT:.4f}m',
        #     'count': len(outlier_frames_abs_fing),
        #     'frames': get_episode_info_for_frames(outlier_frames_abs_fing, episode_ends, episode_names)
        # }
        
        # Extreme errors (5x threshold)
        # 极端错误（5倍阈值）
        dataset_anomalies['anomalies']['diff_state_action_extreme_wrist_translation'] = {
            'description': 'State-Action difference extreme wrist translation errors (5x threshold)',
            'threshold': f'{THRESHOLD_WRIST_TRANSLATION * 5:.4f}m',
            'count': len(extreme_frames_trans),
            'frames': get_episode_info_for_frames(extreme_frames_trans, episode_ends, episode_names)
        }
        
        dataset_anomalies['anomalies']['diff_state_action_extreme_wrist_rotation'] = {
            'description': 'State-Action difference extreme wrist rotation errors (5x threshold)',
            'threshold': f'{THRESHOLD_WRIST_ROTATION * 5:.4f}rad',
            'count': len(extreme_frames_rot),
            'frames': get_episode_info_for_frames(extreme_frames_rot, episode_ends, episode_names)
        }
        
        dataset_anomalies['anomalies']['diff_state_action_extreme_fingertips'] = {
            'description': 'State-Action difference extreme fingertips displacement errors (5x threshold)',
            'threshold': f'{THRESHOLD_FINGERTIPS_DISPLACEMENT * 5:.4f}m',
            'count': len(extreme_frames_fing),
            'frames': get_episode_info_for_frames(extreme_frames_fing, episode_ends, episode_names)
        }
        
        # ===== 2. Stationary Frames Detection =====
        # 停顿帧检测（需要连续150帧都满足条件）
        print(f"  - Detecting stationary frames (requires 150 consecutive frames)...")
        
        MIN_CONSECUTIVE_FRAMES = 150  # Require 150 consecutive frames
        
        # Create valid mask (exclude episode boundaries)
        num_diffs = len(wrist_state) - 1
        valid_mask = np.ones(num_diffs, dtype=bool)
        boundary_indices = episode_ends[:-1] - 1
        valid_mask[boundary_indices] = False
        
        # Calculate frame-to-frame differences for stationary detection
        wrist_diff_state = np.diff(wrist_state, axis=0)
        wrist_diff_action = np.diff(wrist_action, axis=0)
        fingertips_diff_state = np.diff(fingertips_state, axis=0)
        fingertips_diff_action = np.diff(fingertips_action, axis=0)
        
        # Wrist translation magnitude
        wrist_translation_state = wrist_diff_state[:, :6]
        wrist_translation_mag_state = np.linalg.norm(wrist_translation_state, axis=1)
        wrist_translation_action = wrist_diff_action[:, :6]
        wrist_translation_mag_action = np.linalg.norm(wrist_translation_action, axis=1)
        
        # Wrist rotation magnitude
        left_rot6d_state = wrist_state[:, 6:12]
        right_rot6d_state = wrist_state[:, 12:18]
        left_rotation_mag_state = compute_rotation_angle_from_rot6d_diff(left_rot6d_state[:-1], left_rot6d_state[1:])
        right_rotation_mag_state = compute_rotation_angle_from_rot6d_diff(right_rot6d_state[:-1], right_rot6d_state[1:])
        
        left_rot6d_action = wrist_action[:, 6:12]
        right_rot6d_action = wrist_action[:, 12:18]
        left_rotation_mag_action = compute_rotation_angle_from_rot6d_diff(left_rot6d_action[:-1], left_rot6d_action[1:])
        right_rotation_mag_action = compute_rotation_angle_from_rot6d_diff(right_rot6d_action[:-1], right_rot6d_action[1:])
        
        # Fingertips magnitude
        fingertips_diff_3d_state = fingertips_diff_state.reshape(-1, 10, 3)
        fingertips_magnitude_state = np.linalg.norm(fingertips_diff_3d_state, axis=2)
        fingertips_mag_state = np.mean(fingertips_magnitude_state, axis=1)
        
        fingertips_diff_3d_action = fingertips_diff_action.reshape(-1, 10, 3)
        fingertips_magnitude_action = np.linalg.norm(fingertips_diff_3d_action, axis=2)
        fingertips_mag_action = np.mean(fingertips_magnitude_action, axis=1)
        
        # Helper function to detect stationary frames with consecutive requirement
        # 辅助函数：检测需要连续150帧的停顿帧
        def detect_stationary_with_consecutive(translation_mag, rotation_left, rotation_right, 
                                               fingertips_mag, valid_mask):
            """
            Detect stationary frames requiring 150 consecutive frames
            检测需要连续150帧的停顿帧
            """
            # First, find frames that satisfy all thresholds individually
            # 首先，找到满足所有阈值的帧
            single_frame_mask = (
                (translation_mag < STATIONARY_THRESHOLD_WRIST_TRANSLATION) &
                ((rotation_left < STATIONARY_THRESHOLD_WRIST_ROTATION) & 
                 (rotation_right < STATIONARY_THRESHOLD_WRIST_ROTATION)) &
                (fingertips_mag < STATIONARY_THRESHOLD_FINGERTIPS)
            )
            
            # Apply valid_mask to exclude episode boundaries
            # 应用valid_mask排除episode边界
            single_frame_mask_valid = single_frame_mask & valid_mask
            
            # Find indices that satisfy condition (in original diff array)
            # 找到满足条件的索引（在原始diff数组中）
            candidate_indices = np.where(single_frame_mask_valid)[0]
            
            # Use sliding window to find consecutive sequences of at least 150 frames
            # 使用滑动窗口找到至少连续150帧的序列
            stationary_mask_result = np.zeros(len(single_frame_mask), dtype=bool)
            
            if len(candidate_indices) >= MIN_CONSECUTIVE_FRAMES:
                # Find consecutive sequences
                # 查找连续序列
                i = 0
                while i < len(candidate_indices) - MIN_CONSECUTIVE_FRAMES + 1:
                    # Check if next 150 frames are consecutive
                    # 检查接下来150帧是否连续
                    if np.all(np.diff(candidate_indices[i:i+MIN_CONSECUTIVE_FRAMES]) == 1):
                        # Mark all frames in this sequence as stationary
                        # 将这个序列中的所有帧都标记为停顿帧
                        start_idx = candidate_indices[i]
                        end_idx = candidate_indices[i + MIN_CONSECUTIVE_FRAMES - 1]
                        # Mark all frames from start to end (inclusive)
                        # 标记从start到end的所有帧（包含）
                        stationary_mask_result[start_idx:end_idx+1] = True
                        # Skip to after this sequence
                        # 跳过这个序列
                        i += MIN_CONSECUTIVE_FRAMES
                    else:
                        i += 1
            
            # Convert diff indices to original frame indices
            # diff索引i对应原始帧i+1（因为diff[i] = frame[i+1] - frame[i]）
            # 将diff索引转换为原始帧索引
            stationary_diff_indices = np.where(stationary_mask_result)[0]
            # Convert to original frame indices: diff index i -> frame index i+1
            # 转换为原始帧索引：diff索引i -> 帧索引i+1
            stationary_frame_indices = stationary_diff_indices + 1
            return stationary_frame_indices
        
        # State stationary frames
        # 检测State停顿帧
        stationary_frames_state = detect_stationary_with_consecutive(
            wrist_translation_mag_state,
            left_rotation_mag_state,
            right_rotation_mag_state,
            fingertips_mag_state,
            valid_mask
        )
        
        # Action stationary frames
        # 检测Action停顿帧
        stationary_frames_action = detect_stationary_with_consecutive(
            wrist_translation_mag_action,
            left_rotation_mag_action,
            right_rotation_mag_action,
            fingertips_mag_action,
            valid_mask
        )
        
        dataset_anomalies['anomalies']['stationary_frames_state'] = {
            'description': 'Stationary frames with minimal change (State, requires 150 consecutive frames)',
            'threshold': f'Translation<{STATIONARY_THRESHOLD_WRIST_TRANSLATION}m, Rotation<{STATIONARY_THRESHOLD_WRIST_ROTATION}rad, Fingertips<{STATIONARY_THRESHOLD_FINGERTIPS}m, Consecutive={MIN_CONSECUTIVE_FRAMES}',
            'count': len(stationary_frames_state),
            'frames': get_episode_info_for_frames(stationary_frames_state, episode_ends, episode_names)
        }
        
        dataset_anomalies['anomalies']['stationary_frames_action'] = {
            'description': 'Stationary frames with minimal change (Action, requires 150 consecutive frames)',
            'threshold': f'Translation<{STATIONARY_THRESHOLD_WRIST_TRANSLATION}m, Rotation<{STATIONARY_THRESHOLD_WRIST_ROTATION}rad, Fingertips<{STATIONARY_THRESHOLD_FINGERTIPS}m, Consecutive={MIN_CONSECUTIVE_FRAMES}',
            'count': len(stationary_frames_action),
            'frames': get_episode_info_for_frames(stationary_frames_action, episode_ends, episode_names)
        }
        
        # ===== 3. Problematic Episodes (Camera Extrinsic Frame-to-Frame Changes) =====
        # 有问题的 Episodes（相机外参相邻帧变化检测）
        print(f"  - Detecting problematic episodes (extrinsic frame-to-frame changes)...")
        
        # Get problematic episodes data
        problematic_episodes = info.get('problematic_episodes_extrinsic', {})
        
        # Store problematic episodes
        dataset_anomalies['anomalies']['problematic_episodes_extrinsic'] = {
            'description': 'Episodes with large frame-to-frame extrinsic changes (translation or rotation)',
            'translation_threshold': f'{problematic_episodes.get("translation_threshold", 0.5):.2f}m',
            'rotation_threshold': f'{problematic_episodes.get("rotation_threshold", 0.3):.2f}',
            'total_episodes': problematic_episodes.get('total_episodes', 0),
            'num_problematic_episodes': problematic_episodes.get('num_problematic_episodes', 0),
            'problematic_episodes': problematic_episodes.get('problematic_episodes', [])
        }

        # ===== 4. Instruction Anomalies =====
        # 指令异常检测
        print(f"  - Detecting instruction anomalies (empty instructions)...")
        
        instruction = info['instruction']
        instruction_num = info['instruction_num']
        
        # Check for empty instructions where instruction_num > 0
        # 检查指令数量大于0但指令内容为空的情况
        
        # Convert instruction to object array if not already
        instruction_arr = np.asarray(instruction, dtype=object)
        instruction_num_arr = np.asarray(instruction_num, dtype=np.int64)
        max_slots = instruction_arr.shape[1] if len(instruction_arr.shape) > 1 else 5
        instruction_num_clipped = np.minimum(instruction_num_arr, max_slots)
        
        # Identify frames with empty instructions
        empty_instruction_frames = []
        
        # Only check frames where instruction_num > 0
        potential_frames = np.where(instruction_num_clipped > 0)[0]
        
        for idx in potential_frames:
            num_valid = instruction_num_clipped[idx]
            # Check the first num_valid slots
            frame_instructions = instruction_arr[idx, :num_valid]
            
            # Check if any instruction is empty/nan/invalid
            has_empty = False
            for inst in frame_instructions:
                try:
                    if isinstance(inst, bytes):
                        inst_str = inst.decode('utf-8', errors='ignore').strip()
                    else:
                        inst_str = str(inst).strip()
                    
                    # 预处理：移除换行符等空白字符后再检查isprintable
                    # Pre-process: remove newlines and other whitespace before checking isprintable
                    # This fixes the issue where valid instructions with newlines were marked as empty
                    check_str = "".join(inst_str.split())
                    
                    if not inst_str or not check_str.isprintable() or inst_str.lower() == 'nan':
                        has_empty = True
                        break
                except:
                    has_empty = True
                    break
            
            if has_empty:
                empty_instruction_frames.append(idx)
        
        empty_instruction_frames = np.array(empty_instruction_frames)
        
        dataset_anomalies['anomalies']['empty_instructions'] = {
            'description': 'Frames with empty instructions (where instruction_num > 0)',
            'count': len(empty_instruction_frames),
            'frames': get_episode_info_for_frames(empty_instruction_frames, episode_ends, episode_names)
        }
        
        # ===== Summary Statistics =====
        # 汇总统计
        dataset_anomalies['summary'] = {
            'diff_state_action_extreme_errors_total': len(extreme_frames_trans) + len(extreme_frames_rot) + len(extreme_frames_fing),
            'diff_state_action_outliers_total': len(outlier_frames_abs_trans) + len(outlier_frames_abs_rot) + len(outlier_frames_abs_fing),
            'stationary_frames_state_count': len(stationary_frames_state),
            'stationary_frames_action_count': len(stationary_frames_action),
            'problematic_episodes_extrinsic_count': problematic_episodes.get('num_problematic_episodes', 0),
            'empty_instructions_count': len(empty_instruction_frames)
        }
        
        anomaly_report[name] = dataset_anomalies
        
        print(f"  ✓ Collected anomaly data for {name}")
        print(f"    - State-Action difference extreme errors: {dataset_anomalies['summary']['diff_state_action_extreme_errors_total']}")
        print(f"    - State-Action difference outliers: {dataset_anomalies['summary']['diff_state_action_outliers_total']}")
        print(f"    - Stationary frames (State): {dataset_anomalies['summary']['stationary_frames_state_count']}")
        print(f"    - Stationary frames (Action): {dataset_anomalies['summary']['stationary_frames_action_count']}")
        print(f"    - Problematic episodes (extrinsic): {dataset_anomalies['summary']['problematic_episodes_extrinsic_count']}")
        print(f"    - Empty instructions: {dataset_anomalies['summary']['empty_instructions_count']}")
    
    return anomaly_report


def main():
    """Main analysis function (主分析函数)"""
    # Parse command line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Zarr Dataset Episode Analysis (Simple Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 自动识别：Zarr 目录 或 本仓库导出的 .tar（默认推荐）
  python analyze_episodes_full.py --data-format auto --zarr-paths path1.zarr --dataset-names D1

  # 先导出再分析（拷贝 tar 到分析机即可）
  python export_zarr_to_wds_tar.py --zarr path1.zarr --out d1.tar
  python analyze_episodes_full.py --zarr-paths d1.tar --dataset-names D1

  # 远程 / 通配 / pipe 等 WebDataset URI（需显式 webdataset，或 URI 带 * pipe: http 等会自动走 WDS）
  python analyze_episodes_full.py --data-format webdataset --zarr-paths 'pipe:curl -Ls ...' --dataset-names D1
        """
    )
    parser.add_argument(
        '--data-format',
        type=normalize_data_format,
        default='auto',
        help='auto：自动区分 Zarr 目录、本地 tar、WebDataset URI；'
             'zarr / webdataset：强制使用该读取器（见 episode_source 模块说明）',
    )
    parser.add_argument(
        '--input-source',
        dest='_legacy_input_source',
        default=None,
        metavar='FMT',
        help='已弃用，请改用 --data-format（wds-tar / wds-url 与 webdataset 等价）',
    )
    parser.add_argument(
        '--zarr-paths',
        nargs='+',
        type=str,
        default=DEFAULT_ZARR_PATHS,
        help='每个数据集一条路径：Zarr 根目录、export_zarr_to_wds_tar 生成的 .tar、或 WebDataset URI',
    )
    parser.add_argument(
        '--dataset-names',
        nargs='+',
        type=str,
        default=DEFAULT_DATASET_NAMES,
        help='Dataset names corresponding to zarr paths (default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for PDF and JSON files (default: ./output relative to script directory)'
    )
    parser.add_argument(
        '--pdf-name',
        type=str,
        default='zarr_episode_analysis.pdf',
        help='Output PDF filename (default: %(default)s)'
    )
    parser.add_argument(
        '--json-name',
        type=str,
        default='zarr_anomaly_report.json',
        help='Output JSON filename (default: %(default)s)'
    )
    
    args = parser.parse_args()

    if not args.zarr_paths:
        parser.error(
            "未指定 --zarr-paths。请显式传入路径，或设置环境变量 DEFAULT_ZARR_SCAN_DIR 指向包含 zarr 子目录的根路径。"
        )

    if len(args.zarr_paths) != len(args.dataset_names):
        parser.error(
            f"--zarr-paths 数量 ({len(args.zarr_paths)}) 与 --dataset-names ({len(args.dataset_names)}) 不一致"
        )

    if args._legacy_input_source is not None:
        if args.data_format != "auto":
            parser.error("不能同时使用 --data-format 与已弃用的 --input-source")
        data_format_eff: str = normalize_data_format(args._legacy_input_source)
        import warnings

        warnings.warn(
            "--input-source 已弃用，请改用 --data-format",
            DeprecationWarning,
            stacklevel=1,
        )
    else:
        data_format_eff = args.data_format

    print("=" * 60)
    print("Zarr Dataset Episode Analysis (Simple Version)")
    print("=" * 60)
    print()
    print(f"Data format: {data_format_eff}")
    print(f"Data paths: {args.zarr_paths}")
    print(f"Dataset names: {args.dataset_names}")
    print()

    datasets_info = {}

    for zarr_path, dataset_name in zip(args.zarr_paths, args.dataset_names):
        try:
            data_dict, resolved = load_episode_bundle(zarr_path, data_format=data_format_eff)
            print(f"  Loaded via {resolved}: {zarr_path}")
            episode_lengths = calculate_episode_lengths(data_dict['episode_ends'])
            
            # Detect problematic episodes based on extrinsic frame-to-frame changes
            # 使用胸部左相机的外参进行检测
            print(f"  Detecting problematic episodes for {dataset_name}...")
            problematic_episodes = detect_problematic_episodes_extrinsic(
                data_dict['extrinsic_bl'],
                data_dict['episode_ends'],
                data_dict['episode_names'],
                translation_threshold=0.5,  # 0.5m translation threshold
                rotation_threshold=1     # Rotation matrix Frobenius norm threshold (~60 degrees)
            )
            
            # 使用胸部相机的数据作为主要分析数据，保持与原代码的兼容性
            datasets_info[dataset_name] = {
                'episode_names': data_dict['episode_names'],
                'episode_ends': data_dict['episode_ends'],
                'lengths': episode_lengths,
                'instruction': data_dict['instruction'],
                'instruction_num': data_dict['instruction_num'],
                'intrinsic_h': data_dict['intrinsic_h'],
                'intrinsic_b': data_dict['intrinsic_b'],
                'intrinsic': data_dict['intrinsic_b'],  # 为了兼容性，使用胸部相机内参
                'extrinsic_hl': data_dict['extrinsic_hl'],
                'extrinsic_hr': data_dict['extrinsic_hr'],
                'extrinsic_bl': data_dict['extrinsic_bl'],
                'extrinsic_br': data_dict['extrinsic_br'],
                'extrinsic': data_dict['extrinsic_bl'],  # 为了兼容性，使用胸部左相机外参
                'joint_state': data_dict['joint_state'],
                'joint_action': data_dict['joint_action'],
                'wrist_state_h': data_dict['wrist_state_h'],
                'wrist_action_h': data_dict['wrist_action_h'],
                'wrist_state_b': data_dict['wrist_state_b'],
                'wrist_action_b': data_dict['wrist_action_b'],
                'wrist_state': data_dict['wrist_state_b'],  # 为了兼容性，使用胸部相机wrist数据
                'wrist_action': data_dict['wrist_action_b'],
                'fingertips_state_h': data_dict['fingertips_state_h'],
                'fingertips_action_h': data_dict['fingertips_action_h'],
                'fingertips_state_b': data_dict['fingertips_state_b'],
                'fingertips_action_b': data_dict['fingertips_action_b'],
                'fingertips_state': data_dict['fingertips_state_b'],  # 为了兼容性，使用胸部相机fingertips数据
                'fingertips_action': data_dict['fingertips_action_b'],
                'problematic_episodes_extrinsic': problematic_episodes
            }
            
            print(f"    - Problematic episodes: {problematic_episodes['num_problematic_episodes']:,} / {problematic_episodes['total_episodes']:,}")
            print()
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            print()
            import traceback
            traceback.print_exc()
            continue
    
    if not datasets_info:
        print("No data loaded successfully. Exiting.")
        return

    # 新增代码：merge-dataset
    # 合成大字典
    merged_dict = {}
    merged_dict['teleop'] = merge_datasets_info(datasets_info)
    datasets_info = merged_dict
    episode_lengths = calculate_episode_lengths(datasets_info['teleop']['episode_ends'])
    datasets_info['teleop']['lengths'] = episode_lengths
    
    print("=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # Create output directory if not exists
    # 如果输出目录不存在则创建
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output PDF
    # 创建输出 PDF 文件
    output_path = output_dir / args.pdf_name
    
    with PdfPages(output_path) as pdf:
        # Page 1: Dataset Size Pie Charts
        # 第1页: 数据集大小饼图
        print("Creating dataset size pie charts...")
        fig = plot_dataset_size_pie_charts(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Statistics Table
        # 第2页: 基础统计表格
        print("Creating statistics table...")
        fig = create_statistics_table(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Episode Length Distribution (Histograms)
        # 第3页: Episode 长度分布直方图
        print("Creating episode length distributions...")
        fig = plot_episode_length_distribution(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Data Quality Table & Presence Distribution (Combined)
        # 第4页: 数据质量表格 & 手部存在分布 (合并页)
        print("Creating data quality and presence analysis...")
        fig = plot_data_quality_and_presence(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Instruction Number Distribution
        # 第5页: 指令数量分布
        print("Creating instruction number distribution...")
        fig = plot_instruction_num_distribution(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Top Instructions
        # 第6页: 高频指令统计
        print("Creating top instructions analysis...")
        fig = plot_top_instructions(datasets_info, top_n=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 7: Camera Intrinsic Parameters Analysis
        # 第7页: 相机内参分析
        print("Creating camera intrinsic parameters analysis...")
        fig = plot_camera_parameters(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 8: Data Quality Anomalies (IQR Method)
        # 第8页: 数据质量异常检测 (IQR 方法)
        print("Creating data quality anomalies analysis (IQR Method)...")
        fig = plot_data_quality_anomalies(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 9: Wrist Displacement Statistics
        # 第9页: 手腕位移统计
        print("Creating wrist displacement analysis...")
        fig = plot_wrist_displacement(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 10: Fingertips Movement Range
        # 第10页: 指尖移动范围
        print("Creating fingertips movement analysis...")
        fig = plot_fingertips_movement(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 11: Outlier Distribution Analysis & Error Detection
        # 第11页: 异常分布分析与错误检测 (绝对阈值法)
        print("Creating outlier distribution analysis and error detection...")
        fig = plot_outlier_distribution_analysis(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 12: Frame Change Rate Analysis (State)
        # 第12页: 帧间变化率分析 (State)
        print("Creating frame change rate analysis (State)...")
        fig = plot_frame_change_rate(datasets_info, data_type='state')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 13: Frame Change Rate Analysis (Action)
        # 第13页: 帧间变化率分析 (Action)
        print("Creating frame change rate analysis (Action)...")
        fig = plot_frame_change_rate(datasets_info, data_type='action')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 14: Frame Change Rate Outlier Analysis (State)
        # 第14页: 帧间变化率异常分析 (State)
        print("Creating frame change rate outlier analysis (State)...")
        fig = plot_frame_change_rate_outliers(datasets_info, data_type='state')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 15: Frame Change Rate Outlier Analysis (Action)
        # 第15页: 帧间变化率异常分析 (Action)
        print("Creating frame change rate outlier analysis (Action)...")
        fig = plot_frame_change_rate_outliers(datasets_info, data_type='action')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 12: Stationary Frame Detection (State & Action)
        # 第16页: 静止帧检测 (State & Action)
        print("Creating stationary frame detection analysis...")
        fig = plot_stationary_frame_detection(datasets_info)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Zarr Dataset Comprehensive Analysis (Simple Version)'
        d['Author'] = 'Zarr Analysis Script'
        d['Subject'] = 'Episode, Presence, Instruction and Quality Statistics'
        d['Keywords'] = 'Zarr, Episode, Presence, Instruction, Analysis, Statistics'
    
    print()
    print("=" * 60)
    print(f"Analysis complete! PDF saved to: {output_path}")
    print("=" * 60)
    
    # ===== Collect and Export Anomaly Data to JSON =====
    # 收集并导出异常数据到 JSON 文件
    print()
    print("=" * 60)
    print("Collecting anomaly data and exporting to JSON...")
    print("=" * 60)
    
    anomaly_report = collect_anomaly_data(datasets_info)
    
    # Save to JSON file
    # 保存到 JSON 文件
    json_output_path = output_dir / args.json_name
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(anomaly_report, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print(f"Anomaly report saved to: {json_output_path}")
    print("=" * 60)
    print()
    print("JSON report contains:")
    print("  - State-Action difference outliers (normal and extreme errors)")
    print("  - Stationary frames detection (State and Action)")
    print("  - Problematic episodes (camera extrinsic maximum differences)")
    print()
    print("Problematic Episodes Detection:")
    print("  - Enumerates ALL frame pairs within each episode")
    print("  - Finds maximum translation distance across all pairs")
    print("  - Finds maximum rotation matrix distance across all pairs")
    print("  - Translation threshold: 0.5m")
    print("  - Rotation matrix Frobenius norm threshold: 1 (~40 degrees)")
    print("  - If either maximum exceeds threshold, episode is marked problematic")
    print()
    print("Each frame-level anomaly entry includes:")
    print("  - frame_idx: Global frame index in the dataset")
    print("  - episode_idx: Episode index")
    print("  - episode_name: Name of the episode")
    print("  - frame_in_episode: Frame index within the episode")
    print("  - episode_start: Start frame of the episode")
    print("  - episode_end: End frame of the episode")
    print()
    print("Each problematic episode entry includes:")
    print("  - episode_idx, episode_name: Episode identification")
    print("  - episode_start, episode_end, episode_length: Episode boundaries")
    print("  - max_translation_diff: Maximum translation distance in episode")
    print("  - max_translation_frame_pair: Frame pair with maximum translation")
    print("  - max_rotation_diff: Maximum rotation distance in episode")
    print("  - max_rotation_frame_pair: Frame pair with maximum rotation")
    print("  - translation_exceeds, rotation_exceeds: Which threshold was exceeded")
    print("=" * 60)


if __name__ == "__main__":
    main()