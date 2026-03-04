#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双极导联脑网络连接性可视化模块

支持21电极双极导联的26通道脑网络可视化：
1. 矩阵热力图 - 按脑区组织的26×26连接矩阵
2. 圆形连接图 - 节点按脑区排列在圆周上

5脑区定义：
- 左额 (left_frontal): FP1-F7, FP1-F3, F7-F3, F3-FZ
- 左颞 (left_temporal): F7-SPHL, SPHL-T3, T3-T5, T5-O1, T3-C3, T5-P3
- 顶叶 (parietal): FZ-CZ, C3-CZ, P3-PZ, CZ-PZ, CZ-C4, PZ-P4
- 右额 (right_frontal): FP2-F4, FP2-F8, F4-F8, FZ-F4
- 右颞 (right_temporal): F8-SPHR, SPHR-T4, C4-T4, T4-T6, P4-T6, T6-O2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ==============================================================================
# 21电极双极导联定义（26通道）
# ==============================================================================

BIPOLAR_26_CHANNELS = [
    # 左额 (4对)
    'FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ',
    # 左颞 (6对)
    'F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3',
    # 顶叶 (6对)
    'FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4',
    # 右额 (4对)
    'FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4',
    # 右颞 (6对)
    'F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2',
]

# 双极导联脑区定义（5脑区）
BIPOLAR_REGION_DEFINITIONS = {
    'left_frontal': ['FP1-F7', 'FP1-F3', 'F7-F3', 'F3-FZ'],
    'left_temporal': ['F7-SPHL', 'SPHL-T3', 'T3-T5', 'T5-O1', 'T3-C3', 'T5-P3'],
    'parietal': ['FZ-CZ', 'C3-CZ', 'P3-PZ', 'CZ-PZ', 'CZ-C4', 'PZ-P4'],
    'right_frontal': ['FP2-F4', 'FP2-F8', 'F4-F8', 'FZ-F4'],
    'right_temporal': ['F8-SPHR', 'SPHR-T4', 'C4-T4', 'T4-T6', 'P4-T6', 'T6-O2'],
}

# 脑区颜色（5脑区）
BIPOLAR_REGION_COLORS = {
    'left_frontal': '#E74C3C',      # 红色 - 左额
    'left_temporal': '#F39C12',     # 橙色 - 左颞
    'parietal': '#9B59B6',          # 紫色 - 顶叶
    'right_frontal': '#3498DB',     # 蓝色 - 右额
    'right_temporal': '#27AE60',    # 绿色 - 右颞
}

# 脑区显示名称
BIPOLAR_REGION_DISPLAY_NAMES = {
    'left_frontal': 'L-Frontal',
    'left_temporal': 'L-Temporal',
    'parietal': 'Parietal',
    'right_frontal': 'R-Frontal',
    'right_temporal': 'R-Temporal',
}

# ==============================================================================
# 旧版18通道定义（保持向后兼容）
# ==============================================================================
BIPOLAR_18_CHANNELS = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',      # 左颞链 (4)
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',      # 右颞链 (4)
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',      # 左旁正中链 (4)
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',      # 右旁正中链 (4)
    'FZ-CZ', 'CZ-PZ'                           # 中线链 (2)
]

# 旧版脑区定义（保持向后兼容）
BIPOLAR_REGION_DEFINITIONS_18 = {
    'left_temporal': ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1'],
    'right_temporal': ['FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2'],
    'left_parasagittal': ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1'],
    'right_parasagittal': ['FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
    'midline': ['FZ-CZ', 'CZ-PZ']
}

# 旧版脑区颜色（18通道向后兼容）
BIPOLAR_REGION_COLORS_18 = {
    'left_temporal': '#E74C3C',       # 红色
    'right_temporal': '#F39C12',      # 橙色
    'left_parasagittal': '#27AE60',   # 绿色
    'right_parasagittal': '#3498DB',  # 蓝色
    'midline': '#9B59B6'              # 紫色
}

# 旧版脑区显示名称（18通道向后兼容）
BIPOLAR_REGION_DISPLAY_NAMES_18 = {
    'left_temporal': 'L-Temp',
    'right_temporal': 'R-Temp',
    'left_parasagittal': 'L-Para',
    'right_parasagittal': 'R-Para',
    'midline': 'Midline'
}

# 合并所有颜色定义（同时支持新旧命名）
BIPOLAR_REGION_COLORS.update(BIPOLAR_REGION_COLORS_18)
BIPOLAR_REGION_DISPLAY_NAMES.update(BIPOLAR_REGION_DISPLAY_NAMES_18)


def get_bipolar_region_order_indices(use_26_channels: bool = True) -> Tuple[List[int], List[str], Dict[str, Tuple[int, int]]]:
    """
    获取按脑区排序的双极导联通道索引
    
    Args:
        use_26_channels: 是否使用26通道模式（21电极），否则使用18通道模式（19电极）
    
    Returns:
        ordered_indices: 按脑区排序的通道索引
        ordered_names: 按脑区排序的通道名称
        region_bounds: 各脑区的起止索引 {region_name: (start, end)}
    """
    ordered_indices = []
    ordered_names = []
    region_bounds = {}
    
    if use_26_channels:
        # 26通道 / 5脑区模式
        channels = BIPOLAR_26_CHANNELS
        regions = ['left_frontal', 'left_temporal', 'parietal', 'right_frontal', 'right_temporal']
        region_defs = BIPOLAR_REGION_DEFINITIONS
    else:
        # 18通道 / 5脑区模式（向后兼容）
        channels = BIPOLAR_18_CHANNELS
        regions = ['left_temporal', 'right_temporal', 'left_parasagittal', 'right_parasagittal', 'midline']
        region_defs = BIPOLAR_REGION_DEFINITIONS_18
    
    current_idx = 0
    for region in regions:
        region_channels = region_defs[region]
        start_idx = current_idx
        
        for ch in region_channels:
            if ch in channels:
                orig_idx = channels.index(ch)
                ordered_indices.append(orig_idx)
                ordered_names.append(ch)
                current_idx += 1
        
        region_bounds[region] = (start_idx, current_idx)
    
    return ordered_indices, ordered_names, region_bounds


def reorder_bipolar_matrix_by_region(matrix: np.ndarray, use_26_channels: bool = True) -> np.ndarray:
    """
    按脑区重新排列双极导联连接矩阵
    
    Args:
        matrix: 原始连接性矩阵
        use_26_channels: 是否使用26通道模式
    """
    ordered_indices, _, _ = get_bipolar_region_order_indices(use_26_channels)
    reordered = matrix[np.ix_(ordered_indices, ordered_indices)]
    return reordered


def apply_bipolar_connection_filters(
    matrix: np.ndarray,
    threshold: float = None,
    percentile: float = None,
    top_k: int = None,
    zscore_threshold: float = None,
    remove_outliers: bool = True,
    exclude_diagonal: bool = True
) -> np.ndarray:
    """
    应用连接过滤器（双极导联版本）
    """
    filtered = matrix.copy()
    
    # 排除对角线
    if exclude_diagonal:
        np.fill_diagonal(filtered, 0)
    
    # z-score过滤
    if zscore_threshold is not None:
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        mean = np.mean(upper_tri)
        std = np.std(upper_tri)
        if std > 1e-10:
            z_scores = (matrix - mean) / std
            if remove_outliers:
                filtered[np.abs(z_scores) > zscore_threshold] = 0
    
    # 绝对阈值
    if threshold is not None:
        filtered[filtered < threshold] = 0
    
    # 百分位阈值
    if percentile is not None:
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        percentile_value = np.percentile(upper_tri, percentile)
        filtered[filtered < percentile_value] = 0
    
    # Top-K
    if top_k is not None:
        upper_tri_indices = np.triu_indices_from(matrix, k=1)
        values = matrix[upper_tri_indices]
        if len(values) > top_k:
            threshold_val = np.sort(values)[-top_k]
            new_filtered = np.zeros_like(filtered)
            for idx in range(len(values)):
                i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
                if values[idx] >= threshold_val:
                    new_filtered[i, j] = matrix[i, j]
                    new_filtered[j, i] = matrix[j, i]
            filtered = new_filtered
    
    return filtered


def plot_bipolar_connectivity_matrix(
    matrix: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = 'RdYlBu_r',
    vmin: float = None,
    vmax: float = None,
    use_26_channels: bool = None
):
    """
    绘制双极导联连接性矩阵热力图
    
    Args:
        matrix: 连接性矩阵 (n_channels x n_channels)
        title: 图表标题
        save_path: 保存路径
        cmap: 颜色映射
        vmin/vmax: 颜色范围
        use_26_channels: 是否使用26通道模式，None表示自动检测
    """
    n_channels = matrix.shape[0]
    
    # 自动检测通道模式
    if use_26_channels is None:
        use_26_channels = n_channels == 26 or n_channels > 18
    
    # 按脑区重排矩阵
    reordered_matrix = reorder_bipolar_matrix_by_region(matrix, use_26_channels)
    ordered_indices, ordered_names, region_bounds = get_bipolar_region_order_indices(use_26_channels)
    
    n_display = len(ordered_names)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 14) if use_26_channels else (14, 12))
    
    # 设置颜色范围
    if vmin is None:
        vmin = np.nanmin(reordered_matrix)
    if vmax is None:
        vmax = np.nanmax(reordered_matrix)
    
    # 绘制热力图
    im = ax.imshow(reordered_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Connectivity Strength', fontsize=11)
    
    # 设置通道标签（简化显示）
    short_names = [n.replace('-', '→') for n in ordered_names]
    ax.set_xticks(range(n_display))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6 if use_26_channels else 7)
    ax.set_yticks(range(n_display))
    ax.set_yticklabels(short_names, fontsize=6 if use_26_channels else 7)
    
    # 添加脑区分隔线和背景色块
    for region, (start, end) in region_bounds.items():
        if region not in BIPOLAR_REGION_COLORS:
            continue
        color = BIPOLAR_REGION_COLORS[region]
        
        # 绘制分隔线
        if start > 0:
            ax.axhline(y=start - 0.5, color='white', linewidth=2)
            ax.axvline(x=start - 0.5, color='white', linewidth=2)
        
        # 边缘颜色条
        rect_left = mpatches.Rectangle(
            (-1.5, start - 0.5), 0.8, end - start,
            linewidth=0, facecolor=color, alpha=0.8
        )
        ax.add_patch(rect_left)
        
        rect_top = mpatches.Rectangle(
            (start - 0.5, -1.5), end - start, 0.8,
            linewidth=0, facecolor=color, alpha=0.8
        )
        ax.add_patch(rect_top)
    
    ax.set_xlim(-2, n_display)
    ax.set_ylim(n_display, -2)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_patches = []
    for region in region_bounds.keys():
        if region in BIPOLAR_REGION_COLORS and region in BIPOLAR_REGION_DISPLAY_NAMES:
            legend_patches.append(
                mpatches.Patch(color=BIPOLAR_REGION_COLORS[region], 
                              label=BIPOLAR_REGION_DISPLAY_NAMES[region])
            )
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.12, 1.0),
              title='Regions', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_bipolar_circular_connectivity(
    matrix: np.ndarray,
    title: str,
    save_path: str,
    threshold: float = 0.0,
    line_cmap: str = 'Reds',
    max_lines: int = 80,
    is_directed: bool = False,
    use_26_channels: bool = None
):
    """
    绘制双极导联圆形脑网络连接图
    
    Args:
        matrix: 连接性矩阵 (n_channels x n_channels)
        title: 图表标题
        save_path: 保存路径
        threshold: 连接强度阈值
        line_cmap: 连接线颜色映射
        max_lines: 最大显示连接数
        is_directed: 是否为有向连接
        use_26_channels: 是否使用26通道模式，None表示自动检测
    """
    n_matrix = matrix.shape[0]
    
    # 自动检测通道模式
    if use_26_channels is None:
        use_26_channels = n_matrix == 26 or n_matrix > 18
    
    # 获取按脑区排序的通道
    ordered_indices, ordered_names, region_bounds = get_bipolar_region_order_indices(use_26_channels)
    n_channels = len(ordered_names)
    
    # 按脑区重排矩阵
    reordered_matrix = reorder_bipolar_matrix_by_region(matrix, use_26_channels)
    
    # 获取脑区列表
    regions = list(region_bounds.keys())
    
    # 创建图形
    if is_directed:
        fig, ax = plt.subplots(figsize=(14, 14) if use_26_channels else (12, 12))
        ax.set_aspect('equal')
    else:
        fig, ax = plt.subplots(figsize=(14, 14) if use_26_channels else (12, 12), 
                               subplot_kw={'projection': 'polar'})
    
    ax.set_facecolor('white')
    
    # 计算节点角度
    gap_angle = np.pi / 30
    total_gap = len(regions) * gap_angle
    available_angle = 2 * np.pi - total_gap
    
    angles = []
    current_angle = np.pi / 2
    
    for region in regions:
        start, end = region_bounds[region]
        n_ch = end - start
        region_angle = available_angle * (n_ch / n_channels)
        
        for i in range(n_ch):
            angle = current_angle - (i + 0.5) * region_angle / n_ch
            angles.append(angle)
        
        current_angle -= region_angle + gap_angle
    
    angles = np.array(angles)
    node_r = 1.0
    
    if is_directed:
        node_x = node_r * np.cos(angles)
        node_y = node_r * np.sin(angles)
    
    # 绘制脑区弧线
    for region in regions:
        start, end = region_bounds[region]
        if region not in BIPOLAR_REGION_COLORS:
            continue
        color = BIPOLAR_REGION_COLORS[region]
        
        if end <= start:
            continue
            
        arc_start = angles[start] + 0.03
        arc_end = angles[end - 1] - 0.03
        arc_angles = np.linspace(arc_start, arc_end, 50)
        arc_r = np.ones(50) * 1.08
        
        if is_directed:
            arc_x = arc_r * np.cos(arc_angles)
            arc_y = arc_r * np.sin(arc_angles)
            ax.plot(arc_x, arc_y, color=color, linewidth=10, solid_capstyle='round')
        else:
            ax.plot(arc_angles, arc_r, color=color, linewidth=10, solid_capstyle='round')
    
    # 绘制节点和标签
    for i, (angle, name) in enumerate(zip(angles, ordered_names)):
        # 找到节点所属的脑区
        node_color = '#888888'  # 默认颜色
        for region in regions:
            start, end = region_bounds[region]
            if start <= i < end and region in BIPOLAR_REGION_COLORS:
                node_color = BIPOLAR_REGION_COLORS[region]
                break
        
        if is_directed:
            ax.scatter(node_x[i], node_y[i], s=100 if use_26_channels else 120, 
                      c=node_color, edgecolors='white', linewidth=1.5, zorder=5)
            label_r = 1.20
            label_x = label_r * np.cos(angle)
            label_y = label_r * np.sin(angle)
            # 简化标签
            parts = name.split('-')
            short_name = parts[0][:3] + '-' + parts[1][:3] if len(parts) == 2 else name[:6]
            ax.text(label_x, label_y, short_name, fontsize=5 if use_26_channels else 6, 
                   ha='center', va='center', fontweight='bold')
        else:
            ax.scatter(angle, node_r, s=100 if use_26_channels else 120, 
                      c=node_color, edgecolors='white', linewidth=1.5, zorder=5)
            label_r = 1.18
            parts = name.split('-')
            short_name = parts[0][:3] + '-' + parts[1][:3] if len(parts) == 2 else name[:6]
            ax.text(angle, label_r, short_name, fontsize=5 if use_26_channels else 6, 
                   ha='center', va='center', fontweight='bold')
    
    # 收集连接
    connections = []
    if is_directed:
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j and reordered_matrix[i, j] > threshold:
                    connections.append((i, j, reordered_matrix[i, j]))
    else:
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if reordered_matrix[i, j] > threshold:
                    connections.append((i, j, reordered_matrix[i, j]))
    
    connections.sort(key=lambda x: x[2], reverse=True)
    connections = connections[:max_lines]
    
    # 绘制连接线
    if connections:
        strengths = [c[2] for c in connections]
        norm = Normalize(vmin=min(strengths), vmax=max(strengths))
        colormap = cm.get_cmap(line_cmap)
        
        for i, j, strength in connections:
            linewidth = 0.5 + 2 * norm(strength)
            color = colormap(norm(strength))
            
            if is_directed:
                x1, y1 = node_x[i], node_y[i]
                x2, y2 = node_x[j], node_y[j]
                
                # 侧向偏移
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 1e-6:
                    nx, ny = -dy / length, dx / length
                else:
                    nx, ny = 0, 0
                
                offset = 0.06
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                curve_factor = 0.3 + 0.2 * (1 - distance / 2)
                
                ctrl_x = mid_x * curve_factor + nx * offset
                ctrl_y = mid_y * curve_factor + ny * offset
                
                n_points = 30
                t = np.linspace(0, 1, n_points)
                curve_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                curve_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                
                ax.plot(curve_x, curve_y, color=color, linewidth=linewidth, alpha=0.7)
                
                # 箭头
                arrow_idx = int(0.85 * (n_points - 1))
                ax.annotate('', 
                    xy=(curve_x[arrow_idx + 2], curve_y[arrow_idx + 2]),
                    xytext=(curve_x[arrow_idx], curve_y[arrow_idx]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                   mutation_scale=8 + 4 * norm(strength)))
            else:
                angle_i, angle_j = angles[i], angles[j]
                n_points = 50
                t = np.linspace(0, 1, n_points)
                r_mid = 0.3 + 0.4 * (1 - abs(angle_i - angle_j) / np.pi)
                r = node_r * (1 - 4 * t * (1 - t) * (1 - r_mid))
                theta = angle_i + t * (angle_j - angle_i)
                ax.plot(theta, r, color=color, linewidth=linewidth, alpha=0.7)
    
    # 设置坐标属性
    if is_directed:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_ylim(0, 1.35)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
    
    direction_label = " (Directed)" if is_directed else ""
    ax.set_title(f"{title}{direction_label}", fontsize=14, fontweight='bold', pad=20)
    
    # 创建图例
    legend_patches = []
    for region in regions:
        if region in BIPOLAR_REGION_COLORS and region in BIPOLAR_REGION_DISPLAY_NAMES:
            legend_patches.append(
                mpatches.Patch(color=BIPOLAR_REGION_COLORS[region], 
                              label=BIPOLAR_REGION_DISPLAY_NAMES[region])
            )
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.0, 0.5),
               title='Regions', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_bipolar_connectivity(
    connectivity_dict: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = '',
    percentile: float = 70.0,
    zscore_threshold: float = 3.0,
    max_lines: int = 60
):
    """可视化双极导联的所有连接性特征"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    feature_names = {
        'plv': 'PLV (Bipolar)',
        'wpli': 'wPLI (Bipolar)',
        'granger_causality': 'Granger Causality (Bipolar)',
        'transfer_entropy': 'Transfer Entropy (Bipolar)',
        'pearson_corr': 'Pearson Corr (Bipolar)',
        'aec': 'AEC (Bipolar)'
    }
    
    for feature_key, matrix in connectivity_dict.items():
        feature_title = feature_names.get(feature_key, feature_key)
        file_prefix = f"{prefix}_{feature_key}" if prefix else feature_key
        
        is_directed = feature_key in ['granger_causality', 'transfer_entropy']
        
        if feature_key in ['pearson_corr']:
            work_matrix = np.abs(matrix)
        else:
            work_matrix = matrix.copy()
        
        # 应用过滤器
        filtered_matrix = apply_bipolar_connection_filters(
            work_matrix,
            percentile=percentile,
            zscore_threshold=zscore_threshold,
            remove_outliers=True,
            exclude_diagonal=True
        )
        
        # 矩阵热力图
        matrix_path = output_path / f"{file_prefix}_matrix.png"
        plot_bipolar_connectivity_matrix(
            filtered_matrix,
            title=feature_title,
            save_path=str(matrix_path)
        )
        
        # 圆形连接图
        circular_path = output_path / f"{file_prefix}_circular.png"
        plot_bipolar_circular_connectivity(
            filtered_matrix,
            title=feature_title,
            save_path=str(circular_path),
            max_lines=max_lines,
            is_directed=is_directed
        )
    
    print(f"双极导联连接性可视化完成: {output_dir}")


if __name__ == '__main__':
    print("测试双极导联连接性可视化...")
    
    # 测试18通道模式（向后兼容）
    print("\n--- 测试18通道模式 ---")
    np.random.seed(42)
    test_matrix_18 = np.random.rand(18, 18)
    test_matrix_18 = (test_matrix_18 + test_matrix_18.T) / 2
    np.fill_diagonal(test_matrix_18, 1.0)
    
    plot_bipolar_connectivity_matrix(
        test_matrix_18,
        title="Test Bipolar Matrix (18ch)",
        save_path="test_bipolar_matrix_18.png",
        use_26_channels=False
    )
    print("保存: test_bipolar_matrix_18.png")
    
    plot_bipolar_circular_connectivity(
        test_matrix_18,
        title="Test Bipolar Circular (18ch)",
        save_path="test_bipolar_circular_18.png",
        threshold=0.5,
        use_26_channels=False
    )
    print("保存: test_bipolar_circular_18.png")
    
    # 测试26通道模式（21电极）
    print("\n--- 测试26通道模式 ---")
    np.random.seed(42)
    test_matrix_26 = np.random.rand(26, 26)
    test_matrix_26 = (test_matrix_26 + test_matrix_26.T) / 2
    np.fill_diagonal(test_matrix_26, 1.0)
    
    plot_bipolar_connectivity_matrix(
        test_matrix_26,
        title="Test Bipolar Matrix (26ch)",
        save_path="test_bipolar_matrix_26.png"
    )
    print("保存: test_bipolar_matrix_26.png")
    
    plot_bipolar_circular_connectivity(
        test_matrix_26,
        title="Test Bipolar Circular (26ch)",
        save_path="test_bipolar_circular_26.png",
        threshold=0.5
    )
    print("保存: test_bipolar_circular_26.png")
    
    print("\n测试完成!")
