#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑网络连接性可视化模块

两种可视化类型：
1. 矩阵热力图 - 按脑区组织的19×19连接矩阵
2. 圆形连接图 - 节点按脑区排列在圆周上
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 标准19通道
STANDARD_19_CHANNELS = [
    'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
    't3', 'c3', 'cz', 'c4', 't4',
    't5', 'p3', 'pz', 'p4', 't6',
    'o1', 'o2'
]

# 脑区定义
REGION_DEFINITIONS = {
    'frontal': ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8'],
    'temporal': ['t3', 't4', 't5', 't6'],
    'central': ['c3', 'cz', 'c4'],
    'parietal': ['p3', 'pz', 'p4'],
    'occipital': ['o1', 'o2']
}

# 脑区颜色 (参考样例图)
REGION_COLORS = {
    'frontal': '#E74C3C',    # 红色
    'temporal': '#F39C12',   # 橙色
    'central': '#27AE60',    # 绿色
    'parietal': '#3498DB',   # 蓝色
    'occipital': '#9B59B6'   # 紫色
}

# 脑区显示名称
REGION_DISPLAY_NAMES = {
    'frontal': 'Frontal',
    'temporal': 'Temporal',
    'central': 'Central',
    'parietal': 'Parietal',
    'occipital': 'Occipital'
}


def get_region_order_indices() -> Tuple[List[int], List[str], Dict[str, Tuple[int, int]]]:
    """
    获取按脑区排序的通道索引
    
    Returns:
        ordered_indices: 按脑区排序后的原始索引
        ordered_names: 按脑区排序后的通道名
        region_bounds: 每个脑区的起止索引 {region: (start, end)}
    """
    ordered_indices = []
    ordered_names = []
    region_bounds = {}
    
    current_idx = 0
    for region in ['frontal', 'temporal', 'central', 'parietal', 'occipital']:
        channels = REGION_DEFINITIONS[region]
        start_idx = current_idx
        
        for ch in channels:
            orig_idx = STANDARD_19_CHANNELS.index(ch)
            ordered_indices.append(orig_idx)
            ordered_names.append(ch.upper())
            current_idx += 1
        
        region_bounds[region] = (start_idx, current_idx)
    
    return ordered_indices, ordered_names, region_bounds


# ==============================================================================
# 连接筛选方法
# ==============================================================================

def filter_connections_by_threshold(
    matrix: np.ndarray, 
    threshold: float = 0.3,
    mode: str = 'absolute'
) -> np.ndarray:
    """
    基于阈值过滤连接
    
    Args:
        matrix: 连接矩阵
        threshold: 阈值
        mode: 阈值模式
            - 'absolute': 绝对阈值，低于threshold的设为0
            - 'percentile': 百分位阈值，低于第threshold百分位的设为0
    
    Returns:
        过滤后的矩阵
    """
    filtered = matrix.copy()
    
    if mode == 'absolute':
        filtered[filtered < threshold] = 0
    elif mode == 'percentile':
        # 只考虑上三角（不含对角线）的值
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        percentile_value = np.percentile(upper_tri, threshold)
        filtered[filtered < percentile_value] = 0
    
    return filtered


def filter_connections_top_k(
    matrix: np.ndarray, 
    top_k: int = 50,
    per_node: bool = False
) -> np.ndarray:
    """
    保留top-K最强连接
    
    Args:
        matrix: 连接矩阵
        top_k: 保留的连接数
        per_node: 如果True，每个节点保留top-K；否则全局保留top-K
    
    Returns:
        过滤后的矩阵
    """
    n = matrix.shape[0]
    filtered = np.zeros_like(matrix)
    
    if per_node:
        # 每个节点保留top-K连接
        for i in range(n):
            row = matrix[i].copy()
            row[i] = 0  # 排除自连接
            top_indices = np.argsort(row)[-top_k:]
            for j in top_indices:
                if row[j] > 0:
                    filtered[i, j] = matrix[i, j]
                    filtered[j, i] = matrix[j, i]
    else:
        # 全局保留top-K连接
        upper_tri_indices = np.triu_indices_from(matrix, k=1)
        values = matrix[upper_tri_indices]
        
        if len(values) > top_k:
            threshold = np.sort(values)[-top_k]
            for idx in range(len(values)):
                i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
                if values[idx] >= threshold:
                    filtered[i, j] = matrix[i, j]
                    filtered[j, i] = matrix[j, i]
        else:
            filtered = matrix.copy()
    
    return filtered


def filter_connections_zscore(
    matrix: np.ndarray, 
    z_threshold: float = 2.0,
    remove_outliers: bool = True
) -> np.ndarray:
    """
    基于z-score过滤异常连接
    
    Args:
        matrix: 连接矩阵
        z_threshold: z-score阈值
        remove_outliers: True-移除离群值，False-保留离群值
    
    Returns:
        过滤后的矩阵
    """
    filtered = matrix.copy()
    
    # 计算上三角值的统计量
    upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
    mean = np.mean(upper_tri)
    std = np.std(upper_tri)
    
    if std < 1e-10:
        return filtered
    
    # 计算z-score
    z_scores = (matrix - mean) / std
    
    if remove_outliers:
        # 将异常高的值设为阈值对应的值
        outlier_mask = np.abs(z_scores) > z_threshold
        filtered[outlier_mask] = 0
    else:
        # 只保留显著连接
        filtered[np.abs(z_scores) < z_threshold] = 0
    
    return filtered


def filter_connections_exclude_neighbors(
    matrix: np.ndarray, 
    min_distance: int = 2
) -> np.ndarray:
    """
    排除相邻通道的连接（可能受体积传导影响）
    
    Args:
        matrix: 连接矩阵
        min_distance: 最小通道距离（按标准顺序）
    
    Returns:
        过滤后的矩阵
    """
    filtered = matrix.copy()
    n = matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            if abs(i - j) < min_distance:
                filtered[i, j] = 0
    
    return filtered


def apply_connection_filters(
    matrix: np.ndarray,
    threshold: float = None,
    threshold_mode: str = 'absolute',
    percentile: float = None,
    top_k: int = None,
    top_k_per_node: bool = False,
    zscore_threshold: float = None,
    remove_outliers: bool = True,
    exclude_diagonal: bool = True,
    exclude_neighbor_distance: int = None
) -> np.ndarray:
    """
    应用多个连接过滤器
    
    Args:
        matrix: 原始连接矩阵
        threshold: 绝对阈值（低于此值设为0）
        threshold_mode: 'absolute' 或 'percentile'
        percentile: 百分位阈值（0-100，低于此百分位设为0）
        top_k: 保留的top-K连接数
        top_k_per_node: top-K是否按节点计算
        zscore_threshold: z-score阈值（过滤异常值）
        remove_outliers: True移除离群值，False只保留显著连接
        exclude_diagonal: 是否排除对角线
        exclude_neighbor_distance: 排除相邻通道距离
    
    Returns:
        过滤后的矩阵
    """
    filtered = matrix.copy()
    
    # 1. 排除对角线
    if exclude_diagonal:
        np.fill_diagonal(filtered, 0)
    
    # 2. 排除相邻通道
    if exclude_neighbor_distance is not None and exclude_neighbor_distance > 1:
        filtered = filter_connections_exclude_neighbors(filtered, exclude_neighbor_distance)

    # 3. z-score过滤
    if zscore_threshold is not None:
        filtered = filter_connections_zscore(filtered, zscore_threshold, remove_outliers)

    # 4. 绝对阈值或百分位阈值
    if threshold is not None:
        filtered = filter_connections_by_threshold(filtered, threshold, threshold_mode)
    elif percentile is not None:
        filtered = filter_connections_by_threshold(filtered, percentile, 'percentile')

    
    # 5. Top-K过滤
    if top_k is not None:
        filtered = filter_connections_top_k(filtered, top_k, top_k_per_node)
    
    return filtered


def reorder_matrix_by_region(matrix: np.ndarray) -> np.ndarray:
    """
    按脑区重新排列连接矩阵
    
    Args:
        matrix: (19, 19) 原始连接矩阵
    
    Returns:
        reordered: (19, 19) 按脑区排序的矩阵
    """
    ordered_indices, _, _ = get_region_order_indices()
    reordered = matrix[np.ix_(ordered_indices, ordered_indices)]
    return reordered


def plot_connectivity_matrix(
    matrix: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = 'RdYlBu_r',
    vmin: float = None,
    vmax: float = None,
    show_region_labels: bool = True
):
    """
    绘制按脑区组织的连接性矩阵热力图
    
    Args:
        matrix: (19, 19) 连接矩阵
        title: 图标题
        save_path: 保存路径
        cmap: 颜色映射
        vmin, vmax: 颜色范围
        show_region_labels: 是否显示脑区标签
    """
    # 按脑区重排矩阵
    reordered_matrix = reorder_matrix_by_region(matrix)
    ordered_indices, ordered_names, region_bounds = get_region_order_indices()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    
    # 设置通道标签
    ax.set_xticks(range(19))
    ax.set_xticklabels(ordered_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(19))
    ax.set_yticklabels(ordered_names, fontsize=8)
    
    # 添加脑区分隔线和背景色块
    for region, (start, end) in region_bounds.items():
        color = REGION_COLORS[region]
        
        # 绘制分隔线
        if start > 0:
            ax.axhline(y=start - 0.5, color='white', linewidth=2)
            ax.axvline(x=start - 0.5, color='white', linewidth=2)
        
        # 在边缘添加脑区颜色条
        # 左侧颜色条
        rect_left = mpatches.Rectangle(
            (-1.5, start - 0.5), 0.8, end - start,
            linewidth=0, facecolor=color, alpha=0.8
        )
        ax.add_patch(rect_left)
        
        # 顶部颜色条
        rect_top = mpatches.Rectangle(
            (start - 0.5, -1.5), end - start, 0.8,
            linewidth=0, facecolor=color, alpha=0.8
        )
        ax.add_patch(rect_top)
    
    # 添加脑区标签
    if show_region_labels:
        for region, (start, end) in region_bounds.items():
            mid = (start + end) / 2 - 0.5
            # 左侧标签
            ax.text(-2.5, mid, REGION_DISPLAY_NAMES[region][0], 
                    fontsize=9, fontweight='bold', color=REGION_COLORS[region],
                    ha='center', va='center')
            # 顶部标签
            ax.text(mid, -2.5, REGION_DISPLAY_NAMES[region][0],
                    fontsize=9, fontweight='bold', color=REGION_COLORS[region],
                    ha='center', va='center')
    
    # 调整显示范围
    ax.set_xlim(-3, 19)
    ax.set_ylim(19, -3)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_patches = [mpatches.Patch(color=REGION_COLORS[r], label=REGION_DISPLAY_NAMES[r]) 
                      for r in ['frontal', 'temporal', 'central', 'parietal', 'occipital']]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.15, 1.0),
              title='Brain Regions', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_circular_connectivity(
    matrix: np.ndarray,
    title: str,
    save_path: str,
    threshold: float = 0.3,
    line_cmap: str = 'Reds',
    max_lines: int = 100,
    is_directed: bool = False
):
    """
    绘制圆形脑网络连接图
    
    Args:
        matrix: (19, 19) 连接矩阵
        title: 图标题
        save_path: 保存路径
        threshold: 连接强度阈值（低于此值不显示）
        line_cmap: 连接线颜色映射
        max_lines: 最大显示连接数
        is_directed: 是否为有向连接（如果True，使用箭头显示方向）
    """
    n_channels = 19
    
    # 获取按脑区排序的通道
    ordered_indices, ordered_names, region_bounds = get_region_order_indices()
    
    # 按脑区重排矩阵
    reordered_matrix = reorder_matrix_by_region(matrix)
    
    # 创建图形（有向图使用普通坐标，更容易绘制箭头）
    if is_directed:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
    else:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # 设置背景
    ax.set_facecolor('white')
    
    # 计算节点角度（均匀分布在圆周上）
    # 每个脑区之间留一点间隙
    total_channels = 19
    gap_angle = np.pi / 36  # 脑区间隙
    total_gap = 5 * gap_angle  # 5个脑区，5个间隙
    available_angle = 2 * np.pi - total_gap
    
    angles = []
    current_angle = np.pi / 2  # 从顶部开始
    
    for region in ['frontal', 'temporal', 'central', 'parietal', 'occipital']:
        n_ch = len(REGION_DEFINITIONS[region])
        region_angle = available_angle * (n_ch / total_channels)
        
        for i in range(n_ch):
            angle = current_angle - (i + 0.5) * region_angle / n_ch
            angles.append(angle)
        
        current_angle -= region_angle + gap_angle
    
    angles = np.array(angles)
    
    # 有向图使用笛卡尔坐标
    node_r = 1.0
    if is_directed:
        node_x = node_r * np.cos(angles)
        node_y = node_r * np.sin(angles)
    
    # 绘制脑区弧线
    for region, (start, end) in region_bounds.items():
        color = REGION_COLORS[region]
        n_ch = end - start
        
        # 弧线角度范围
        arc_start = angles[start] + 0.05
        arc_end = angles[end - 1] - 0.05
        
        # 绘制弧线
        arc_angles = np.linspace(arc_start, arc_end, 50)
        arc_r = np.ones(50) * 1.05
        
        if is_directed:
            arc_x = arc_r * np.cos(arc_angles)
            arc_y = arc_r * np.sin(arc_angles)
            ax.plot(arc_x, arc_y, color=color, linewidth=8, solid_capstyle='round')
        else:
            ax.plot(arc_angles, arc_r, color=color, linewidth=8, solid_capstyle='round')
    
    # 绘制节点
    for i, (angle, name) in enumerate(zip(angles, ordered_names)):
        # 确定节点颜色
        for region, (start, end) in region_bounds.items():
            if start <= i < end:
                color = REGION_COLORS[region]
                break
        
        # 绘制节点
        if is_directed:
            ax.scatter(node_x[i], node_y[i], s=100, c=color, edgecolors='white', linewidth=1, zorder=5)
            # 添加标签
            label_r = 1.15
            label_x = label_r * np.cos(angle)
            label_y = label_r * np.sin(angle)
            ax.text(label_x, label_y, name, fontsize=7, ha='center', va='center', fontweight='bold')
        else:
            ax.scatter(angle, node_r, s=100, c=color, edgecolors='white', linewidth=1, zorder=5)
            # 添加标签
            label_r = 1.15
            ax.text(angle, label_r, name, fontsize=7, ha='center', va='center',
                    rotation=0, fontweight='bold')
    
    # 收集所有连接并排序
    connections = []
    if is_directed:
        # 有向：遍历整个矩阵（i -> j）
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j and reordered_matrix[i, j] > threshold:
                    connections.append((i, j, reordered_matrix[i, j]))
    else:
        # 无向：只遍历上三角
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if reordered_matrix[i, j] > threshold:
                    connections.append((i, j, reordered_matrix[i, j]))
    
    # 按强度排序并限制数量
    connections.sort(key=lambda x: x[2], reverse=True)
    connections = connections[:max_lines]
    
    # 绘制连接线
    if connections:
        # 归一化连接强度用于颜色映射
        strengths = [c[2] for c in connections]
        norm = Normalize(vmin=min(strengths), vmax=max(strengths))
        colormap = cm.get_cmap(line_cmap)
        
        for i, j, strength in connections:
            # 线宽和颜色根据强度变化
            linewidth = 0.5 + 2 * norm(strength)
            color = colormap(norm(strength))
            
            if is_directed:
                # 有向连接：使用箭头，i→j和j→i使用不同方向的曲线
                angle_i, angle_j = angles[i], angles[j]
                x1, y1 = node_x[i], node_y[i]
                x2, y2 = node_x[j], node_y[j]
                
                # 计算连接方向的法向量（用于侧向偏移）
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 1e-6:
                    # 单位法向量（垂直于连接方向）
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0, 0
                
                # 侧向偏移量（让i→j和j→i分开显示）
                offset = 0.08  # 偏移距离
                
                # 计算中点
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 控制点：向圆心方向弯曲 + 侧向偏移
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                curve_factor = 0.3 + 0.2 * (1 - distance / 2)
                
                # 侧向偏移（总是偏向一侧，这样反向连接会偏向另一侧）
                ctrl_x = mid_x * curve_factor + nx * offset
                ctrl_y = mid_y * curve_factor + ny * offset
                
                # 使用贝塞尔曲线
                n_points = 30
                t = np.linspace(0, 1, n_points)
                # 二次贝塞尔曲线
                curve_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                curve_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                
                ax.plot(curve_x, curve_y, color=color, linewidth=linewidth, alpha=0.7)
                
                # 在终点附近添加箭头
                arrow_pos = 0.85  # 箭头位置（0-1）
                arrow_idx = int(arrow_pos * (n_points - 1))
                
                ax.annotate('', 
                    xy=(curve_x[arrow_idx + 2], curve_y[arrow_idx + 2]),
                    xytext=(curve_x[arrow_idx], curve_y[arrow_idx]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth, 
                                   mutation_scale=8 + 4 * norm(strength)))
            else:
                # 无向连接：普通曲线
                angle_i, angle_j = angles[i], angles[j]
                
                # 使用贝塞尔曲线风格的连接
                n_points = 50
                t = np.linspace(0, 1, n_points)
                
                # 简单的内凹曲线
                r_mid = 0.3 + 0.4 * (1 - abs(angle_i - angle_j) / np.pi)
                r = node_r * (1 - 4 * t * (1 - t) * (1 - r_mid))
                theta = angle_i + t * (angle_j - angle_i)
                
                ax.plot(theta, r, color=color, linewidth=linewidth, alpha=0.7)
    
    # 设置坐标属性
    if is_directed:
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
    
    # 标题
    direction_label = " (Directed)" if is_directed else ""
    ax.set_title(f"{title}{direction_label}", fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_patches = [mpatches.Patch(color=REGION_COLORS[r], label=REGION_DISPLAY_NAMES[r]) 
                      for r in ['frontal', 'temporal', 'central', 'parietal', 'occipital']]
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.0, 0.5),
               title='Brain Regions', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_all_connectivity(
    connectivity_dict: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = '',
    # 过滤参数
    threshold: float = None,
    percentile: float = 70.0,
    top_k: int = None,
    top_k_per_node: bool = False,
    zscore_threshold: float = 3.0,
    remove_outliers: bool = True,
    exclude_diagonal: bool = True,
    exclude_neighbor_distance: int = None,
    # 可视化参数
    circular_threshold: float = None,
    max_lines: int = 100,
    line_cmap: str = 'Reds'
):
    """
    可视化所有连接性特征（支持多种过滤方法）
    
    Args:
        connectivity_dict: 连接性矩阵字典
        output_dir: 输出目录
        prefix: 文件名前缀
        
        # 过滤参数（应用于矩阵热力图）
        threshold: 绝对阈值，低于此值的连接设为0
        percentile: 百分位阈值（0-100），低于此百分位的连接设为0
        top_k: 保留的top-K最强连接数
        top_k_per_node: top-K是否按节点计算
        zscore_threshold: z-score阈值，用于过滤异常值
        remove_outliers: True-移除离群值，False-只保留显著连接
        exclude_diagonal: 是否排除对角线（自连接）
        exclude_neighbor_distance: 排除相邻通道距离
        
        # 可视化参数
        circular_threshold: 圆形图的绝对阈值（如果为None，使用百分位阈值过滤后的结果）
        max_lines: 圆形图最大显示连接数
        line_cmap: 圆形图连接线颜色映射
        
    过滤策略说明：
        1. 绝对阈值 (threshold): 适用于已知连接强度范围的情况
        2. 百分位阈值 (percentile): 适用于保留最强的N%连接
        3. Top-K: 适用于只想显示固定数量的最强连接
        4. Z-score: 适用于过滤统计离群值
        5. 排除相邻通道: 减少体积传导的影响
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    feature_names = {
        'plv': 'Phase Locking Value (PLV)',
        'wpli': 'weighted Phase Lag Index (wPLI)',
        'granger_causality': 'Granger Causality',
        'transfer_entropy': 'Transfer Entropy',
        'pearson_corr': 'Pearson Correlation',
        'aec': 'Amplitude Envelope Correlation (AEC)'
    }
    
    for feature_key, matrix in connectivity_dict.items():
        feature_title = feature_names.get(feature_key, feature_key)
        file_prefix = f"{prefix}_{feature_key}" if prefix else feature_key
        
        # 检测是否为有向连接特征
        is_directed = feature_key in ['granger_causality', 'transfer_entropy']
        
        # 对于相关性矩阵，使用绝对值
        if feature_key in ['pearson_corr']:
            work_matrix = np.abs(matrix)
        else:
            work_matrix = matrix.copy()
        
        # 应用过滤器
        filtered_matrix = apply_connection_filters(
            work_matrix,
            threshold=threshold,
            percentile=percentile,
            top_k=top_k,
            top_k_per_node=top_k_per_node,
            zscore_threshold=zscore_threshold,
            remove_outliers=remove_outliers,
            exclude_diagonal=exclude_diagonal,
            exclude_neighbor_distance=exclude_neighbor_distance
        )
        
        # 矩阵热力图（使用过滤后的矩阵）
        # 有向连接显示完整矩阵，无向连接显示对称矩阵
        matrix_path = output_path / f"{file_prefix}_matrix.png"
        plot_connectivity_matrix(
            filtered_matrix,
            title=f"{feature_title}",
            save_path=str(matrix_path)
        )
        
        # 圆形连接图
        circular_path = output_path / f"{file_prefix}_circular.png"
        
        # 圆形图使用过滤后的矩阵，有向图显示箭头
        plot_circular_connectivity(
            filtered_matrix,
            title=f"{feature_title}",
            save_path=str(circular_path),
            threshold=circular_threshold if circular_threshold is not None else 0.0,
            max_lines=max_lines,
            line_cmap=line_cmap,
            is_directed=is_directed
        )
    
    print(f"连接性可视化完成: {output_dir}")


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试连接性可视化模块...")
    
    # 生成测试矩阵
    np.random.seed(42)
    test_matrix = np.random.rand(19, 19)
    test_matrix = (test_matrix + test_matrix.T) / 2  # 对称化
    np.fill_diagonal(test_matrix, 1.0)
    
    # 测试矩阵热力图
    print("1. 测试矩阵热力图...")
    plot_connectivity_matrix(
        test_matrix,
        title="Test Connectivity Matrix",
        save_path="test_matrix.png"
    )
    print("   保存: test_matrix.png")
    
    # 测试圆形图
    print("2. 测试圆形连接图...")
    plot_circular_connectivity(
        test_matrix,
        title="Test Circular Connectivity",
        save_path="test_circular.png",
        threshold=0.5
    )
    print("   保存: test_circular.png")
    
    print("\n测试完成!")
