#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本级别评估与可视化工具

用于分析每个样本的预测结果，识别问题样本，生成可视化报告。
支持三种任务类型：channel（通道）、hemi（半球）、onset_zone（脑区）
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import f1_score, precision_score, recall_score

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from dataset import (
    PrivateEEGDataset, create_dataloader, 
    STANDARD_19_CHANNELS, STANDARD_21_CHANNELS,
    BIPOLAR_CHANNEL_NAMES, BIPOLAR_CHANNEL_NAMES_26,
    create_cross_validation_splits
)
from dataset_with_connectivity import MultiModalEEGDataset, create_multimodal_dataloader
from model_wrapper import create_model
from eegnet_model import create_eegnet_model
from multi_branch_model import create_multi_branch_model
from trainer import get_device, set_seed

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# 常量定义
# ==============================================================================

# 脑区名称（新的5脑区定义）
ONSET_ZONE_NAMES = ['left_frontal', 'left_temporal', 'parietal', 'right_frontal', 'right_temporal']

# 半球名称
HEMI_NAMES = ['L', 'R', 'B', 'U']

# 通道名称（默认，会在运行时根据配置更新）
CHANNEL_NAMES = STANDARD_19_CHANNELS


# ==============================================================================
# 样本级别评估函数
# ==============================================================================

def evaluate_samples_detailed(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str = 'channel',
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    对每个样本进行详细评估
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        task_type: 任务类型 ('channel', 'hemi', 'onset_zone')
        threshold: 二值化阈值（多标签任务使用）
    
    Returns:
        包含每个样本详细信息的DataFrame
    """
    model.eval()
    results = []
    
    # 根据任务类型确定标签名称
    if task_type == 'channel':
        label_names = CHANNEL_NAMES
    elif task_type == 'onset_zone':
        label_names = ONSET_ZONE_NAMES
    elif task_type == 'hemi':
        label_names = HEMI_NAMES
    else:
        raise ValueError(f"未知任务类型: {task_type}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch['data'].to(device)
            labels = batch['labels'].to(device)
            pt_ids = batch['pt_id']
            fns = batch['fn']
            sz_idxs = batch['sz_idx']
            
            # 前向传播
            # 检查是否为多模态数据输入（MultiBranchModel需要多个输入）
            if isinstance(batch, dict) and 'connectivity' in batch and 'graph_metrics' in batch:
                eeg_data = batch['eeg_data'].to(device)
                connectivity = batch['connectivity'].to(device)
                graph_metrics = batch['graph_metrics'].to(device)
                outputs = model(eeg_data, connectivity, graph_metrics)
            else:
                outputs = model(data)

            if isinstance(outputs, dict):
                outputs = outputs.get(task_type.split('_')[0], outputs.get('channel'))
            
            # 计算每个样本的损失
            batch_size = data.shape[0]
            for i in range(batch_size):
                sample_output = outputs[i:i+1]
                sample_label = labels[i:i+1]
                
                # 单样本损失
                sample_loss = criterion(sample_output, sample_label).item()
                
                # 预测结果
                if task_type == 'hemi':
                    # 多分类：使用softmax
                    probs = torch.softmax(sample_output, dim=-1).cpu().numpy().flatten()
                    pred_class = np.argmax(probs)
                    true_class = np.argmax(sample_label.cpu().numpy().flatten())
                    preds = np.zeros(len(label_names))
                    preds[pred_class] = 1
                    true_labels = sample_label.cpu().numpy().flatten()
                    
                    # 计算指标
                    is_correct = pred_class == true_class
                    sample_f1 = 1.0 if is_correct else 0.0
                    n_correct = 1 if is_correct else 0
                    n_errors = 1 - n_correct
                else:
                    # 多标签分类：使用sigmoid
                    probs = torch.sigmoid(sample_output).cpu().numpy().flatten()
                    preds = (probs > threshold).astype(int)
                    true_labels = sample_label.cpu().numpy().flatten()
                    
                    # 计算样本级别F1
                    if true_labels.sum() > 0 or preds.sum() > 0:
                        sample_f1 = f1_score(true_labels, preds, zero_division=0)
                    else:
                        sample_f1 = 1.0  # 全0对全0是完美预测
                    
                    # 统计正确和错误
                    n_correct = np.sum(preds == true_labels)
                    n_errors = np.sum(preds != true_labels)
                
                # 逐标签错误详情
                error_details = []
                for j, name in enumerate(label_names):
                    if preds[j] != true_labels[j]:
                        if preds[j] == 1 and true_labels[j] == 0:
                            error_details.append(f"{name}:FP")
                        else:
                            error_details.append(f"{name}:FN")
                
                # 记录结果
                result = {
                    'sample_idx': batch_idx * data_loader.batch_size + i,
                    'pt_id': pt_ids[i],
                    'fn': fns[i],
                    'sz_idx': int(sz_idxs[i]) if isinstance(sz_idxs[i], (int, np.integer)) else sz_idxs[i].item(),
                    'sample_loss': sample_loss,
                    'sample_f1': sample_f1,
                    'n_correct': n_correct,
                    'n_errors': n_errors,
                    'true_labels': ','.join(map(str, true_labels.astype(int))),
                    'pred_labels': ','.join(map(str, preds.astype(int))),
                    'pred_probs': ','.join([f'{p:.4f}' for p in probs]),
                    'error_details': ';'.join(error_details) if error_details else '',
                }
                results.append(result)
    
    df = pd.DataFrame(results)
    return df


def get_poor_samples(
    df: pd.DataFrame,
    sort_by: str = 'sample_loss',
    top_n: int = None,
    loss_threshold: float = None,
    f1_threshold: float = None
) -> pd.DataFrame:
    """
    筛选和排序问题样本
    
    Args:
        df: 样本评估结果DataFrame
        sort_by: 排序字段 ('sample_loss', 'n_errors', 'sample_f1')
        top_n: 返回前N个最差样本
        loss_threshold: 损失值阈值（大于此值的样本）
        f1_threshold: F1阈值（小于此值的样本）
    
    Returns:
        问题样本DataFrame
    """
    poor_df = df.copy()
    
    # 应用筛选条件
    if loss_threshold is not None:
        poor_df = poor_df[poor_df['sample_loss'] > loss_threshold]
    
    if f1_threshold is not None:
        poor_df = poor_df[poor_df['sample_f1'] < f1_threshold]
    
    # 排序
    if sort_by == 'sample_loss':
        poor_df = poor_df.sort_values('sample_loss', ascending=False)
    elif sort_by == 'n_errors':
        poor_df = poor_df.sort_values('n_errors', ascending=False)
    elif sort_by == 'sample_f1':
        poor_df = poor_df.sort_values('sample_f1', ascending=True)
    
    # 截取前N个
    if top_n is not None:
        poor_df = poor_df.head(top_n)
    
    return poor_df.reset_index(drop=True)


# ==============================================================================
# 可视化函数
# ==============================================================================

def plot_prediction_heatmap(
    df: pd.DataFrame,
    output_path: str,
    task_type: str = 'channel',
    max_samples: int = 50,
    title: str = None
):
    """
    绘制预测结果热力图
    
    Args:
        df: 样本评估结果DataFrame
        output_path: 输出图片路径
        task_type: 任务类型
        max_samples: 最大显示样本数
        title: 图表标题
    """
    # 确定标签名称
    if task_type == 'channel':
        label_names = CHANNEL_NAMES
    elif task_type == 'onset_zone':
        label_names = ONSET_ZONE_NAMES
    elif task_type == 'hemi':
        label_names = HEMI_NAMES
    else:
        label_names = [f'L{i}' for i in range(len(df['true_labels'].iloc[0].split(',')))]
    
    # 限制样本数量
    plot_df = df.head(max_samples)
    n_samples = len(plot_df)
    n_labels = len(label_names)
    
    # 解析标签
    true_matrix = np.zeros((n_samples, n_labels))
    pred_matrix = np.zeros((n_samples, n_labels))
    
    for i, row in plot_df.iterrows():
        true_labels = list(map(int, row['true_labels'].split(',')))
        pred_labels = list(map(int, row['pred_labels'].split(',')))
        true_matrix[i] = true_labels
        pred_matrix[i] = pred_labels
    
    # 创建误差矩阵：0=正确, 1=FP(误报), -1=FN(漏报)
    error_matrix = np.zeros((n_samples, n_labels))
    for i in range(n_samples):
        for j in range(n_labels):
            if true_matrix[i, j] == pred_matrix[i, j]:
                error_matrix[i, j] = 0  # 正确
            elif pred_matrix[i, j] == 1:
                error_matrix[i, j] = 1  # 误报 (FP)
            else:
                error_matrix[i, j] = -1  # 漏报 (FN)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(max(12, n_labels * 0.6), max(8, n_samples * 0.3)))
    
    # 自定义颜色：绿色=正确, 橙色=FP, 红色=FN
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#FF6B6B', '#90EE90', '#FFA500'])  # FN, 正确, FP
    
    im = ax.imshow(error_matrix + 1, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # 设置刻度
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_samples))
    
    # Y轴标签：样本信息
    y_labels = [f"{row['pt_id']}_SZ{row['sz_idx']}" for _, row in plot_df.iterrows()]
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # 标题
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'样本预测结果热力图 ({task_type})', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('标签', fontsize=10)
    ax.set_ylabel('样本', fontsize=10)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='正确预测'),
        Patch(facecolor='#FFA500', label='误报 (FP)'),
        Patch(facecolor='#FF6B6B', label='漏报 (FN)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"热力图已保存: {output_path}")


def plot_error_distribution(
    df: pd.DataFrame,
    output_path: str,
    task_type: str = 'channel'
):
    """
    绘制误差分布统计图
    
    Args:
        df: 样本评估结果DataFrame
        output_path: 输出图片路径
        task_type: 任务类型
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 损失值分布
    ax1 = axes[0, 0]
    ax1.hist(df['sample_loss'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(df['sample_loss'].mean(), color='red', linestyle='--', label=f"均值: {df['sample_loss'].mean():.4f}")
    ax1.axvline(df['sample_loss'].median(), color='orange', linestyle='--', label=f"中位数: {df['sample_loss'].median():.4f}")
    ax1.set_xlabel('样本损失值', fontsize=10)
    ax1.set_ylabel('样本数量', fontsize=10)
    ax1.set_title('损失值分布', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # 2. F1分数分布
    ax2 = axes[0, 1]
    ax2.hist(df['sample_f1'], bins=20, color='seagreen', edgecolor='white', alpha=0.8)
    ax2.axvline(df['sample_f1'].mean(), color='red', linestyle='--', label=f"均值: {df['sample_f1'].mean():.4f}")
    ax2.set_xlabel('样本F1分数', fontsize=10)
    ax2.set_ylabel('样本数量', fontsize=10)
    ax2.set_title('F1分数分布', fontsize=11, fontweight='bold')
    ax2.legend()
    
    # 3. 错误数量分布
    ax3 = axes[1, 0]
    error_counts = df['n_errors'].value_counts().sort_index()
    ax3.bar(error_counts.index, error_counts.values, color='coral', edgecolor='white', alpha=0.8)
    ax3.set_xlabel('错误数量', fontsize=10)
    ax3.set_ylabel('样本数量', fontsize=10)
    ax3.set_title('每个样本的错误数量分布', fontsize=11, fontweight='bold')
    
    # 4. 各标签错误统计
    ax4 = axes[1, 1]
    
    # 确定标签名称
    if task_type == 'channel':
        label_names = CHANNEL_NAMES
    elif task_type == 'onset_zone':
        label_names = ONSET_ZONE_NAMES
    elif task_type == 'hemi':
        label_names = HEMI_NAMES
    else:
        n_labels = len(df['true_labels'].iloc[0].split(','))
        label_names = [f'L{i}' for i in range(n_labels)]
    
    # 统计每个标签的FP和FN
    fp_counts = {name: 0 for name in label_names}
    fn_counts = {name: 0 for name in label_names}
    
    for _, row in df.iterrows():
        if row['error_details']:
            for error in row['error_details'].split(';'):
                if ':' in error:
                    label, error_type = error.split(':')
                    if error_type == 'FP' and label in fp_counts:
                        fp_counts[label] += 1
                    elif error_type == 'FN' and label in fn_counts:
                        fn_counts[label] += 1
    
    x = np.arange(len(label_names))
    width = 0.35
    
    ax4.bar(x - width/2, [fp_counts[n] for n in label_names], width, label='误报 (FP)', color='#FFA500', alpha=0.8)
    ax4.bar(x + width/2, [fn_counts[n] for n in label_names], width, label='漏报 (FN)', color='#FF6B6B', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
    ax4.set_xlabel('标签', fontsize=10)
    ax4.set_ylabel('错误次数', fontsize=10)
    ax4.set_title('各标签错误统计', fontsize=11, fontweight='bold')
    ax4.legend()
    
    plt.suptitle(f'样本预测误差分析 ({task_type})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"误差分布图已保存: {output_path}")


def plot_sample_detail(
    row: pd.Series,
    output_path: str,
    task_type: str = 'channel'
):
    """
    绘制单个样本的详细预测图
    
    Args:
        row: 样本行数据
        output_path: 输出图片路径
        task_type: 任务类型
    """
    # 确定标签名称
    if task_type == 'channel':
        label_names = CHANNEL_NAMES
    elif task_type == 'onset_zone':
        label_names = ONSET_ZONE_NAMES
    elif task_type == 'hemi':
        label_names = HEMI_NAMES
    else:
        n_labels = len(row['true_labels'].split(','))
        label_names = [f'L{i}' for i in range(n_labels)]
    
    # 解析数据
    true_labels = np.array(list(map(int, row['true_labels'].split(','))))
    pred_labels = np.array(list(map(int, row['pred_labels'].split(','))))
    probs = np.array(list(map(float, row['pred_probs'].split(','))))
    
    n_labels = len(label_names)
    
    fig, ax = plt.subplots(figsize=(max(12, n_labels * 0.6), 6))
    
    x = np.arange(n_labels)
    width = 0.6
    
    # 绘制预测概率柱状图
    colors = []
    for i in range(n_labels):
        if true_labels[i] == pred_labels[i]:
            colors.append('#90EE90')  # 正确 - 绿色
        elif pred_labels[i] == 1:
            colors.append('#FFA500')  # FP - 橙色
        else:
            colors.append('#FF6B6B')  # FN - 红色
    
    bars = ax.bar(x, probs, width, color=colors, edgecolor='black', alpha=0.8)
    
    # 添加阈值线
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='阈值 (0.5)')
    
    # 标记真实标签位置
    for i in range(n_labels):
        if true_labels[i] == 1:
            ax.scatter(i, -0.05, marker='^', s=100, c='blue', zorder=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('标签', fontsize=10)
    ax.set_ylabel('预测概率', fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    
    title = f"样本: {row['pt_id']}_SZ{row['sz_idx']} | Loss: {row['sample_loss']:.4f} | F1: {row['sample_f1']:.4f}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#90EE90', edgecolor='black', label='正确预测'),
        Patch(facecolor='#FFA500', edgecolor='black', label='误报 (FP)'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='漏报 (FN)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='真实正标签'),
        Line2D([0], [0], color='gray', linestyle='--', label='阈值 (0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_visualization_report(
    df: pd.DataFrame,
    output_dir: str,
    task_type: str = 'channel',
    top_n_poor: int = 20
):
    """
    生成完整的可视化报告
    
    Args:
        df: 样本评估结果DataFrame
        output_dir: 输出目录
        task_type: 任务类型
        top_n_poor: 问题样本数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存完整预测结果
    df.to_csv(output_path / 'sample_predictions.csv', index=False, encoding='utf-8-sig')
    logger.info(f"样本预测结果已保存: {output_path / 'sample_predictions.csv'}")
    
    # 2. 获取并保存问题样本
    poor_df = get_poor_samples(df, sort_by='sample_loss', top_n=top_n_poor)
    poor_df.to_csv(output_path / 'poor_samples_report.csv', index=False, encoding='utf-8-sig')
    logger.info(f"问题样本报告已保存: {output_path / 'poor_samples_report.csv'}")
    
    # 3. 生成统计摘要
    summary = {
        'total_samples': len(df),
        'mean_loss': float(df['sample_loss'].mean()),
        'median_loss': float(df['sample_loss'].median()),
        'mean_f1': float(df['sample_f1'].mean()),
        'perfect_samples': int((df['n_errors'] == 0).sum()),
        'error_rate': float((df['n_errors'] > 0).sum() / len(df)),
        'top_10_worst_samples': poor_df.head(10)[['pt_id', 'fn', 'sz_idx', 'sample_loss', 'sample_f1', 'n_errors']].to_dict('records')
    }
    
    with open(output_path / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"评估摘要已保存: {output_path / 'evaluation_summary.json'}")
    
    # 4. 生成可视化
    vis_dir = output_path / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 总体误差分布图
    plot_error_distribution(df, str(vis_dir / 'error_distribution.png'), task_type)
    
    # 问题样本热力图
    if len(poor_df) > 0:
        plot_prediction_heatmap(
            poor_df,
            str(vis_dir / 'poor_samples_heatmap.png'),
            task_type,
            max_samples=min(30, len(poor_df)),
            title=f'问题样本预测热力图 (Top {min(30, len(poor_df))})'
        )
        
        # 为Top 10问题样本生成详细图
        sample_dir = vis_dir / 'sample_details'
        sample_dir.mkdir(exist_ok=True)
        
        for i, (_, row) in enumerate(poor_df.head(10).iterrows()):
            plot_sample_detail(
                row,
                str(sample_dir / f'sample_{i+1}_{row["pt_id"]}_SZ{row["sz_idx"]}.png'),
                task_type
            )
    
    logger.info(f"可视化报告生成完成: {output_dir}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("样本级别评估摘要")
    print("="*60)
    print(f"总样本数: {summary['total_samples']}")
    print(f"平均损失: {summary['mean_loss']:.4f}")
    print(f"平均F1: {summary['mean_f1']:.4f}")
    print(f"完美预测样本: {summary['perfect_samples']} ({summary['perfect_samples']/summary['total_samples']*100:.1f}%)")
    print(f"有错误样本: {summary['total_samples'] - summary['perfect_samples']} ({summary['error_rate']*100:.1f}%)")
    print("\nTop 5 问题样本:")
    for i, sample in enumerate(summary['top_10_worst_samples'][:5]):
        print(f"  {i+1}. {sample['fn']}_segment_{sample['sz_idx']}: Loss={sample['sample_loss']:.4f}, F1={sample['sample_f1']:.4f}, Errors={sample['n_errors']}")
    print("="*60)


# ==============================================================================
# 评估主函数
# ==============================================================================

def run_evaluation(
    checkpoint_path: str,
    task_type: str = 'channel',
    model_type: str = 'stgnn',  # 'stgnn' 或 'eegnet'
    output_dir: str = None,
    manifest_path: str = None,
    data_roots: List[str] = None,
    fold: int = None,
    n_folds: int = 5,
    batch_size: int = 4,
    device: str = 'cuda',
    seed: int = 42,
    # Multi-branch specific args
    fusion_type: str = 'attention',
    segment_length: float = 20.0
):
    """
    运行样本级别评估
    
    Args:
        checkpoint_path: 模型检查点路径
        task_type: 任务类型
        model_type: 模型架构类型 ('stgnn' 或 'eegnet')
        output_dir: 输出目录
        manifest_path: manifest文件路径
        data_roots: 数据根目录列表
        fold: 使用哪个fold的验证集（None表示使用全部数据）
        n_folds: 交叉验证折数
        batch_size: 批大小
        device: 设备
        seed: 随机种子
    """
    set_seed(seed)
    
    # 加载配置
    config = get_config()
    
    if manifest_path:
        config.data.manifest_path = manifest_path
    if data_roots:
        config.data.edf_data_roots = data_roots
    
    # 根据config中的use_bipolar设置正确的通道数
    use_21_channels = getattr(config.data, 'use_21_channels', False)
    global CHANNEL_NAMES
    
    if config.data.use_bipolar:
        if use_21_channels:
            config.model.n_channels = 26
            CHANNEL_NAMES = BIPOLAR_CHANNEL_NAMES_26
            logger.info("使用TCP双极导联模式（21电极），通道数: 26")
        else:
            config.model.n_channels = 18
            CHANNEL_NAMES = BIPOLAR_CHANNEL_NAMES
            logger.info("使用TCP双极导联模式（19电极），通道数: 18")
    else:
        if use_21_channels:
            config.model.n_channels = 21
            CHANNEL_NAMES = STANDARD_21_CHANNELS
            logger.info("使用单极导联模式（21电极），通道数: 21")
        else:
            config.model.n_channels = 19
            CHANNEL_NAMES = STANDARD_19_CHANNELS
            logger.info("使用单极导联模式（19电极），通道数: 19")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = str(Path(checkpoint_path).parent / 'evaluation_results')
    
    # 设置设备
    device = get_device(device == 'cuda')
    logger.info(f"使用设备: {device}")
    
    # 确定验证集患者
    patient_ids = None
    if fold is not None:
        splits = create_cross_validation_splits(
            config.data.manifest_path,
            n_folds=n_folds,
            seed=seed
        )
        _, patient_ids = splits[fold]
        logger.info(f"使用 Fold {fold} 的验证集，共 {len(patient_ids)} 个患者")
    
    # 创建数据集
    if model_type == 'multi_branch':
        logger.info("创建多模态数据集用于Multi-Branch模型评估...")
        # 确定连接性类型 (这里默认使用训练时的全集，或者可以通过args传入)
        connectivity_types = ['plv', 'wpli', 'aec', 'pearson', 'granger', 'transfer_entropy']
        
        dataset = MultiModalEEGDataset(
            manifest_path=config.data.manifest_path,
            data_roots=config.data.edf_data_roots,
            label_type=task_type,
            patient_ids=patient_ids,
            config=config.data,
            segment_length=segment_length,
            segment_overlap=0.0, # 评估时不重叠
            connectivity_types=connectivity_types,
            compute_online=True,
            include_directed=True
        )
        
        data_loader = create_multimodal_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 # 评估时使用单线程避免问题
        )
    else:
        dataset = PrivateEEGDataset(
            manifest_path=config.data.manifest_path,
            data_roots=config.data.edf_data_roots,
            label_type=task_type,
            patient_ids=patient_ids,
            config=config.data
        )
        
        data_loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # 评估时使用单线程避免问题
        )
    
    logger.info(f"数据集大小: {len(dataset)} 样本")
    
    # 根据模型类型创建模型
    if model_type.lower() == 'eegnet':
        # EEGNet模型配置
        model_config = {
            'n_channels': config.model.n_channels,
            'n_samples': config.model.time_steps,
            'n_windows': config.data.n_windows,
            'n_classes': 5 if task_type == 'onset_zone' else (4 if task_type == 'hemi' else config.model.n_channels),
            'dropout': 0.5,
            'F1': 8,
            'D': 2,
            'F2': 16,
            'kernel_length': 64,
            'temporal_aggregation': 'attention'
        }
        model = create_eegnet_model(task_type, model_config)
        logger.info("使用 EEGNet 模型架构")
    elif model_type == 'multi_branch':
        # Detect dimensions from first batch
        sample_batch = next(iter(data_loader))
        n_windows = sample_batch['eeg_data'].shape[1]
        n_samples = sample_batch['eeg_data'].shape[3]
        n_connectivity_types = sample_batch['connectivity'].shape[1]
        n_graph_features = sample_batch['graph_metrics'].shape[2]
        
        logger.info(f"检测到输入维度: n_windows={n_windows}, n_samples={n_samples}")

        model_config = {
            'n_channels': config.model.n_channels,
            'n_samples': n_samples,
            'n_windows': n_windows,
            'n_classes': 5 if task_type in ['onset_zone', 'region_5'] else (4 if task_type == 'hemi' else config.model.n_channels),
            'n_connectivity_types': n_connectivity_types,
            'n_graph_features': n_graph_features,
            'fusion_type': fusion_type,
            'dropout': 0.5
        }
        model = create_multi_branch_model(model_config)
        logger.info(f"使用 Multi-Branch Fusion 模型架构 (Fusion: {fusion_type})")
    else:
        # STGNN模型配置（默认）
        model_config = {
            'n_channels': config.model.n_channels,
            'n_bands': config.model.n_bands,
            'time_steps': config.model.time_steps,
            'n_windows': config.data.n_windows,
            'temporal_hidden_dim': 32,
            'graph_hidden_dim': 32,
            'dropout': 0.6
        }
        model = create_model(task_type, model_config)
        logger.info("使用 STGNN 模型架构")
    
    model = model.to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"模型已加载: {checkpoint_path}")
    
    # 创建损失函数
    if task_type == 'hemi':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 评估
    logger.info("开始样本级别评估...")
    df = evaluate_samples_detailed(model, data_loader, criterion, device, task_type)
    
    # 生成报告
    generate_visualization_report(df, output_dir, task_type)
    
    return df


# ==============================================================================
# 命令行接口
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='样本级别评估与可视化')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--task-type', type=str, default='onset_zone',
                        choices=['channel', 'hemi', 'onset_zone', 'region_5'],
                        help='任务类型')
    parser.add_argument('--model-type', type=str, default='stgnn',
                        choices=['stgnn', 'eegnet', 'multi_branch'],
                        help='模型架构类型')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--manifest', type=str, default=None,
                        help='manifest文件路径')
    parser.add_argument('--data-roots', type=str, nargs='+', default=None,
                        help='数据根目录')
    parser.add_argument('--fold', type=int, default=None,
                        help='验证集fold编号')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--fusion-type', type=str, default='attention',
                        help='Fusion type for multi-branch model')
    parser.add_argument('--segment-length', type=float, default=20.0,
                        help='Segment length for multi-branch model')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        task_type=args.task_type,
        model_type=args.model_type,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        data_roots=args.data_roots,
        fold=args.fold,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        fusion_type=args.fusion_type,
        segment_length=args.segment_length
    )


if __name__ == '__main__':
    main()
