#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOZExplainer: 可解释性分析工具

对 LaBraM-TimeFilter-SOZ 模型的 SOZ 预测进行可视化和归因分析。

功能:
    1. 依赖图可视化
       - 过滤后邻接矩阵热力图 (440×440)，按过滤器着色
       - 时空热力图: 边权重投影回 19通道×20补丁 网格
    2. 关键补丁识别 (Integrated Gradients)
       - 计算每个补丁对 SOZ 预测的梯度贡献
       - 输出 onset 附近时间窗的贡献峰值
    3. 频段贡献分析
       - δ/θ/α/β/γ 五频段消融实验
       - 各频段对 SOZ 概率的贡献度
    4. HTML/PDF 报告生成
       - 包含热力图、时序贡献曲线、频段分析三联图

Usage:
    from TUSZ.models.explainer import SOZExplainer

    explainer = SOZExplainer(model, device='cuda')

    # 单窗口分析
    report = explainer.analyze(X, y_true=y_soz, onset_time=5.0)

    # 生成HTML报告
    explainer.generate_report(report, 'output/patient_001_sz0.html')
"""

from __future__ import annotations

import base64
import io
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---- 可视化库 (延迟导入) ----
_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
except ImportError:
    _HAS_MPL = False

try:
    from .labram_timefilter_soz import (
        LaBraM_TimeFilter_SOZ,
        ModelConfig,
        STANDARD_19,
        TCP_PAIRS,
        N_TCP,
        _ELECTRODE_3D,
    )
except ImportError:
    from labram_timefilter_soz import (
        LaBraM_TimeFilter_SOZ,
        ModelConfig,
        STANDARD_19,
        TCP_PAIRS,
        N_TCP,
        _ELECTRODE_3D,
    )


# =============================================================================
# 配置
# =============================================================================

# TCP通道名
TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]

# 频段定义 (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 250.0),
}

# 过滤器配色
FILTER_COLORS = {
    'temporal':     '#2196F3',   # 蓝 (时序)
    'spatial':      '#4CAF50',   # 绿 (空间)
    'pathological': '#F44336',   # 红 (病理)
}


@dataclass
class ExplainerConfig:
    """可解释性分析配置"""
    # Integrated Gradients
    ig_steps: int = 50                        # IG积分步数
    ig_internal_batch_size: int = 10          # 内部批次大小

    # 图可视化
    top_edge_pct: float = 0.05                # 显示Top-5%权重的边
    adj_figsize: Tuple[int, int] = (14, 5)    # 邻接矩阵图尺寸

    # 时间参数
    fs: float = 200.0                         # 采样率
    patch_len: int = 100                      # 补丁长度 (采样点)
    n_patches: int = 20                       # 补丁数
    n_channels: int = 22                      # TCP通道数
    onset_window_pre: float = 0.5             # onset前0.5s
    onset_window_post: float = 1.0            # onset后1.0s

    # 频段分析
    filter_order: int = 4                     # 带通滤波器阶数

    # 报告
    report_figsize: Tuple[int, int] = (18, 24)
    dpi: int = 150


# =============================================================================
# 1. 依赖图可视化
# =============================================================================

class GraphVisualizer:
    """
    邻接矩阵 & 时空热力图可视化
    """

    def __init__(self, cfg: ExplainerConfig):
        self.cfg = cfg

    # -----------------------------------------------------------------
    # 1a. 分解过滤后的邻接矩阵 (per-filter)
    # -----------------------------------------------------------------

    @staticmethod
    def decompose_adj_by_filter(
        model: LaBraM_TimeFilter_SOZ,
        h: torch.Tensor,
        gamma_energy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        对特征 h 分别通过三个过滤器, 返回per-filter的邻接矩阵

        Args:
            model: 模型
            h: [1, N, D]  backbone输出特征 (单样本)
            gamma_energy: optional

        Returns:
            dict: { 'knn_raw': adj, 'temporal': adj_t, 'spatial': adj_s,
                     'pathological': adj_p, 'fused': adj_fused }
        """
        tf = model.timefilter

        with torch.no_grad():
            adj_raw = tf.distance_knn(h)           # [1, N, N]
            adj_t = tf.filter_temporal(adj_raw)     # [1, N, N]
            adj_s = tf.filter_spatial(adj_raw)      # [1, N, N]
            adj_p = tf.filter_pathological(adj_raw, h, gamma_energy)  # [1, N, N]
            adj_fused = tf(h, gamma_energy)         # [1, N, N]

        return {
            'knn_raw': adj_raw[0].cpu().numpy(),
            'temporal': adj_t[0].cpu().numpy(),
            'spatial': adj_s[0].cpu().numpy(),
            'pathological': adj_p[0].cpu().numpy(),
            'fused': adj_fused[0].cpu().numpy(),
        }

    # -----------------------------------------------------------------
    # 1b. 邻接矩阵可视化 (Top-5%边高亮, 按filter着色)
    # -----------------------------------------------------------------

    def plot_adjacency(
        self,
        adjs: Dict[str, np.ndarray],
        top_pct: float = None,
    ) -> plt.Figure:
        """
        绘制过滤后邻接矩阵:
            - 左: 全局440×440热力图 (fused)
            - 中: Top-5%边按过滤器着色 (overlay)
            - 右: 通道级22×22热力图 (对每通道所有patch取max)

        Returns:
            matplotlib Figure
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for visualization")

        top_pct = top_pct or self.cfg.top_edge_pct
        adj_fused = adjs['fused']
        N = adj_fused.shape[0]
        C = self.cfg.n_channels
        P = self.cfg.n_patches

        fig, axes = plt.subplots(1, 3, figsize=self.cfg.adj_figsize)

        # ---- 左: fused 邻接热力图 ----
        ax = axes[0]
        im = ax.imshow(adj_fused, cmap='hot', interpolation='nearest', aspect='auto')
        ax.set_title('Fused Adjacency (440x440)', fontsize=10)
        ax.set_xlabel('Node j')
        ax.set_ylabel('Node i')
        # 通道分隔线
        for ch in range(1, C):
            ax.axhline(ch * P - 0.5, color='white', linewidth=0.3, alpha=0.5)
            ax.axvline(ch * P - 0.5, color='white', linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046)

        # ---- 中: Top-5%边按过滤器类型着色 ----
        ax = axes[1]
        # 找top edges阈值
        vals = adj_fused[adj_fused > 0]
        if len(vals) == 0:
            ax.set_title('No edges (Top-5%)')
        else:
            threshold = np.percentile(vals, 100 * (1 - top_pct))
            # 创建RGB图像
            color_img = np.ones((N, N, 3))  # 白底
            for name, color_hex in FILTER_COLORS.items():
                adj_f = adjs.get(name, np.zeros((N, N)))
                mask_top = (adj_f > 0) & (adj_fused >= threshold)
                if mask_top.any():
                    rgb = mcolors.hex2color(color_hex)
                    for c_idx in range(3):
                        color_img[:, :, c_idx][mask_top] = rgb[c_idx]

            ax.imshow(color_img, interpolation='nearest', aspect='auto')
            ax.set_title(f'Top-{top_pct*100:.0f}% Edges by Filter', fontsize=10)
            # 图例
            from matplotlib.patches import Patch
            legend_patches = [
                Patch(color=FILTER_COLORS['temporal'], label='Temporal'),
                Patch(color=FILTER_COLORS['spatial'], label='Spatial'),
                Patch(color=FILTER_COLORS['pathological'], label='Pathological'),
            ]
            ax.legend(handles=legend_patches, loc='upper right', fontsize=7)

        # ---- 右: 通道级22×22 ----
        ax = axes[2]
        ch_adj = np.zeros((C, C))
        for ci in range(C):
            for cj in range(C):
                block = adj_fused[ci*P:(ci+1)*P, cj*P:(cj+1)*P]
                ch_adj[ci, cj] = block.max()

        im = ax.imshow(ch_adj, cmap='YlOrRd', interpolation='nearest', aspect='equal')
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(TCP_NAMES, rotation=90, fontsize=5)
        ax.set_yticklabels(TCP_NAMES, fontsize=5)
        ax.set_title('Channel-Level Adjacency (22x22)', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 1c. 时空热力图: 边权重投影回 通道×补丁 网格
    # -----------------------------------------------------------------

    def plot_spatiotemporal_heatmap(
        self,
        adj_fused: np.ndarray,
        monopolar_probs: np.ndarray = None,
        y_true: np.ndarray = None,
        temporal_attn: np.ndarray = None,
        onset_patch: int = None,
    ) -> plt.Figure:
        """
        时空热力图: 将邻接矩阵投影到 通道(22或19) × 补丁(20) 的网格

        显示 "哪些通道在何时被激活"

        投影方式: 每个节点(ch, patch)的度(连接强度) = sum_j adj[node, j]
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib")

        C = self.cfg.n_channels
        P = self.cfg.n_patches

        # 节点度: [N] → [C, P]
        node_degree = adj_fused.sum(axis=1)  # [N]
        heatmap = node_degree.reshape(C, P)  # [22, 20]

        n_plots = 1
        has_attn = temporal_attn is not None
        has_probs = monopolar_probs is not None
        if has_attn:
            n_plots += 1
        if has_probs:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        idx = 0
        # --- 节点度热力图 ---
        ax = axes[idx]
        im = ax.imshow(heatmap, cmap='inferno', aspect='auto', interpolation='bilinear')
        ax.set_yticks(range(C))
        ax.set_yticklabels(TCP_NAMES, fontsize=6)
        time_ticks = np.linspace(0, P - 1, 5)
        time_labels = [f'{t * self.cfg.patch_len / self.cfg.fs:.1f}s' for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels)
        ax.set_xlabel('Time')
        ax.set_ylabel('TCP Channel')
        ax.set_title('Spatiotemporal Activation (Node Degree)', fontsize=11)
        if onset_patch is not None:
            ax.axvline(onset_patch, color='lime', linewidth=2, linestyle='--',
                       label=f'Onset (patch {onset_patch})')
            ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03)
        idx += 1

        # --- 时序注意力权重 ---
        if has_attn:
            ax = axes[idx]
            im = ax.imshow(temporal_attn, cmap='viridis', aspect='auto', interpolation='bilinear')
            ax.set_yticks(range(C))
            ax.set_yticklabels(TCP_NAMES, fontsize=6)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels(time_labels)
            ax.set_xlabel('Time')
            ax.set_title('Temporal Attention Weights', fontsize=11)
            if onset_patch is not None:
                ax.axvline(onset_patch, color='lime', linewidth=2, linestyle='--')
            fig.colorbar(im, ax=ax, fraction=0.03)
            idx += 1

        # --- SOZ概率条形图 ---
        if has_probs:
            ax = axes[idx]
            ch_names = list(STANDARD_19)[:len(monopolar_probs)]
            colors = ['#F44336' if p > 0.5 else '#2196F3' for p in monopolar_probs]
            bars = ax.barh(range(len(monopolar_probs)), monopolar_probs, color=colors)
            if y_true is not None:
                for i, yt in enumerate(y_true):
                    if yt > 0.5:
                        ax.barh(i, monopolar_probs[i], color='#FF9800', edgecolor='red',
                                linewidth=2)
            ax.set_yticks(range(len(ch_names)))
            ax.set_yticklabels(ch_names, fontsize=7)
            ax.set_xlabel('SOZ Probability')
            ax.set_title('Monopolar SOZ Prediction', fontsize=11)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            idx += 1

        fig.tight_layout()
        return fig


# =============================================================================
# 2. Integrated Gradients (关键补丁识别)
# =============================================================================

class IntegratedGradientsExplainer:
    """
    Integrated Gradients (IG) 对每个补丁节点的归因

    IG(x) = (x - x_baseline) * ∫₀¹ ∂F/∂x(baseline + α·(x - baseline)) dα

    baseline: 零输入 (全零信号)
    """

    def __init__(self, cfg: ExplainerConfig):
        self.cfg = cfg

    def compute_ig(
        self,
        model: LaBraM_TimeFilter_SOZ,
        X: torch.Tensor,
        target_channels: List[int] = None,
        n_steps: int = None,
        device: torch.device = None,
    ) -> np.ndarray:
        """
        计算 Integrated Gradients

        Args:
            model: SOZ模型
            X: [1, 22, 20, 100]  单个发作窗口输入
            target_channels: 目标通道索引列表 (None=对所有19通道求和)
            n_steps: 积分步数
            device: 计算设备

        Returns:
            attributions: [22, 20]  每个补丁的归因值
        """
        n_steps = n_steps or self.cfg.ig_steps
        device = device or X.device

        model.eval()
        X = X.to(device)

        # 基线: 零输入
        baseline = torch.zeros_like(X)

        # 差值
        delta = X - baseline

        # 沿路径积分
        total_grads = torch.zeros(1, 22, 20, 100, device=device)

        batch_size = self.cfg.ig_internal_batch_size
        alphas = torch.linspace(0, 1, n_steps + 1, device=device)

        for start in range(0, n_steps + 1, batch_size):
            end = min(start + batch_size, n_steps + 1)
            alpha_batch = alphas[start:end]
            B_ig = len(alpha_batch)

            # 插值输入: [B_ig, 22, 20, 100]
            interpolated = baseline + alpha_batch.view(B_ig, 1, 1, 1) * delta
            interpolated.requires_grad_(True)

            outputs = model(interpolated)
            logits = outputs['monopolar_logits']  # [B_ig, 19]

            # 选择目标通道
            if target_channels is not None:
                target_score = logits[:, target_channels].sum(dim=-1)
            else:
                target_score = logits.sum(dim=-1)

            target_score = target_score.sum()  # 标量

            model.zero_grad()
            target_score.backward()

            grad = interpolated.grad  # [B_ig, 22, 20, 100]
            total_grads += grad.sum(dim=0, keepdim=True)

        # IG = delta * avg_grad
        avg_grads = total_grads / (n_steps + 1)
        ig = (delta * avg_grads)  # [1, 22, 20, 100]

        # 对采样点维度取绝对值的均值 → [22, 20]
        attributions = ig.abs().mean(dim=-1).squeeze(0).detach().cpu().numpy()

        return attributions

    def find_onset_peaks(
        self,
        attributions: np.ndarray,
        onset_time: float,
    ) -> Dict:
        """
        在 onset 附近查找贡献峰值

        Args:
            attributions: [22, 20]
            onset_time: 发作起始时间 (秒, 相对于窗口起始)

        Returns:
            dict with peak info
        """
        C, P = attributions.shape
        patch_dur = self.cfg.patch_len / self.cfg.fs  # 每个补丁的时长 (秒)
        onset_patch = int(onset_time / patch_dur)
        onset_patch = np.clip(onset_patch, 0, P - 1)

        pre_patches = int(self.cfg.onset_window_pre / patch_dur)
        post_patches = int(self.cfg.onset_window_post / patch_dur)

        start_p = max(0, onset_patch - pre_patches)
        end_p = min(P, onset_patch + post_patches + 1)

        # 提取onset窗口
        onset_window = attributions[:, start_p:end_p]  # [C, window_len]

        # 每通道峰值
        channel_peaks = {}
        for ch in range(C):
            peak_patch = start_p + np.argmax(onset_window[ch])
            peak_value = onset_window[ch].max()
            peak_time = peak_patch * patch_dur
            channel_peaks[TCP_NAMES[ch]] = {
                'peak_patch': int(peak_patch),
                'peak_time': float(peak_time),
                'peak_value': float(peak_value),
                'mean_value': float(onset_window[ch].mean()),
            }

        # Top通道 (按onset窗口内最大贡献排序)
        ch_max = onset_window.max(axis=1)  # [C]
        top_channels = np.argsort(ch_max)[::-1]

        return {
            'onset_patch': onset_patch,
            'onset_time': onset_time,
            'window_range': (start_p, end_p),
            'channel_peaks': channel_peaks,
            'top_channels': [TCP_NAMES[c] for c in top_channels[:5]],
            'top_values': [float(ch_max[c]) for c in top_channels[:5]],
        }

    def plot_ig(
        self,
        attributions: np.ndarray,
        onset_info: Dict = None,
    ) -> plt.Figure:
        """绘制 IG 归因热力图 + onset附近的时序贡献曲线"""
        if not _HAS_MPL:
            raise ImportError("matplotlib")

        C, P = attributions.shape

        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={'height_ratios': [2, 1]})

        # --- 上: 热力图 ---
        ax = axes[0]
        im = ax.imshow(attributions, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
        ax.set_yticks(range(C))
        ax.set_yticklabels(TCP_NAMES, fontsize=6)
        patch_dur = self.cfg.patch_len / self.cfg.fs
        time_ticks = np.linspace(0, P - 1, 5)
        time_labels = [f'{t * patch_dur:.1f}s' for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels)
        ax.set_xlabel('Time')
        ax.set_ylabel('TCP Channel')
        ax.set_title('Integrated Gradients Attribution (per-patch)', fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.03)

        if onset_info:
            ax.axvline(onset_info['onset_patch'], color='lime', linewidth=2,
                       linestyle='--', label=f"Onset t={onset_info['onset_time']:.1f}s")
            # 标注onset窗口
            ws, we = onset_info['window_range']
            ax.axvspan(ws, we, alpha=0.15, color='lime')
            ax.legend(fontsize=8)

        # --- 下: Top-5通道的时序贡献曲线 ---
        ax = axes[1]
        if onset_info and 'top_channels' in onset_info:
            top_chs = onset_info['top_channels'][:5]
            colors_list = plt.cm.Set1(np.linspace(0, 1, len(top_chs)))
            time_axis = np.arange(P) * patch_dur
            for idx_ch, ch_name in enumerate(top_chs):
                ch_idx = TCP_NAMES.index(ch_name) if ch_name in TCP_NAMES else 0
                ax.plot(time_axis, attributions[ch_idx], label=ch_name,
                        color=colors_list[idx_ch], linewidth=1.5)
            if onset_info:
                ax.axvline(onset_info['onset_time'], color='black', linestyle='--',
                           linewidth=1.5, label='Onset')
            ax.legend(fontsize=8, loc='upper right')
        else:
            # 所有通道平均
            time_axis = np.arange(P) * patch_dur
            ax.plot(time_axis, attributions.mean(axis=0), color='red', linewidth=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Attribution')
        ax.set_title('Patch-Level Attribution Curves (Top-5 channels)', fontsize=11)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig


# =============================================================================
# 3. 频段贡献分析
# =============================================================================

class FrequencyBandAnalyzer:
    """
    五频段消融实验: 对输入信号做带通滤波, 分别前向传播, 比较SOZ概率变化
    """

    def __init__(self, cfg: ExplainerConfig):
        self.cfg = cfg

    @staticmethod
    def bandpass_filter_signal(
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float = 200.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        带通滤波 (零相位)

        Args:
            signal: (..., n_samples)
            lowcut, highcut: 频率范围
            fs: 采样率
            order: 滤波器阶数
        """
        from scipy.signal import butter, filtfilt

        nyq = fs / 2.0
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)

        if low >= high:
            return np.zeros_like(signal)

        b, a = butter(order, [low, high], btype='band')

        # 处理多维度
        orig_shape = signal.shape
        flat = signal.reshape(-1, signal.shape[-1])
        filtered = np.zeros_like(flat)
        for i in range(len(flat)):
            if len(flat[i]) > 3 * max(len(a), len(b)):
                try:
                    filtered[i] = filtfilt(b, a, flat[i])
                except ValueError:
                    filtered[i] = flat[i]
            else:
                filtered[i] = flat[i]
        return filtered.reshape(orig_shape)

    def analyze_bands(
        self,
        model: LaBraM_TimeFilter_SOZ,
        X: torch.Tensor,
        device: torch.device = None,
    ) -> Dict:
        """
        五频段消融分析

        对每个频段:
          1. 带通滤波输入信号 → 仅保留该频段
          2. 前向传播得到 SOZ 概率
          3. 与原始 SOZ 概率比较

        Returns:
            dict with:
                'original_probs': [19]
                'band_probs': {band: [19]}
                'contributions': {band: float}  (L2距离 / cos相似度)
        """
        device = device or X.device
        model.eval()

        X_np = X.squeeze(0).cpu().numpy()  # [22, 20, 100]
        fs = self.cfg.fs

        # 原始概率
        with torch.no_grad():
            orig_out = model(X.to(device))
            orig_probs = orig_out['monopolar_probs'][0].cpu().numpy()  # [19]

        results = {
            'original_probs': orig_probs.tolist(),
            'band_probs': {},
            'band_contributions': {},
            'band_similarity': {},
        }

        for band_name, (flo, fhi) in FREQ_BANDS.items():
            # 对原始信号做带通滤波 (在 patch 展平后做)
            # X_np: [22, 20, 100] → 拼成 [22, 2000] → 滤波 → 切回 [22, 20, 100]
            C, P, L = X_np.shape
            sig = X_np.reshape(C, P * L)  # [22, 2000]

            # 滤波
            sig_filtered = self.bandpass_filter_signal(sig, flo, fhi, fs=fs,
                                                       order=self.cfg.filter_order)
            X_band = sig_filtered.reshape(C, P, L)
            X_band_t = torch.from_numpy(X_band).float().unsqueeze(0).to(device)

            with torch.no_grad():
                band_out = model(X_band_t)
                band_probs = band_out['monopolar_probs'][0].cpu().numpy()

            results['band_probs'][band_name] = band_probs.tolist()

            # 贡献度: 该频段单独的预测与原始预测的余弦相似度
            cos_sim = float(np.dot(orig_probs, band_probs) / (
                np.linalg.norm(orig_probs) * np.linalg.norm(band_probs) + 1e-8
            ))
            # L2距离 (越近说明这个频段越能恢复原始预测)
            l2_dist = float(np.linalg.norm(orig_probs - band_probs))

            results['band_contributions'][band_name] = cos_sim
            results['band_similarity'][band_name] = {
                'cosine': cos_sim,
                'l2_distance': l2_dist,
                'mean_prob': float(band_probs.mean()),
                'max_prob': float(band_probs.max()),
            }

        return results

    def plot_band_analysis(
        self,
        band_results: Dict,
        y_true: np.ndarray = None,
    ) -> plt.Figure:
        """绘制频段贡献分析图"""
        if not _HAS_MPL:
            raise ImportError("matplotlib")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        band_names = list(FREQ_BANDS.keys())
        orig_probs = np.array(band_results['original_probs'])

        # ---- 左: 各频段的SOZ概率对比 (堆叠条形图) ----
        ax = axes[0]
        ch_names = list(STANDARD_19)[:len(orig_probs)]
        x = np.arange(len(ch_names))
        width = 0.13
        band_colors = ['#3F51B5', '#009688', '#8BC34A', '#FF9800', '#F44336']

        for i, band in enumerate(band_names):
            probs = np.array(band_results['band_probs'][band])
            ax.bar(x + i * width, probs, width, label=band, color=band_colors[i], alpha=0.8)

        ax.bar(x + len(band_names) * width, orig_probs, width, label='Original',
               color='black', alpha=0.6)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_ylabel('SOZ Probability')
        ax.set_title('SOZ Probability by Frequency Band', fontsize=10)
        ax.legend(fontsize=6, loc='upper right')
        if y_true is not None:
            for i, yt in enumerate(y_true):
                if yt > 0.5:
                    ax.axvspan(i - 0.4, i + 0.4 + len(band_names) * width,
                               alpha=0.1, color='red')

        # ---- 中: 频段贡献度 (余弦相似度) ----
        ax = axes[1]
        sims = [band_results['band_contributions'][b] for b in band_names]
        bars = ax.bar(band_names, sims, color=band_colors, alpha=0.85)
        ax.set_ylabel('Cosine Similarity to Original')
        ax.set_title('Band Contribution (Similarity)', fontsize=10)
        ax.set_ylim(0, 1)
        for bar, sim in zip(bars, sims):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{sim:.3f}', ha='center', fontsize=8)

        # ---- 右: 频段L2距离 (越小越接近原始预测) ----
        ax = axes[2]
        l2s = [band_results['band_similarity'][b]['l2_distance'] for b in band_names]
        bars = ax.bar(band_names, l2s, color=band_colors, alpha=0.85)
        ax.set_ylabel('L2 Distance from Original')
        ax.set_title('Band Reconstruction Error', fontsize=10)
        for bar, l2 in zip(bars, l2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{l2:.3f}', ha='center', fontsize=8)

        fig.tight_layout()
        return fig


# =============================================================================
# 4. 报告生成
# =============================================================================

class ReportGenerator:
    """
    生成HTML/PDF可解释性报告
    """

    def __init__(self, cfg: ExplainerConfig):
        self.cfg = cfg

    @staticmethod
    def _fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
        """将 matplotlib Figure 编码为 base64 字符串"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64

    def generate_html(
        self,
        report_data: Dict,
        output_path: str,
    ) -> str:
        """
        生成HTML可解释性报告

        Args:
            report_data: SOZExplainer.analyze() 返回的完整分析结果
            output_path: 输出HTML路径

        Returns:
            HTML文件路径
        """
        figs = report_data.get('figures', {})
        meta = report_data.get('metadata', {})
        onset_info = report_data.get('onset_info', {})
        band_results = report_data.get('band_analysis', {})
        probs = report_data.get('monopolar_probs', [])
        y_true = report_data.get('y_true', None)

        # 图片编码
        img_sections = []

        if 'adjacency' in figs:
            b64 = self._fig_to_base64(figs['adjacency'], self.cfg.dpi)
            img_sections.append(('Graph Adjacency Visualization', b64))

        if 'spatiotemporal' in figs:
            b64 = self._fig_to_base64(figs['spatiotemporal'], self.cfg.dpi)
            img_sections.append(('Spatiotemporal Activation Heatmap', b64))

        if 'ig_attribution' in figs:
            b64 = self._fig_to_base64(figs['ig_attribution'], self.cfg.dpi)
            img_sections.append(('Integrated Gradients Attribution', b64))

        if 'band_analysis' in figs:
            b64 = self._fig_to_base64(figs['band_analysis'], self.cfg.dpi)
            img_sections.append(('Frequency Band Contribution Analysis', b64))

        # SOZ概率表
        prob_table_rows = ''
        ch_names = list(STANDARD_19)[:len(probs)]
        for i, ch in enumerate(ch_names):
            p = probs[i] if i < len(probs) else 0
            is_soz = y_true[i] > 0.5 if y_true is not None and i < len(y_true) else False
            row_style = ' style="background-color: #FFCDD2; font-weight: bold;"' if is_soz else ''
            bar_color = '#F44336' if p > 0.5 else '#2196F3'
            bar_width = max(1, int(p * 100))
            prob_table_rows += f'''
                <tr{row_style}>
                    <td>{ch}</td>
                    <td>{p:.4f}</td>
                    <td>
                        <div style="background:{bar_color}; width:{bar_width}%;
                             height:16px; border-radius:3px;"></div>
                    </td>
                    <td>{'SOZ' if is_soz else ''}</td>
                </tr>'''

        # Onset贡献峰值表
        onset_table = ''
        if onset_info and 'top_channels' in onset_info:
            onset_table = '<h3>Top-5 Contributing Channels (around onset)</h3><table>'
            onset_table += '<tr><th>Rank</th><th>Channel</th><th>Peak Time</th><th>Peak Value</th></tr>'
            for rank, (ch, val) in enumerate(
                zip(onset_info['top_channels'][:5], onset_info['top_values'][:5])
            ):
                peak_info = onset_info['channel_peaks'].get(ch, {})
                t = peak_info.get('peak_time', 0)
                onset_table += f'<tr><td>{rank+1}</td><td>{ch}</td><td>{t:.2f}s</td>'
                onset_table += f'<td>{val:.4f}</td></tr>'
            onset_table += '</table>'

        # 频段贡献表
        band_table = ''
        if band_results and 'band_contributions' in band_results:
            band_table = '<h3>Frequency Band Contributions</h3><table>'
            band_table += '<tr><th>Band</th><th>Freq Range</th><th>Cosine Sim</th>'
            band_table += '<th>L2 Distance</th><th>Mean Prob</th></tr>'
            for band, (flo, fhi) in FREQ_BANDS.items():
                sim = band_results['band_contributions'].get(band, 0)
                detail = band_results['band_similarity'].get(band, {})
                l2 = detail.get('l2_distance', 0)
                mp = detail.get('mean_prob', 0)
                band_table += (
                    f'<tr><td><b>{band}</b></td><td>{flo}-{fhi} Hz</td>'
                    f'<td>{sim:.4f}</td><td>{l2:.4f}</td><td>{mp:.4f}</td></tr>'
                )
            band_table += '</table>'

        # 拼接HTML
        image_html = ''
        for title, b64 in img_sections:
            image_html += f'''
            <div class="section">
                <h2>{title}</h2>
                <img src="data:image/png;base64,{b64}" style="max-width:100%;">
            </div>'''

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>SOZ Explainability Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px 40px;
            background: #FAFAFA;
            color: #333;
        }}
        h1 {{
            color: #1565C0;
            border-bottom: 3px solid #1565C0;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #424242;
            margin-top: 30px;
            border-left: 4px solid #1565C0;
            padding-left: 12px;
        }}
        h3 {{
            color: #616161;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background: #E3F2FD;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #FAFAFA;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .meta-item {{
            background: #E3F2FD;
            padding: 8px 15px;
            border-radius: 5px;
        }}
        img {{
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        .footer {{
            margin-top: 40px;
            padding: 15px;
            text-align: center;
            color: #999;
            font-size: 12px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <h1>SOZ Explainability Report</h1>

    <div class="section">
        <h2>Sample Metadata</h2>
        <div class="meta-grid">
            <div class="meta-item"><b>Patient:</b> {meta.get('patient_id', 'N/A')}</div>
            <div class="meta-item"><b>Source:</b> {meta.get('source', 'N/A')}</div>
            <div class="meta-item"><b>Onset Time:</b> {meta.get('onset_time', 'N/A')}s</div>
            <div class="meta-item"><b>Window:</b> {meta.get('window_info', 'N/A')}</div>
        </div>
    </div>

    <div class="section">
        <h2>SOZ Prediction Summary</h2>
        <table>
            <tr><th>Channel</th><th>Probability</th><th>Visual</th><th>Ground Truth</th></tr>
            {prob_table_rows}
        </table>
    </div>

    {image_html}

    <div class="section">
        {onset_table}
        {band_table}
    </div>

    <div class="footer">
        Generated by SOZExplainer | LaBraM-TimeFilter-SOZ
    </div>
</body>
</html>"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved: {output_path}")
        return output_path

    def generate_pdf(self, html_path: str, pdf_path: str = None) -> Optional[str]:
        """
        从HTML生成PDF (供临床医生审核)

        需要: pip install weasyprint 或 pip install pdfkit
        """
        pdf_path = pdf_path or str(Path(html_path).with_suffix('.pdf'))

        # 尝试 weasyprint
        try:
            from weasyprint import HTML as WeasyprintHTML
            WeasyprintHTML(filename=html_path).write_pdf(pdf_path)
            logger.info(f"PDF report saved (weasyprint): {pdf_path}")
            return pdf_path
        except ImportError:
            pass

        # 尝试 pdfkit (需要 wkhtmltopdf)
        try:
            import pdfkit
            pdfkit.from_file(html_path, pdf_path)
            logger.info(f"PDF report saved (pdfkit): {pdf_path}")
            return pdf_path
        except (ImportError, OSError):
            pass

        # 尝试 matplotlib 的 PdfPages 作为最后手段 (仅保存图片)
        logger.warning(
            "PDF generation requires 'weasyprint' or 'pdfkit'. "
            "Install: pip install weasyprint"
        )
        return None


# =============================================================================
# 5. SOZExplainer 主类
# =============================================================================

class SOZExplainer:
    """
    SOZ可解释性分析工具

    整合: 图可视化 + IG归因 + 频段分析 + 报告生成

    Usage:
        explainer = SOZExplainer(model, device='cuda')
        report = explainer.analyze(X, y_true=y_soz, onset_time=5.0)
        explainer.generate_report(report, 'output/report.html')
        explainer.generate_report(report, 'output/report.html', pdf=True)
    """

    def __init__(
        self,
        model: LaBraM_TimeFilter_SOZ,
        device: str = 'cpu',
        cfg: ExplainerConfig = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.cfg = cfg or ExplainerConfig()

        self.graph_viz = GraphVisualizer(self.cfg)
        self.ig_explainer = IntegratedGradientsExplainer(self.cfg)
        self.band_analyzer = FrequencyBandAnalyzer(self.cfg)
        self.report_gen = ReportGenerator(self.cfg)

    def analyze(
        self,
        X: torch.Tensor,
        y_true: np.ndarray = None,
        onset_time: float = 5.0,
        target_channels: List[int] = None,
        metadata: Dict = None,
        compute_ig: bool = True,
        compute_bands: bool = True,
    ) -> Dict:
        """
        完整可解释性分析

        Args:
            X: [1, 22, 20, 100]  单个发作窗口
            y_true: [19]  真实SOZ标签 (可选)
            onset_time: 发作起始时间 (秒, 相对窗口起始)
            target_channels: IG的目标通道 (None=全部)
            metadata: 样本元数据 (patient_id, source等)
            compute_ig: 是否计算Integrated Gradients
            compute_bands: 是否计算频段分析

        Returns:
            dict with all analysis results and figures
        """
        if not _HAS_MPL:
            raise ImportError(
                "matplotlib is required. Install: pip install matplotlib"
            )

        self.model.eval()
        X = X.to(self.device)
        if X.dim() == 3:
            X = X.unsqueeze(0)

        report = {
            'metadata': metadata or {},
            'figures': {},
        }
        report['metadata']['onset_time'] = onset_time

        # ---- (0) 前向传播: 获取所有中间输出 ----
        with torch.no_grad():
            out = self.model(X)

        monopolar_probs = out['monopolar_probs'][0].cpu().numpy()
        temporal_attn = out['temporal_attn'][0].cpu().numpy()    # [22, 20]
        filtered_adj = out['filtered_adj'][0].cpu().numpy()      # [440, 440]

        report['monopolar_probs'] = monopolar_probs.tolist()
        report['y_true'] = y_true.tolist() if y_true is not None else None

        onset_patch = int(onset_time / (self.cfg.patch_len / self.cfg.fs))
        onset_patch = min(onset_patch, self.cfg.n_patches - 1)

        # ---- (1) 图可视化 ----
        logger.info("Computing graph visualizations...")

        # 获取中间backbone输出用于per-filter分解
        with torch.no_grad():
            h = self.model.patch_embed(X)
            h = self.model.backbone(h)

        adjs = GraphVisualizer.decompose_adj_by_filter(self.model, h)
        report['adjacency_matrices'] = {
            k: v.tolist() for k, v in adjs.items()
            if k in ('knn_raw', 'fused')
        }

        fig_adj = self.graph_viz.plot_adjacency(adjs)
        report['figures']['adjacency'] = fig_adj

        fig_st = self.graph_viz.plot_spatiotemporal_heatmap(
            filtered_adj,
            monopolar_probs=monopolar_probs,
            y_true=y_true,
            temporal_attn=temporal_attn,
            onset_patch=onset_patch,
        )
        report['figures']['spatiotemporal'] = fig_st

        # ---- (2) Integrated Gradients ----
        if compute_ig:
            logger.info("Computing Integrated Gradients...")
            ig_attr = self.ig_explainer.compute_ig(
                self.model, X,
                target_channels=target_channels,
                device=self.device,
            )
            report['ig_attributions'] = ig_attr.tolist()

            onset_info = self.ig_explainer.find_onset_peaks(ig_attr, onset_time)
            report['onset_info'] = onset_info

            fig_ig = self.ig_explainer.plot_ig(ig_attr, onset_info)
            report['figures']['ig_attribution'] = fig_ig
        else:
            report['onset_info'] = {}

        # ---- (3) 频段分析 ----
        if compute_bands:
            logger.info("Computing frequency band analysis...")
            band_results = self.band_analyzer.analyze_bands(
                self.model, X, device=self.device,
            )
            report['band_analysis'] = band_results

            fig_band = self.band_analyzer.plot_band_analysis(band_results, y_true)
            report['figures']['band_analysis'] = fig_band
        else:
            report['band_analysis'] = {}

        logger.info("Analysis complete.")
        return report

    def generate_report(
        self,
        report_data: Dict,
        output_path: str,
        pdf: bool = False,
    ) -> str:
        """
        生成HTML报告 (可选PDF)

        Args:
            report_data: analyze() 返回的结果
            output_path: HTML输出路径
            pdf: 是否同时生成PDF

        Returns:
            HTML文件路径
        """
        html_path = self.report_gen.generate_html(report_data, output_path)

        if pdf:
            pdf_path = str(Path(output_path).with_suffix('.pdf'))
            self.report_gen.generate_pdf(html_path, pdf_path)

        return html_path

    def batch_analyze(
        self,
        dataset,
        indices: List[int] = None,
        output_dir: str = './reports',
        max_samples: int = 10,
        **kwargs,
    ) -> List[str]:
        """
        批量分析并生成报告

        Args:
            dataset: TimeFilterDataset
            indices: 样本索引列表
            output_dir: 输出目录
            max_samples: 最多分析的样本数

        Returns:
            报告文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = list(range(min(len(dataset), max_samples)))
        else:
            indices = indices[:max_samples]

        report_paths = []
        for idx in indices:
            X, y_soz, mask, meta = dataset[idx]
            if isinstance(X, torch.Tensor):
                X = X.unsqueeze(0)
            else:
                X = torch.from_numpy(X).float().unsqueeze(0)

            y_true = y_soz.numpy() if isinstance(y_soz, torch.Tensor) else y_soz

            pid = meta.get('patient_id', f'sample_{idx}')
            report = self.analyze(
                X, y_true=y_true,
                metadata=meta,
                **kwargs,
            )

            html_path = str(output_dir / f'{pid}_idx{idx}.html')
            self.generate_report(report, html_path)
            report_paths.append(html_path)

            logger.info(f"  [{idx+1}/{len(indices)}] {html_path}")

        return report_paths


# =============================================================================
# 自测
# =============================================================================

def _self_test():
    """无数据自测: 验证全部可解释性功能"""
    print("=" * 60)
    print("SOZExplainer Self-Test")
    print("=" * 60)

    cfg = ModelConfig(
        n_transformer_layers=4,
        n_frozen_layers=2,
        embed_dim=64,
        use_domain_adversarial=False,
    )
    model = LaBraM_TimeFilter_SOZ(cfg)
    model.eval()

    X = torch.randn(1, 22, 20, 100)
    y_true = np.zeros(19)
    y_true[[2, 5, 10]] = 1.0

    ex_cfg = ExplainerConfig(ig_steps=5, n_patches=20, n_channels=22)
    explainer = SOZExplainer(model, device='cpu', cfg=ex_cfg)

    # 1. 图可视化
    print("\n[1] Graph visualization...")
    with torch.no_grad():
        h = model.patch_embed(X)
        h = model.backbone(h)
    adjs = GraphVisualizer.decompose_adj_by_filter(model, h)
    for name, adj in adjs.items():
        print(f"  {name}: shape={adj.shape}, non-zero={np.count_nonzero(adj)}")
    fig = explainer.graph_viz.plot_adjacency(adjs)
    print(f"  [OK] adjacency figure created")
    plt.close(fig)

    out = model(X)
    fig_st = explainer.graph_viz.plot_spatiotemporal_heatmap(
        out['filtered_adj'][0].detach().numpy(),
        monopolar_probs=out['monopolar_probs'][0].detach().numpy(),
        y_true=y_true,
        temporal_attn=out['temporal_attn'][0].detach().numpy(),
        onset_patch=10,
    )
    print(f"  [OK] spatiotemporal heatmap created")
    plt.close(fig_st)

    # 2. Integrated Gradients
    print("\n[2] Integrated Gradients...")
    ig_attr = explainer.ig_explainer.compute_ig(model, X, n_steps=5)
    print(f"  IG shape: {ig_attr.shape}, range=[{ig_attr.min():.4f}, {ig_attr.max():.4f}]")
    onset_info = explainer.ig_explainer.find_onset_peaks(ig_attr, onset_time=5.0)
    print(f"  Onset patch: {onset_info['onset_patch']}")
    print(f"  Top channels: {onset_info['top_channels'][:3]}")
    fig_ig = explainer.ig_explainer.plot_ig(ig_attr, onset_info)
    print(f"  [OK] IG figure created")
    plt.close(fig_ig)

    # 3. 频段分析
    print("\n[3] Frequency band analysis...")
    band_results = explainer.band_analyzer.analyze_bands(model, X)
    for band, sim in band_results['band_contributions'].items():
        detail = band_results['band_similarity'][band]
        print(f"  {band:12s}: cos_sim={sim:.4f}, L2={detail['l2_distance']:.4f}")
    fig_band = explainer.band_analyzer.plot_band_analysis(band_results, y_true)
    print(f"  [OK] band analysis figure created")
    plt.close(fig_band)

    # 4. 完整分析 + 报告生成
    print("\n[4] Full analysis + HTML report...")
    report = explainer.analyze(
        X, y_true=y_true, onset_time=5.0,
        metadata={'patient_id': 'TEST001', 'source': 'synthetic'},
    )
    print(f"  SOZ probs (top-3): ", end='')
    probs = np.array(report['monopolar_probs'])
    top3 = np.argsort(probs)[-3:][::-1]
    for i in top3:
        ch = list(STANDARD_19)[i]
        print(f"{ch}={probs[i]:.3f} ", end='')
    print()

    html_path = explainer.generate_report(
        report, './test_runs/explainer_selftest/test_report.html'
    )
    print(f"  HTML: {html_path}")

    # 检查HTML文件
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    assert '<title>SOZ Explainability Report</title>' in html_content
    assert 'Integrated Gradients' in html_content
    assert 'Frequency Band' in html_content
    print(f"  HTML size: {len(html_content):,} bytes")

    # 清理
    import shutil
    if Path('./test_runs/explainer_selftest').exists():
        shutil.rmtree('./test_runs/explainer_selftest')

    print(f"\n[OK] All SOZExplainer tests passed!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    warnings.filterwarnings('ignore', category=UserWarning)
    _self_test()
