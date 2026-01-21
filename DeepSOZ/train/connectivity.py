#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脑网络连接性特征计算模块

支持的连接性指标：
1. PLV (Phase Locking Value) - 相位锁定值
2. wPLI (weighted Phase Lag Index) - 加权相位滞后指数
3. Granger Causality - 格兰杰因果
4. Transfer Entropy - 传递熵
5. Pearson Correlation - 皮尔逊相关
6. AEC (Amplitude Envelope Correlation) - 振幅包络相关
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional
import warnings

# 尝试导入可选依赖
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

DS_FLAG = False # 是否降采样(只对有向信号时有效)


def bandpass_filter_signal(data: np.ndarray, fs: float, 
                           low: float = 1.0, high: float = 45.0) -> np.ndarray:
    """带通滤波"""
    nyq = fs / 2
    low_norm = max(low / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    b, a = butter(4, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_plv(data: np.ndarray, fs: float = 200.0,
                freq_band: Tuple[float, float] = (8, 13)) -> np.ndarray:
    """
    计算相位锁定值 (Phase Locking Value)
    
    PLV衡量两个信号在特定频段的相位同步性
    
    Args:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        freq_band: 频带范围 (low, high)
    
    Returns:
        plv_matrix: (n_channels, n_channels) PLV矩阵
    """
    n_channels, n_samples = data.shape
    
    # 带通滤波到目标频段
    filtered = bandpass_filter_signal(data, fs, freq_band[0], freq_band[1])
    
    # 希尔伯特变换获取解析信号
    analytic = hilbert(filtered, axis=-1)
    phase = np.angle(analytic)
    
    # 计算PLV矩阵
    plv_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            # PLV = |mean(exp(j * (phase_i - phase_j)))|
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
    
    return plv_matrix


def compute_wpli(data: np.ndarray, fs: float = 200.0,
                 freq_band: Tuple[float, float] = (8, 13)) -> np.ndarray:
    """
    计算加权相位滞后指数 (weighted Phase Lag Index)
    
    wPLI对体积传导不敏感，是更稳健的相位连接性指标
    
    Args:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        freq_band: 频带范围
    
    Returns:
        wpli_matrix: (n_channels, n_channels) wPLI矩阵
    """
    n_channels, n_samples = data.shape
    
    # 带通滤波
    filtered = bandpass_filter_signal(data, fs, freq_band[0], freq_band[1])
    
    # 希尔伯特变换
    analytic = hilbert(filtered, axis=-1)
    
    # 计算wPLI矩阵
    wpli_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # 计算交叉谱
            cross_spectrum = analytic[i] * np.conj(analytic[j])
            imag_part = np.imag(cross_spectrum)
            
            # wPLI = |mean(|imag| * sign(imag))| / mean(|imag|)
            abs_imag = np.abs(imag_part)
            if np.sum(abs_imag) > 1e-10:
                wpli = np.abs(np.sum(abs_imag * np.sign(imag_part))) / np.sum(abs_imag)
            else:
                wpli = 0.0
            
            wpli_matrix[i, j] = wpli
            wpli_matrix[j, i] = wpli
    
    # 对角线设为1
    np.fill_diagonal(wpli_matrix, 1.0)
    
    return wpli_matrix


def compute_granger_causality(data: np.ndarray, fs: float = 250.0,
                              max_lag: int = 5) -> np.ndarray:
    """
    计算格兰杰因果 (Granger Causality)

    Args:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        max_lag: 最大滞后阶数 (建议不要太大，防止过拟合)

    Returns:
        gc_matrix: (n_channels, n_channels) GC矩阵
                   gc_matrix[i, j] 代表从 j 到 i 的连接强度 (Source j -> Target i)
    """
    if not HAS_STATSMODELS:
        warnings.warn("statsmodels未安装，Granger因果返回零矩阵")
        return np.zeros((data.shape[0], data.shape[0]))

    n_channels, n_samples = data.shape
    gc_matrix = np.zeros((n_channels, n_channels))

    # 1. 预处理：一阶差分，确保平稳性 (关键!)
    # axis=1 代表在时间维度上做差分
    data_diff = np.diff(data, axis=1)

    # 2. 降采样 (可选，取决于原始fs。如果是250Hz，降采样到125Hz或60Hz左右计算GC比较稳健)
    # 你的逻辑: max(1, int(fs / 50)) -> 如果fs=250, factor=5 -> fs_new=50Hz。这有点低，但可以接受。
    # 建议保留更多细节，例如降到 100Hz
    target_fs = 100
    downsample_factor = max(1, int(fs / target_fs)) if DS_FLAG else 1
    data_ds = data_diff[:, ::downsample_factor]

    # 打印一下形状以便调试
    # print(f"Processing GC with shape: {data_ds.shape}")

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue

            try:
                # 准备数据: [Target, Source] -> 检验 Source(j) -> Target(i)
                # 转置以符合 (n_samples, 2)
                test_data = np.column_stack([data_ds[i], data_ds[j]])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # verbose=False 非常重要，否则控制台会被刷屏
                    result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                # 策略调整：通常取 max_lag 对应的 F 值，或者所有 lag 中 F 值最大的那个
                # 这里我们取所有 lag 中最大的 F-statistic
                max_f_stat = 0.0

                for lag in range(1, max_lag + 1):
                    # result 结构: {lag: ({test_stats}, [res_target, res_source, ...])}
                    # index 0 是 test_stats 字典
                    if lag in result:
                        # 'ssr_ftest' 是 (F-stat, p-value, df_denom, df_num)
                        f_stat = result[lag][0]['ssr_ftest'][0]
                        if f_stat > max_f_stat:
                            max_f_stat = f_stat

                # 直接保存 F 值，不做除以10的操作
                gc_matrix[i, j] = max_f_stat

            except Exception as e:
                # print(f"Error at {i}, {j}: {e}")
                gc_matrix[i, j] = 0.0

    # 3. (可选) 全局归一化
    # 如果你一定要归一化到 0-1 便于画图，请在矩阵计算完后，除以矩阵的最大值
    if np.max(gc_matrix) > 0:
        gc_matrix = gc_matrix / np.max(gc_matrix)

    return gc_matrix


def compute_transfer_entropy(data: np.ndarray, fs: float = 200.0,
                              n_bins: int = 8, lag: int = 1) -> np.ndarray:
    """
    计算传递熵 (Transfer Entropy)
    
    TE是有向信息论指标，衡量信息从一个信号传递到另一个信号的量
    
    Args:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        n_bins: 直方图分箱数
        lag: 时间滞后
    
    Returns:
        te_matrix: (n_channels, n_channels) TE矩阵 (有向)
    """
    n_channels, n_samples = data.shape
    te_matrix = np.zeros((n_channels, n_channels))
    
    # 数据离散化
    def discretize(x, n_bins):
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x, percentiles)
        bins[0] -= 1e-10
        bins[-1] += 1e-10
        return np.digitize(x, bins[1:-1])
    
    # 降采样
    downsample_factor = max(1, int(fs / 50)) if DS_FLAG else 1
    data_ds = data[:, ::downsample_factor]
    n_samples_ds = data_ds.shape[1]
    
    # 离散化所有通道
    data_disc = np.array([discretize(data_ds[i], n_bins) for i in range(n_channels)])
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            
            try:
                # TE(j->i) = H(i_future | i_past) - H(i_future | i_past, j_past)
                x_past = data_disc[i, :-lag]
                x_future = data_disc[i, lag:]
                y_past = data_disc[j, :-lag]
                
                # 简化计算：使用条件熵的估计
                # 联合分布
                n = len(x_future)
                
                # H(X_future | X_past)
                joint_xx = np.zeros((n_bins, n_bins))
                for t in range(n):
                    joint_xx[x_past[t], x_future[t]] += 1
                joint_xx /= n
                
                marginal_x = joint_xx.sum(axis=1)
                h_cond_1 = 0.0
                for a in range(n_bins):
                    if marginal_x[a] > 0:
                        for b in range(n_bins):
                            if joint_xx[a, b] > 0:
                                h_cond_1 -= joint_xx[a, b] * np.log2(joint_xx[a, b] / marginal_x[a])
                
                # H(X_future | X_past, Y_past) - 简化为联合熵估计
                joint_xxy = np.zeros((n_bins, n_bins, n_bins))
                for t in range(n):
                    joint_xxy[x_past[t], y_past[t], x_future[t]] += 1
                joint_xxy /= n
                
                marginal_xy = joint_xxy.sum(axis=2)
                h_cond_2 = 0.0
                for a in range(n_bins):
                    for b in range(n_bins):
                        if marginal_xy[a, b] > 0:
                            for c in range(n_bins):
                                if joint_xxy[a, b, c] > 0:
                                    h_cond_2 -= joint_xxy[a, b, c] * np.log2(
                                        joint_xxy[a, b, c] / marginal_xy[a, b])
                
                te = max(0, h_cond_1 - h_cond_2)
                te_matrix[i, j] = min(1.0, te / 2.0)  # 归一化
                
            except Exception:
                te_matrix[i, j] = 0.0
    
    return te_matrix


def compute_pearson_corr(data: np.ndarray) -> np.ndarray:
    """
    计算皮尔逊相关系数
    
    Args:
        data: (n_channels, n_samples) EEG数据
    
    Returns:
        corr_matrix: (n_channels, n_channels) 相关矩阵
    """
    return np.corrcoef(data)


def compute_aec(data: np.ndarray, fs: float = 200.0,
                freq_band: Tuple[float, float] = (8, 13)) -> np.ndarray:
    """
    计算振幅包络相关 (Amplitude Envelope Correlation)
    
    AEC衡量两个信号振幅包络的相关性
    
    Args:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        freq_band: 频带范围
    
    Returns:
        aec_matrix: (n_channels, n_channels) AEC矩阵
    """
    n_channels, n_samples = data.shape
    
    # 带通滤波
    filtered = bandpass_filter_signal(data, fs, freq_band[0], freq_band[1])
    
    # 希尔伯特变换获取振幅包络
    analytic = hilbert(filtered, axis=-1)
    envelope = np.abs(analytic)
    
    # 计算包络的相关系数
    aec_matrix = np.corrcoef(envelope)
    
    # 取绝对值（AEC通常取绝对值）
    aec_matrix = np.abs(aec_matrix)
    
    return aec_matrix


def compute_all_connectivity(data: np.ndarray, fs: float = 200.0,
                              freq_band: Tuple[float, float] = (8, 13),
                              include_directed: bool = True) -> Dict[str, np.ndarray]:
    """
    计算所有连接性特征
    
    Args:
        data: (n_channels, n_samples) EEG数据
        fs: 采样率
        freq_band: 频带范围
        include_directed: 是否包含有向指标（Granger, TE）
    
    Returns:
        connectivity_dict: 包含所有连接性矩阵的字典
    """
    result = {}
    
    # 无向连接性指标
    result['plv'] = compute_plv(data, fs, freq_band)
    result['wpli'] = compute_wpli(data, fs, freq_band)
    result['pearson_corr'] = compute_pearson_corr(data)
    result['aec'] = compute_aec(data, fs, freq_band)
    
    # 有向连接性指标（计算较慢）
    if include_directed:
        result['granger_causality'] = compute_granger_causality(data, fs)
        result['transfer_entropy'] = compute_transfer_entropy(data, fs)
    
    return result


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == '__main__':
    print("测试连接性计算模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n_channels = 19
    n_samples = 1000
    fs = 200.0
    
    # 模拟数据
    data = np.random.randn(n_channels, n_samples)
    
    # 添加一些相关性
    data[1] = 0.8 * data[0] + 0.2 * np.random.randn(n_samples)
    data[2] = 0.6 * data[0] + 0.4 * np.random.randn(n_samples)
    
    print(f"数据形状: {data.shape}")
    
    # 测试各个指标
    print("\n1. PLV:")
    plv = compute_plv(data, fs)
    print(f"   形状: {plv.shape}, 范围: [{plv.min():.3f}, {plv.max():.3f}]")
    
    print("\n2. wPLI:")
    wpli = compute_wpli(data, fs)
    print(f"   形状: {wpli.shape}, 范围: [{wpli.min():.3f}, {wpli.max():.3f}]")
    
    print("\n3. Pearson Correlation:")
    corr = compute_pearson_corr(data)
    print(f"   形状: {corr.shape}, 范围: [{corr.min():.3f}, {corr.max():.3f}]")
    
    print("\n4. AEC:")
    aec = compute_aec(data, fs)
    print(f"   形状: {aec.shape}, 范围: [{aec.min():.3f}, {aec.max():.3f}]")
    
    print("\n5. Granger Causality:")
    gc = compute_granger_causality(data, fs)
    print(f"   形状: {gc.shape}, 范围: [{gc.min():.3f}, {gc.max():.3f}]")
    
    print("\n6. Transfer Entropy:")
    te = compute_transfer_entropy(data, fs)
    print(f"   形状: {te.shape}, 范围: [{te.min():.3f}, {te.max():.3f}]")
    
    print("\n测试完成!")
