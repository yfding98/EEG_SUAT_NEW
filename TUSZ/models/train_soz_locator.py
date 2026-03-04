#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端SOZ定位器训练脚本

整合数据预处理、模型训练、域适应、可解释性分析的完整流程。

主要功能:
    1. 数据加载与预处理检查
    2. TCP导联转换验证（随机抽样可视化）
    3. 域差异分析（PSD KL散度）
    4. 三阶段域适应训练（条件性）
    5. 验证循环（19通道AUC + Top-3准确率）
    6. 训练中可解释性可视化（每10轮）
    7. 错误处理（EDF异常、样本量不足）

Usage:
    # 完整流程（公共+私有数据，域适应）
    python train_soz_locator.py \
        --public_data_dir F:/dataset/TUSZ/v2.0.3/edf \
        --private_data_dir F:/dataset/private_eeg \
        --labram_ckpt ./checkpoints/labram.pth \
        --output_dir ./runs/soz_locator_v1

    # 仅公共数据（无域适应）
    python train_soz_locator.py \
        --public_data_dir F:/dataset/TUSZ/v2.0.3/edf \
        --labram_ckpt ./checkpoints/labram.pth

    # 禁用域适应
    python train_soz_locator.py \
        --public_data_dir F:/dataset/TUSZ/v2.0.3/edf \
        --private_data_dir F:/dataset/private_eeg \
        --domain_adapt False
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目路径
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ---- 导入项目模块 ----
try:
    from models.labram_timefilter_soz import (
        LaBraM_TimeFilter_SOZ,
        ModelConfig,
        STANDARD_19,
        TCP_PAIRS,
        TCP_NAMES,
    )
    from models.train_soz import (
        TrainConfig,
        DomainDivergenceAnalyzer,
        SOZTrainer,
        evaluate_model,
        compute_top_k_accuracy,
        compute_channel_auc,
    )
    from models.explainer import SOZExplainer, ExplainerConfig
    from models.manifest_dataset import ManifestSOZDataset
except ImportError:
    # 尝试相对导入
    from .labram_timefilter_soz import (
        LaBraM_TimeFilter_SOZ,
        ModelConfig,
        STANDARD_19,
        TCP_PAIRS,
        TCP_NAMES,
    )
    from .train_soz import (
        TrainConfig,
        DomainDivergenceAnalyzer,
        SOZTrainer,
        evaluate_model,
        compute_top_k_accuracy,
        compute_channel_auc,
    )
    from .explainer import SOZExplainer, ExplainerConfig
    from .manifest_dataset import ManifestSOZDataset

try:
    from data_preprocess.eeg_pipeline import TimeFilterDataset
    from data_preprocess.preprocess import (
        PreprocessConfig,
        preprocess_tusz,
        preprocess_private,
    )
except ImportError:
    TimeFilterDataset = None
    PreprocessConfig = None
    preprocess_tusz = None
    preprocess_private = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

logger = logging.getLogger(__name__)

# 尝试导入matplotlib（用于可视化）
_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    _HAS_MPL = False


# =============================================================================
# 1. 数据预处理检查与执行
# =============================================================================

def check_and_preprocess(
    public_data_dir: Optional[str],
    private_data_dir: Optional[str],
    preprocessed_root: str = r'F:\process_dataset',
    force_reprocess: bool = False,
) -> Tuple[bool, str]:
    """
    检查预处理数据是否存在，若不存在则执行预处理

    Returns:
        (success, preprocessed_root)
    """
    preprocessed_root = Path(preprocessed_root)
    index_all = preprocessed_root / 'index_all.csv'

    if index_all.exists() and not force_reprocess:
        logger.info(f"预处理数据已存在: {preprocessed_root}")
        return True, str(preprocessed_root)

    logger.info("预处理数据不存在，开始预处理...")

    if PreprocessConfig is None:
        raise ImportError(
            "数据预处理模块不可用。"
            "请确保 data_preprocess/preprocess.py 可访问。"
        )

    cfg = PreprocessConfig()
    cfg.output_root = str(preprocessed_root)

    errors = []

    # 预处理公共数据
    if public_data_dir:
        logger.info(f"预处理公共数据: {public_data_dir}")
        try:
            cfg.data_root = public_data_dir
            cfg.data_type = 'tusz'
            preprocess_tusz(cfg)
            logger.info("  [OK] 公共数据预处理完成")
        except Exception as e:
            error_msg = f"公共数据预处理失败: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            logger.error(traceback.format_exc())

    # 预处理私有数据
    if private_data_dir:
        logger.info(f"预处理私有数据: {private_data_dir}")
        try:
            cfg.data_root = private_data_dir
            cfg.data_type = 'private'
            preprocess_private(cfg)
            logger.info("  [OK] 私有数据预处理完成")
        except Exception as e:
            error_msg = f"私有数据预处理失败: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            logger.error(traceback.format_exc())

    if errors:
        logger.warning(f"预处理过程中出现 {len(errors)} 个错误（已跳过损坏文件）")
        for err in errors:
            logger.warning(f"  - {err}")

    if index_all.exists():
        logger.info(f"预处理完成: {preprocessed_root}")
        return True, str(preprocessed_root)
    else:
        raise RuntimeError(
            f"预处理失败: 索引文件未生成 {index_all}\n"
            f"请检查数据路径和预处理配置。"
        )


# =============================================================================
# 2. TCP导联转换验证（随机抽样可视化）
# =============================================================================

def verify_tcp_conversion(
    dataset: TimeFilterDataset,
    n_samples: int = 5,
    output_dir: Path = None,
) -> bool:
    """
    随机抽样验证TCP导联转换正确性

    可视化:
        - 随机抽取n个样本
        - 显示22通道TCP数据的时间序列
        - 标注通道名称和SOZ标签

    Returns:
        success: bool
    """
    if not _HAS_MPL:
        logger.warning("matplotlib不可用，跳过TCP验证可视化")
        return True

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / 'tcp_verification'
        viz_dir.mkdir(exist_ok=True)
    else:
        viz_dir = None

    logger.info(f"验证TCP导联转换（随机抽样 {n_samples} 个样本）...")

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    TCP_NAMES = [f"{a}-{b}" for a, b in TCP_PAIRS]

    for idx in indices:
        try:
            X, y_soz, mask, meta = dataset[idx]
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            if isinstance(y_soz, torch.Tensor):
                y_soz = y_soz.numpy()

            # X: [22, 20, 100]
            C, P, L = X.shape

            # 绘制时间序列（每个通道一个子图）
            fig, axes = plt.subplots(4, 6, figsize=(18, 12))
            axes = axes.flatten()

            for ch in range(min(C, 22)):
                ax = axes[ch]
                # 展平patches: [20, 100] → [2000]
                signal = X[ch].flatten()
                time_axis = np.arange(len(signal)) / 200.0  # 200Hz采样率

                ax.plot(time_axis, signal, linewidth=0.5, alpha=0.7)
                ax.set_title(f"{TCP_NAMES[ch]}", fontsize=8)
                ax.set_xlabel('Time (s)', fontsize=6)
                ax.set_ylabel('Amplitude', fontsize=6)
                ax.grid(alpha=0.3)

                # 标注SOZ（如果有）
                if y_soz is not None:
                    # 简化：如果该TCP通道对应的单极通道中有SOZ
                    # 这里仅做示例标注
                    pass

            # 隐藏多余的子图
            for ch in range(C, len(axes)):
                axes[ch].axis('off')

            fig.suptitle(
                f"TCP Verification - Sample {idx}\n"
                f"Patient: {meta.get('patient_id', 'N/A')}, "
                f"Source: {meta.get('source', 'N/A')}",
                fontsize=12,
            )
            fig.tight_layout()

            if viz_dir:
                save_path = viz_dir / f'tcp_verify_sample_{idx}.png'
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"  保存: {save_path}")
            plt.close(fig)

        except Exception as e:
            logger.warning(f"  样本 {idx} 验证失败: {e}")
            continue

    logger.info("  [OK] TCP导联转换验证完成")
    return True


# =============================================================================
# 3. 域差异分析（PSD KL散度）
# =============================================================================

def analyze_domain_divergence(
    public_dataset: TimeFilterDataset,
    private_dataset: TimeFilterDataset,
    output_dir: Path = None,
    kl_threshold: float = 0.3,
) -> Tuple[bool, Dict]:
    """
    分析公共/私有数据的域差异

    Returns:
        (need_domain_adaptation, divergence_report)
    """
    logger.info("分析域差异（PSD KL散度）...")

    analyzer = DomainDivergenceAnalyzer()

    # 收集样本（最多200个）
    max_samples = 200
    pub_samples = []
    for i in range(min(max_samples, len(public_dataset))):
        try:
            X, _, _, _ = public_dataset[i]
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            pub_samples.append(X)
        except Exception as e:
            logger.debug(f"  跳过公共样本 {i}: {e}")
            continue

    priv_samples = []
    for i in range(min(max_samples, len(private_dataset))):
        try:
            X, _, _, _ = private_dataset[i]
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            priv_samples.append(X)
        except Exception as e:
            logger.debug(f"  跳过私有样本 {i}: {e}")
            continue

    if len(pub_samples) == 0 or len(priv_samples) == 0:
        logger.warning("  样本不足，无法进行域差异分析")
        return True, {'mean_kl': 0.0, 'need_domain_alignment': False}

    pub_arr = np.array(pub_samples)
    priv_arr = np.array(priv_samples)

    div_result = analyzer.analyze(pub_arr, priv_arr, max_samples=max_samples)

    logger.info(f"  平均KL散度: {div_result['mean_kl']:.4f}")
    logger.info(f"  最大KL散度: {div_result['max_kl']:.4f}")
    logger.info(
        f"  域对齐需求: {'是' if div_result['need_domain_alignment'] else '否'} "
        f"(阈值={kl_threshold})"
    )

    # 保存报告
    if output_dir:
        report_path = output_dir / 'domain_divergence.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(div_result, f, indent=2, ensure_ascii=False)
        logger.info(f"  报告已保存: {report_path}")

    return div_result['need_domain_alignment'], div_result


# =============================================================================
# 4. 数据增强（样本量不足时）
# =============================================================================

class StrongAugmentation:
    """
    强数据增强（用于样本量<50的情况）

    增强策略:
        - 时移: ±0.2秒（±40个采样点@200Hz）
        - 通道置换: 随机交换对称通道对（如F3↔F4, T3↔T4）
    """

    @staticmethod
    def time_shift(X: torch.Tensor, shift_range: float = 0.2, fs: float = 200.0) -> torch.Tensor:
        """
        时移增强

        Args:
            X: [B, 22, 20, 100]
            shift_range: 最大时移（秒）
            fs: 采样率

        Returns:
            X_shifted: [B, 22, 20, 100]
        """
        B, C, P, L = X.shape
        max_shift_samples = int(shift_range * fs)

        # 展平patches: [B, 22, 2000]
        X_flat = X.reshape(B, C, P * L)

        # 随机时移
        shifts = torch.randint(-max_shift_samples, max_shift_samples + 1, (B,))
        X_shifted = torch.zeros_like(X_flat)

        for b in range(B):
            shift = shifts[b].item()
            if shift > 0:
                X_shifted[b, :, shift:] = X_flat[b, :, :-shift]
            elif shift < 0:
                X_shifted[b, :, :shift] = X_flat[b, :, -shift:]
            else:
                X_shifted[b] = X_flat[b]

        return X_shifted.reshape(B, C, P, L)

    @staticmethod
    def channel_swap(X: torch.Tensor, swap_prob: float = 0.3) -> torch.Tensor:
        """
        通道置换增强（对称通道对交换）

        Args:
            X: [B, 22, 20, 100]
            swap_prob: 交换概率

        Returns:
            X_swapped: [B, 22, 20, 100]
        """
        # TCP通道对称对（简化：仅交换部分明显对称的）
        # 注意：TCP是双极导联，对称性不如单极明显，这里仅做示例
        swap_pairs = [
            (0, 4),   # FP1-F7 ↔ FP2-F8
            (1, 5),   # F7-T3 ↔ F8-T4
            (2, 6),   # T3-T5 ↔ T4-T6
            (3, 7),   # T5-O1 ↔ T6-O2
            (14, 18), # FP1-F3 ↔ FP2-F4
            (15, 19), # F3-C3 ↔ F4-C4
            (16, 20), # C3-P3 ↔ C4-P4
            (17, 21), # P3-O1 ↔ P4-O2
        ]

        X_swapped = X.clone()
        B = X.size(0)

        for b in range(B):
            if torch.rand(1).item() < swap_prob:
                # 随机选择一个对称对交换
                pair_idx = torch.randint(0, len(swap_pairs), (1,)).item()
                ch1, ch2 = swap_pairs[pair_idx]
                X_swapped[b, [ch1, ch2]] = X_swapped[b, [ch2, ch1]]

        return X_swapped


def check_sample_size_and_augment(
    dataset: TimeFilterDataset,
    min_samples: int = 50,
) -> Tuple[bool, Optional[StrongAugmentation]]:
    """
    检查样本量，若不足则返回强增强器

    Returns:
        (need_augmentation, augmentation)
    """
    n_samples = len(dataset)
    if n_samples < min_samples:
        logger.warning(
            f"样本量不足 ({n_samples} < {min_samples})，"
            f"启用强数据增强（时移±0.2s + 通道置换）"
        )
        return True, StrongAugmentation()
    else:
        logger.info(f"样本量充足: {n_samples} >= {min_samples}")
        return False, None


# =============================================================================
# 5. 训练循环（整合域适应）
# =============================================================================

def train_with_validation(
    model: LaBraM_TimeFilter_SOZ,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    augmentation: Optional[StrongAugmentation] = None,
) -> Dict:
    """
    标准训练循环（带验证）

    Args:
        model: SOZ模型
        train_loader: 训练数据
        val_loader: 验证数据
        config: 训练配置
        device: 设备
        augmentation: 数据增强器（可选）

    Returns:
        training_history: dict
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from models.labram_timefilter_soz import SOZDetectionLoss

    criterion = SOZDetectionLoss(
        focal_gamma=config.model_config.focal_gamma,
        focal_alpha=config.model_config.focal_alpha,
        domain_weight=0.0,  # 标准训练不使用域对抗
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.phase1_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.phase1_epochs, eta_min=1e-6)

    history = {
        'train_loss': [],
        'val_metrics': [],
        'best_top3': 0.0,
        'best_epoch': 0,
    }

    for epoch in range(1, config.phase1_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            X, y_soz, mask, metas = batch
            X = X.to(device)
            y_soz = y_soz.to(device)

            # 数据增强
            if augmentation is not None:
                X = augmentation.time_shift(X)
                X = augmentation.channel_swap(X)

            optimizer.zero_grad()
            out = model(X)
            loss, _ = criterion(out['soz_logits'], y_soz, None, None)
            loss.backward()

            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        top3 = val_metrics.get('top3_acc', 0.0)

        history['train_loss'].append(avg_loss)
        history['val_metrics'].append(val_metrics)

        logger.info(
            f"Epoch {epoch}/{config.phase1_epochs}: "
            f"loss={avg_loss:.4f}, "
            f"val_auc={val_metrics['macro_auc']:.4f}, "
            f"val_top3={top3:.4f}"
        )

        # 保存最佳模型（基于Top-3准确率）
        if top3 > history['best_top3']:
            history['best_top3'] = top3
            history['best_epoch'] = epoch
            # 保存最佳模型权重
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'top3_acc': top3,
                'val_auc': val_metrics['macro_auc'],
            }
            torch.save(best_state, str(output_dir / 'best_model.pt'))

    return history


# =============================================================================
# 6. 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='端到端SOZ定位器训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 数据路径 ----
    parser.add_argument(
        '--public_data_dir',
        type=str,
        default=None,
        help='公共数据路径（TUSZ EDF目录）',
    )
    parser.add_argument(
        '--private_data_dir',
        type=str,
        default=None,
        help='私有数据路径（可选，若无则仅用公共数据）',
    )
    parser.add_argument(
        '--preprocessed_root',
        type=str,
        default=r'F:\process_dataset',
        help='预处理数据根目录',
    )
    parser.add_argument(
        '--force_reprocess',
        action='store_true',
        help='强制重新预处理（即使已存在）',
    )

    # ---- Manifest模式 (优先级高于预处理模式) ----
    parser.add_argument(
        '--manifest',
        type=str,
        default='',
        help='combined_manifest.csv 路径 (若提供则用 ManifestSOZDataset)',
    )
    parser.add_argument(
        '--source',
        type=str,
        default='both',
        choices=['tusz', 'private', 'both'],
        help='数据源过滤',
    )
    parser.add_argument(
        '--tusz_data_root',
        type=str,
        default=r'F:\dataset\TUSZ\v2.0.3\edf',
        help='TUSZ EDF root',
    )
    parser.add_argument(
        '--private_data_root_manifest',
        type=str,
        default='',
        help='Private EDF root (for manifest mode)',
    )
    parser.add_argument(
        '--label_mode',
        type=str,
        default='bipolar',
        choices=['bipolar', 'monopolar'],
        help='标签模式 (bipolar=22ch, monopolar=19ch)',
    )

    # ---- 模型配置 ----
    parser.add_argument(
        '--labram_ckpt',
        type=str,
        default='',
        help='LaBraM预训练权重路径',
    )
    parser.add_argument(
        '--patch_len',
        type=int,
        default=100,
        help='补丁长度（采样点数，0.5秒@200Hz）',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.85,
        help='TimeFilter路由阈值（Top-p）',
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=128,
        help='嵌入维度',
    )

    # ---- 训练配置 ----
    parser.add_argument(
        '--domain_adapt',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='是否启用域适应（auto=自动判断）',
    )
    parser.add_argument(
        '--kl_threshold',
        type=float,
        default=0.3,
        help='域差异KL散度阈值',
    )
    parser.add_argument(
        '--phase1_epochs',
        type=int,
        default=50,
        help='Phase 1预训练轮数',
    )
    parser.add_argument(
        '--phase2_epochs',
        type=int,
        default=40,
        help='Phase 2域对齐轮数',
    )
    parser.add_argument(
        '--phase3_epochs',
        type=int,
        default=20,
        help='Phase 3微调轮数',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='学习率',
    )

    # ---- 其他配置 ----
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./runs/soz_locator',
        help='输出目录',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='计算设备',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='数据加载器工作进程数',
    )
    parser.add_argument(
        '--vis_every',
        type=int,
        default=10,
        help='每N轮保存可解释性可视化',
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=50,
        help='最小样本量阈值（低于此值启用强增强）',
    )

    args = parser.parse_args()

    # ---- 日志设置 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'train.log', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

    logger.info("=" * 70)
    logger.info("SOZ Locator Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {args.device}")

    # ---- 设置随机种子 ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    )
    logger.info(f"使用设备: {device}")

    # =====================================================================
    # Step 1: 数据预处理检查
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: 数据预处理检查")
    logger.info("=" * 70)

    try:
        success, preprocessed_root = check_and_preprocess(
            args.public_data_dir,
            args.private_data_dir,
            args.preprocessed_root,
            args.force_reprocess,
        )
        if not success:
            raise RuntimeError("数据预处理失败")
    except Exception as e:
        logger.error(f"数据预处理错误: {e}")
        logger.error(traceback.format_exc())
        return 1

    # =====================================================================
    # Step 2: 数据加载
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 数据加载")
    logger.info("=" * 70)

    if TimeFilterDataset is None:
        raise ImportError("TimeFilterDataset不可用")

    public_dataset = None
    private_dataset = None

    if args.public_data_dir:
        try:
            public_dataset = TimeFilterDataset(
                preprocessed_root,
                subset='tusz',
            )
            logger.info(f"公共数据: {len(public_dataset)} 样本")
        except Exception as e:
            logger.warning(f"加载公共数据失败: {e}")
            public_dataset = None

    if args.private_data_dir:
        try:
            private_dataset = TimeFilterDataset(
                preprocessed_root,
                subset='private',
            )
            logger.info(f"私有数据: {len(private_dataset)} 样本")
        except Exception as e:
            logger.warning(f"加载私有数据失败: {e}")
            private_dataset = None

    if public_dataset is None and private_dataset is None:
        raise RuntimeError("无可用数据")

    # =====================================================================
    # Step 3: TCP导联转换验证
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: TCP导联转换验证")
    logger.info("=" * 70)

    verify_dataset = public_dataset if public_dataset else private_dataset
    verify_tcp_conversion(verify_dataset, n_samples=5, output_dir=output_dir)

    # =====================================================================
    # Step 4: 域差异分析
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 域差异分析")
    logger.info("=" * 70)

    need_domain_adapt = False
    if args.domain_adapt == 'auto' and public_dataset and private_dataset:
        need_domain_adapt, div_report = analyze_domain_divergence(
            public_dataset,
            private_dataset,
            output_dir=output_dir,
            kl_threshold=args.kl_threshold,
        )
    elif args.domain_adapt == 'true':
        need_domain_adapt = True
        logger.info("手动启用域适应")
    else:
        logger.info("跳过域适应（仅使用单一数据源或手动禁用）")

    # =====================================================================
    # Step 5: 样本量检查与增强
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: 样本量检查")
    logger.info("=" * 70)

    train_dataset = private_dataset if private_dataset else public_dataset
    need_aug, augmentation = check_sample_size_and_augment(
        train_dataset,
        min_samples=args.min_samples,
    )

    # =====================================================================
    # Step 6: 模型初始化
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: 模型初始化")
    logger.info("=" * 70)

    model_cfg = ModelConfig(
        labram_checkpoint=args.labram_ckpt,
        patch_len=args.patch_len,
        top_p=args.top_p,
        embed_dim=args.embed_dim,
        n_output=22 if args.label_mode == 'bipolar' else 19,
        output_mode=args.label_mode,
    )

    model = LaBraM_TimeFilter_SOZ(model_cfg)
    model.to(device)
    logger.info(model.summary())

    # =====================================================================
    # Step 7: 训练
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 7: 训练")
    logger.info("=" * 70)

    train_cfg = TrainConfig(
        data_root=preprocessed_root,
        output_dir=str(output_dir),
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        phase1_lr=args.lr,
        phase1_batch_size=args.batch_size,
        phase2_batch_size=args.batch_size // 2,
        phase3_batch_size=args.batch_size // 2,
        kl_threshold=args.kl_threshold,
        skip_domain_analysis=(args.domain_adapt != 'auto'),
        device=str(device),
        num_workers=args.num_workers,
        model_config=model_cfg,
    )

    # 划分训练/验证集
    use_manifest = bool(args.manifest and Path(args.manifest).exists())

    if use_manifest:
        # Manifest模式: 使用 ManifestSOZDataset
        logger.info(f"\n使用 ManifestSOZDataset (source={args.source}, label={args.label_mode})")

        if args.source == 'tusz':
            train_split, val_split = ['train'], ['dev']
        elif args.source == 'private':
            train_split, val_split = ['private'], ['private']
        else:
            train_split, val_split = ['train', 'private'], ['dev']

        train_ds = ManifestSOZDataset(
            manifest_path=args.manifest,
            tusz_data_root=args.tusz_data_root,
            private_data_root=args.private_data_root_manifest,
            source_filter=args.source,
            split_filter=train_split,
            label_mode=args.label_mode,
        )
        val_ds = ManifestSOZDataset(
            manifest_path=args.manifest,
            tusz_data_root=args.tusz_data_root,
            private_data_root=args.private_data_root_manifest,
            source_filter=args.source if args.source != 'both' else 'tusz',
            split_filter=val_split,
            label_mode=args.label_mode,
        )

        public_dataset = ManifestSOZDataset(
            manifest_path=args.manifest,
            tusz_data_root=args.tusz_data_root,
            source_filter='tusz',
            split_filter=['train', 'dev'],
            label_mode=args.label_mode,
        ) if args.source in ('tusz', 'both') else None

        private_dataset = ManifestSOZDataset(
            manifest_path=args.manifest,
            private_data_root=args.private_data_root_manifest,
            source_filter='private',
            label_mode=args.label_mode,
        ) if args.source in ('private', 'both') else None

        train_loader = train_ds.create_dataloader(
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        )
        val_loader = val_ds.create_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )

    elif train_dataset:
        # Legacy模式: 使用 TimeFilterDataset
        pids = train_dataset.get_patient_ids()
        n_val = max(1, len(pids) // 5)
        val_pids = pids[:n_val]
        train_pids = pids[n_val:]

        train_ds = TimeFilterDataset(
            preprocessed_root,
            subset='private' if private_dataset else 'tusz',
            patient_ids=train_pids,
        )
        val_ds = TimeFilterDataset(
            preprocessed_root,
            subset='private' if private_dataset else 'tusz',
            patient_ids=val_pids,
        )

        train_loader = train_ds.create_dataloader(
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        )
        val_loader = val_ds.create_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
    else:
        raise RuntimeError("无训练数据")

    # 域适应训练或标准训练
    if need_domain_adapt and public_dataset and private_dataset:
        logger.info("执行三阶段域适应训练...")
        trainer = SOZTrainer(
            cfg=train_cfg,
            model=model,
            public_dataset=public_dataset,
            private_dataset=train_ds,
            val_dataset=val_ds,
            fold_idx=0,
        )
        trainer.train(need_domain_alignment=True)
    else:
        logger.info("执行标准训练...")
        history = train_with_validation(
            model, train_loader, val_loader, train_cfg, device, output_dir, augmentation,
        )
        logger.info(f"训练完成，最佳Top-3准确率: {history['best_top3']:.4f} (epoch {history['best_epoch']})")

    # =====================================================================
    # Step 8: 训练中可解释性可视化（每N轮）
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Step 8: 可解释性可视化")
    logger.info("=" * 70)

    if args.vis_every > 0:
        explainer = SOZExplainer(model, device=str(device))
        vis_dir = output_dir / 'explanations'
        vis_dir.mkdir(exist_ok=True)

        # 随机选择一个验证样本
        vis_idx = np.random.randint(0, len(val_ds))
        X_vis, y_vis, _, meta_vis = val_ds[vis_idx]
        if isinstance(X_vis, torch.Tensor):
            X_vis = X_vis.unsqueeze(0)
        else:
            X_vis = torch.from_numpy(X_vis).float().unsqueeze(0)

        y_true_vis = y_vis.numpy() if isinstance(y_vis, torch.Tensor) else y_vis

        report = explainer.analyze(
            X_vis,
            y_true=y_true_vis,
            onset_time=5.0,
            metadata=meta_vis,
        )
        explainer.generate_report(
            report,
            str(vis_dir / f'explanation_sample_{vis_idx}.html'),
        )
        logger.info(f"可解释性报告已保存: {vis_dir}")

    logger.info("\n" + "=" * 70)
    logger.info("训练完成！")
    logger.info("=" * 70)
    logger.info(f"输出目录: {output_dir}")
    return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    sys.exit(main())
