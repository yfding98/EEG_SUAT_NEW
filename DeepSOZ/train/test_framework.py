#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练框架测试脚本

测试所有组件是否正确集成
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from dataset import PrivateEEGDataset, create_dataloader
from losses import get_loss_function
from model_wrapper import create_model
from trainer import compute_metrics, count_parameters


def test_config():
    """测试配置加载"""
    print("1. 测试配置...")
    config = get_config()
    print(f"   数据路径: {config.data.manifest_path}")
    print(f"   通道数: {config.model.n_channels}")
    print(f"   学习率: {config.training.learning_rate}")
    print("   ✓ 配置加载成功\n")


def test_dataset():
    """测试数据集"""
    print("2. 测试数据集...")
    try:
        config = get_config()
        
        # 创建小规模数据集进行测试
        dataset = PrivateEEGDataset(
            manifest_path=config.data.manifest_path,
            data_roots=config.data.edf_data_roots,
            label_type='channel',
            config=config.data,
            max_seizures=2  # 限制发作数用于测试
        )
        
        print(f"   数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   数据形状: {sample['data'].shape}")
            print(f"   标签形状: {sample['labels'].shape}")
            print(f"   患者ID: {sample['pt_id']}")
        
        print("   ✓ 数据集测试成功\n")
    except Exception as e:
        print(f"   ✗ 数据集测试失败: {e}\n")


def test_model():
    """测试模型"""
    print("3. 测试模型...")
    try:
        # 创建模型
        model = create_model('channel', {
            'n_channels': 19,
            'n_bands': 5,
            'time_steps': 200,
            'n_windows': 45,
            'temporal_hidden_dim': 16,  # 减小用于测试
            'graph_hidden_dim': 16,
            'dropout': 0.3
        })
        
        # 统计参数
        total_params, trainable_params = count_parameters(model)
        print(f"   总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        batch_size = 2
        x = torch.randn(batch_size, 45, 19, 200)  # (B, T, C, L)
        output = model(x)
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        
        print("   ✓ 模型测试成功\n")
    except Exception as e:
        print(f"   ✗ 模型测试失败: {e}\n")


def test_losses():
    """测试损失函数"""
    print("4. 测试损失函数...")
    try:
        # 测试通道级别损失
        loss_fn = get_loss_function('channel', {
            'classification_loss': 'bce',
            'pos_weight': 2.0
        })
        
        batch_size = 4
        pred = torch.randn(batch_size, 19)
        target = torch.zeros(batch_size, 19)
        target[:, [3, 7, 8]] = 1  # 模拟SOZ通道
        
        loss = loss_fn(pred, target)
        print(f"   损失值: {loss.item():.4f}")
        
        print("   ✓ 损失函数测试成功\n")
    except Exception as e:
        print(f"   ✗ 损失函数测试失败: {e}\n")


def test_metrics():
    """测试评估指标"""
    print("5. 测试评估指标...")
    try:
        # 多标签分类指标
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
        y_prob = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.1], [0.6, 0.7, 0.3]])
        
        metrics = compute_metrics(y_true, y_pred, y_prob, 'multilabel')
        print(f"   多标签F1-Macro: {metrics.get('f1_macro', 0):.4f}")
        
        # 多分类指标
        y_true_mc = np.array([0, 1, 2, 0])
        y_prob_mc = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1]
        ])
        
        metrics_mc = compute_metrics(y_true_mc, None, y_prob_mc, 'multiclass')
        print(f"   多分类Accuracy: {metrics_mc.get('accuracy', 0):.4f}")
        
        print("   ✓ 评估指标测试成功\n")
    except Exception as e:
        print(f"   ✗ 评估指标测试失败: {e}\n")


def test_integration():
    """测试端到端集成"""
    print("6. 测试端到端集成...")
    try:
        # 创建小模型
        model = create_model('channel', {
            'n_channels': 19,
            'n_bands': 5,
            'time_steps': 200,
            'n_windows': 45,
            'temporal_hidden_dim': 8,
            'graph_hidden_dim': 8,
            'dropout': 0.1
        })
        
        # 创建损失函数
        criterion = get_loss_function('channel')
        
        # 模拟数据
        batch_size = 2
        x = torch.randn(batch_size, 45, 19, 200)
        y = torch.zeros(batch_size, 19)
        y[:, [3, 7]] = 1
        
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # 反向传播
        loss.backward()
        
        print(f"   损失值: {loss.item():.4f}")
        print(f"   输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
        
        print("   ✓ 端到端集成测试成功\n")
    except Exception as e:
        print(f"   ✗ 端到端集成测试失败: {e}\n")


def main():
    """主测试函数"""
    print("="*60)
    print("DeepSOZ 私有数据集训练框架 - 组件测试")
    print("="*60)
    
    test_config()
    test_dataset()
    test_model()
    test_losses()
    test_metrics()
    test_integration()
    
    print("="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
