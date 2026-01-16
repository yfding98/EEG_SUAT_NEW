"""
脑区级别分类训练脚本
以region（脑区）作为标签进行分类训练

基于DeepSOZ项目的训练策略，使用自定义损失函数
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
from datetime import datetime

# 导入本地模块
from utils import (
    read_manifest, get_patient_ids_from_manifest, split_patients,
    create_kfold_splits, save_checkpoint, load_checkpoint, EarlyStopping,
    REGION_TO_IDX, IDX_TO_REGION
)
from dataloader import RegionDataset, create_data_loaders
from model import RegionClassifier, HybridSOZModel, TransformerSOZModel, create_model
from loss import RegionClassificationLoss, RegionSOZLoss, MapLossL2PosSum, MapLossL2Neg, MapLossMargin


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, verbose=True):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        verbose: 是否打印详细信息
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        data = batch['data'].to(device)
        label = batch['region_label_onehot'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        if hasattr(model, 'region_head'):
            # 混合模型
            _, output = model(data)
        else:
            output = model(data)
        
        # 计算损失
        loss = criterion(output, label)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测结果（多标签转单标签）
        pred = torch.argmax(torch.sigmoid(output), dim=1)
        true = torch.argmax(label, dim=1) if label.dim() > 1 else label
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(true.cpu().numpy())
        
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失, 准确率, 预测结果, 真实标签
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch['data'].to(device)
            label = batch['region_label_onehot'].to(device)
            
            # 前向传播
            if hasattr(model, 'region_head'):
                _, output = model(data)
            else:
                output = model(data)
            
            # 计算损失
            loss = criterion(output, label)
            total_loss += loss.item()
            
            # 收集预测结果
            probs = torch.sigmoid(output)
            pred = torch.argmax(probs, dim=1)
            true = torch.argmax(label, dim=1) if label.dim() > 1 else label
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(true.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_region_classifier(
    data_root,
    manifest_path,
    output_dir='checkpoints/region',
    model_type='region',
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=15,
    device='cuda',
    seed=42
):
    """
    训练脑区分类器
    
    Args:
        data_root: 数据根目录
        manifest_path: manifest文件路径
        output_dir: 输出目录
        model_type: 模型类型
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        device: 设备
        seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 读取manifest
    print(f"读取manifest: {manifest_path}")
    manifest = read_manifest(manifest_path)
    patient_ids = get_patient_ids_from_manifest(manifest)
    print(f"共有 {len(patient_ids)} 个患者, {len(manifest)} 条记录")
    
    # 划分数据集
    train_ids, val_ids, test_ids = split_patients(patient_ids, 0.7, 0.15, seed)
    print(f"数据集划分: 训练 {len(train_ids)}, 验证 {len(val_ids)}, 测试 {len(test_ids)}")
    
    # 创建数据集
    train_dataset = RegionDataset(data_root, train_ids, manifest, normalize=True)
    val_dataset = RegionDataset(data_root, val_ids, manifest, normalize=True)
    test_dataset = RegionDataset(data_root, test_ids, manifest, normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    model = create_model(model_type, num_channels=19, num_regions=5, dropout=0.3)
    model = model.to(device)
    model = model.double()  # 使用double精度与DeepSOZ保持一致
    
    print(f"模型类型: {model_type}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数 (多标签分类)
    criterion = RegionClassificationLoss(num_regions=5, multi_label=True)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 早停
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, verbose=False
        )
        
        # 验证
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s) - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> 保存最佳模型 (val_loss: {val_loss:.4f})')
        
        # 早停检查
        early_stopping(val_loss, model, best_model_path)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break
    
    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time/60:.1f} 分钟")
    
    # 加载最佳模型并在测试集上评估
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # 详细分类报告
    region_names = list(REGION_TO_IDX.keys())
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=region_names, zero_division=0))
    
    # 混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    print("混淆矩阵:")
    print(cm)
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # 保存测试结果
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': test_preds.tolist(),
        'labels': test_labels.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    results_df = pd.DataFrame({
        'metric': ['test_loss', 'test_accuracy'],
        'value': [test_loss, test_acc]
    })
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    print(f"\n结果已保存至: {output_dir}")
    
    return model, history, results


def main():
    parser = argparse.ArgumentParser(description='脑区级别分类训练')
    parser.add_argument('--data-root', type=str, default='.',
                        help='数据根目录')
    parser.add_argument('--manifest', type=str, default='converted_manifest.csv',
                        help='manifest文件路径')
    parser.add_argument('--output-dir', type=str, default='checkpoints/region',
                        help='输出目录')
    parser.add_argument('--model-type', type=str, default='region',
                        choices=['region', 'hybrid', 'transformer'],
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    if not os.path.isabs(args.manifest):
        args.manifest = os.path.join(script_dir, args.manifest)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    if not os.path.isabs(args.data_root):
        args.data_root = script_dir
    
    # 检查文件
    if not os.path.exists(args.manifest):
        print(f"错误: manifest文件不存在: {args.manifest}")
        return
    
    # 开始训练
    train_region_classifier(
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
