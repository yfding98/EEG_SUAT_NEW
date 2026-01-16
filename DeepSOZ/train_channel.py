"""
通道级别分类训练脚本
以通道级别作为标签进行分类训练，预测每个通道是否为SOZ

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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix, roc_auc_score
)
import time
from datetime import datetime

# 导入本地模块
from utils import (
    read_manifest, get_patient_ids_from_manifest, split_patients,
    create_kfold_splits, save_checkpoint, load_checkpoint, EarlyStopping,
    DEEPSOZ_CHANNEL_ORDER, CHANNEL_TO_REGION, REGION_TO_IDX
)
from dataloader import ChannelDataset, create_data_loaders
from model import ChannelClassifier, HybridSOZModel, TransformerSOZModel, DeepSOZModel, create_model
from loss import (
    ChannelClassificationLoss, SimplifiedSOZLoss, 
    MapLossL2PosSum, MapLossL2Neg, MapLossMargin, CombinedSOZLoss
)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, 
                use_map_loss=False, map_losses=None, verbose=True):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        use_map_loss: 是否使用map损失
        map_losses: map损失函数字典
        verbose: 是否打印详细信息
    
    Returns:
        平均损失, 平均准确率
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        data = batch['data'].to(device)
        label = batch['channel_label'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        if hasattr(model, 'channel_head'):
            # 混合模型
            output, _ = model(data)
        else:
            output = model(data)
        
        # 基础分类损失
        loss = criterion(output, label)
        
        # 可选：添加map损失
        if use_map_loss and map_losses is not None:
            map_pred = torch.sigmoid(output)  # 将logits转换为概率
            map_loss_pos = map_losses['pos'](map_pred, label)
            map_loss_neg = map_losses['neg'](map_pred, label)
            map_loss_margin = map_losses['margin'](map_pred, label)
            
            loss = loss + 2.0 * map_loss_pos + 1.0 * map_loss_neg + 1.0 * map_loss_margin
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率（阈值0.5）
        pred = (torch.sigmoid(output) > 0.5).float()
        correct = (pred == label).sum().item()
        total_correct += correct
        total_samples += label.numel()
        
        if verbose and (batch_idx + 1) % 10 == 0:
            batch_acc = correct / label.numel()
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_acc


def evaluate(model, data_loader, criterion, device, threshold=0.5):
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        threshold: 预测阈值
    
    Returns:
        评估结果字典
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch['data'].to(device)
            label = batch['channel_label'].to(device)
            
            # 前向传播
            if hasattr(model, 'channel_head'):
                output, _ = model(data)
            else:
                output = model(data)
            
            # 计算损失
            loss = criterion(output, label)
            total_loss += loss.item()
            
            # 收集预测结果
            probs = torch.sigmoid(output)
            pred = (probs > threshold).float()
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 合并结果
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # 计算指标
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    # 样本级别准确率（所有通道都预测正确）
    sample_acc = np.mean(np.all(all_preds == all_labels, axis=1))
    
    # 通道级别准确率
    channel_acc = np.mean(all_preds == all_labels)
    
    # 计算每个通道的指标
    channel_f1 = []
    channel_precision = []
    channel_recall = []
    
    for ch in range(all_labels.shape[1]):
        if np.sum(all_labels[:, ch]) > 0:
            f1 = f1_score(all_labels[:, ch], all_preds[:, ch], zero_division=0)
            prec = precision_score(all_labels[:, ch], all_preds[:, ch], zero_division=0)
            rec = recall_score(all_labels[:, ch], all_preds[:, ch], zero_division=0)
        else:
            f1 = prec = rec = 0.0
        channel_f1.append(f1)
        channel_precision.append(prec)
        channel_recall.append(rec)
    
    # 宏平均
    macro_f1 = np.mean(channel_f1)
    macro_precision = np.mean(channel_precision)
    macro_recall = np.mean(channel_recall)
    
    # 微平均
    micro_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    
    # AUC (如果有正样本)
    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro')
    except ValueError:
        auc = 0.0
    
    results = {
        'loss': avg_loss,
        'sample_accuracy': sample_acc,
        'channel_accuracy': channel_acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'auc': auc,
        'channel_f1': channel_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results


def evaluate_samples_detailed(model, data_loader, criterion, device, channel_names, threshold=0.5):
    """
    对每个样本进行详细评估
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        channel_names: 通道名称列表
        threshold: 预测阈值
    
    Returns:
        包含每个样本详细信息的DataFrame
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch['data'].to(device)
            label = batch['channel_label'].to(device)
            
            # 前向传播
            if hasattr(model, 'channel_head'):
                output, _ = model(data)
            else:
                output = model(data)
            
            batch_size = data.shape[0]
            for i in range(batch_size):
                sample_output = output[i:i+1]
                sample_label = label[i:i+1]
                
                # 单样本损失
                sample_loss = criterion(sample_output, sample_label).item()
                
                # 预测结果
                probs = torch.sigmoid(sample_output).cpu().numpy().flatten()
                preds = (probs > threshold).astype(int)
                true_labels = sample_label.cpu().numpy().flatten()
                
                # 计算样本级别F1
                if true_labels.sum() > 0 or preds.sum() > 0:
                    sample_f1 = f1_score(true_labels, preds, zero_division=0)
                else:
                    sample_f1 = 1.0
                
                n_correct = np.sum(preds == true_labels)
                n_errors = np.sum(preds != true_labels)
                
                # 逐标签错误详情
                error_details = []
                for j, name in enumerate(channel_names):
                    if preds[j] != true_labels[j]:
                        if preds[j] == 1:
                            error_details.append(f"{name}:FP")
                        else:
                            error_details.append(f"{name}:FN")
                
                result = {
                    'sample_idx': batch_idx * data_loader.batch_size + i,
                    'pt_id': batch.get('patient_id', ['unknown'])[i] if 'patient_id' in batch else f'batch{batch_idx}_{i}',
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
    
    return pd.DataFrame(results)


def generate_sample_visualization(df, output_dir, channel_names):
    """
    生成样本级别可视化报告
    
    Args:
        df: 样本评估结果DataFrame
        output_dir: 输出目录
        channel_names: 通道名称列表
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    vis_dir = os.path.join(output_dir, 'sample_analysis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. 保存详细预测结果
    df.to_csv(os.path.join(vis_dir, 'sample_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 问题样本报告（按损失排序）
    poor_df = df.sort_values('sample_loss', ascending=False).head(20)
    poor_df.to_csv(os.path.join(vis_dir, 'poor_samples_report.csv'), index=False, encoding='utf-8-sig')
    
    # 3. 误差分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失分布
    ax1 = axes[0, 0]
    ax1.hist(df['sample_loss'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(df['sample_loss'].mean(), color='red', linestyle='--', 
                label=f"Mean: {df['sample_loss'].mean():.4f}")
    ax1.set_xlabel('Sample Loss')
    ax1.set_ylabel('Count')
    ax1.set_title('Loss Distribution')
    ax1.legend()
    
    # F1分布
    ax2 = axes[0, 1]
    ax2.hist(df['sample_f1'], bins=20, color='seagreen', edgecolor='white', alpha=0.8)
    ax2.axvline(df['sample_f1'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['sample_f1'].mean():.4f}")
    ax2.set_xlabel('Sample F1')
    ax2.set_ylabel('Count')
    ax2.set_title('F1 Score Distribution')
    ax2.legend()
    
    # 错误数量分布
    ax3 = axes[1, 0]
    error_counts = df['n_errors'].value_counts().sort_index()
    ax3.bar(error_counts.index, error_counts.values, color='coral', edgecolor='white', alpha=0.8)
    ax3.set_xlabel('Number of Errors')
    ax3.set_ylabel('Count')
    ax3.set_title('Errors per Sample')
    
    # 各通道错误统计
    ax4 = axes[1, 1]
    fp_counts = {name: 0 for name in channel_names}
    fn_counts = {name: 0 for name in channel_names}
    
    for _, row in df.iterrows():
        if row['error_details']:
            for error in row['error_details'].split(';'):
                if ':' in error:
                    label, error_type = error.split(':')
                    if error_type == 'FP' and label in fp_counts:
                        fp_counts[label] += 1
                    elif error_type == 'FN' and label in fn_counts:
                        fn_counts[label] += 1
    
    x = np.arange(len(channel_names))
    width = 0.35
    ax4.bar(x - width/2, [fp_counts[n] for n in channel_names], width, label='FP', color='#FFA500', alpha=0.8)
    ax4.bar(x + width/2, [fn_counts[n] for n in channel_names], width, label='FN', color='#FF6B6B', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Error Count')
    ax4.set_title('Errors by Channel')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 摘要统计
    summary = {
        'total_samples': len(df),
        'mean_loss': float(df['sample_loss'].mean()),
        'mean_f1': float(df['sample_f1'].mean()),
        'perfect_samples': int((df['n_errors'] == 0).sum()),
        'error_rate': float((df['n_errors'] > 0).sum() / len(df))
    }
    
    import json
    with open(os.path.join(vis_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n样本级别分析报告已保存至: {vis_dir}")
    print(f"  总样本数: {summary['total_samples']}")
    print(f"  平均损失: {summary['mean_loss']:.4f}")
    print(f"  平均F1: {summary['mean_f1']:.4f}")
    print(f"  完美预测: {summary['perfect_samples']} ({summary['perfect_samples']/summary['total_samples']*100:.1f}%)")
    print(f"  错误率: {summary['error_rate']*100:.1f}%")


def train_channel_classifier(
    data_root,
    manifest_path,
    output_dir='checkpoints/channel',
    model_type='channel',
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=15,
    pos_weight=3.0,
    use_focal_loss=False,
    use_map_loss=True,
    device='cuda',
    seed=42,
    visualize=False
):
    """
    训练通道分类器
    
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
        pos_weight: 正样本权重
        use_focal_loss: 是否使用Focal Loss
        use_map_loss: 是否使用Map损失（与DeepSOZ一致）
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
    train_dataset = ChannelDataset(data_root, train_ids, manifest, normalize=True)
    val_dataset = ChannelDataset(data_root, val_ids, manifest, normalize=True)
    test_dataset = ChannelDataset(data_root, test_ids, manifest, normalize=True)
    
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
    
    # 损失函数
    if use_focal_loss:
        criterion = SimplifiedSOZLoss(pos_weight=pos_weight, use_focal=True, gamma=2.0)
    else:
        criterion = SimplifiedSOZLoss(pos_weight=pos_weight, use_focal=False)
    
    # Map损失函数（与DeepSOZ一致）
    map_losses = None
    if use_map_loss:
        map_losses = {
            'pos': MapLossL2PosSum(scale=True).to(device),
            'neg': MapLossL2Neg(scale=True).to(device),
            'margin': MapLossMargin().to(device)
        }
    
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
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    print("\n开始训练...")
    print(f"使用Map损失: {use_map_loss}")
    print(f"正样本权重: {pos_weight}")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            use_map_loss=use_map_loss, map_losses=map_losses, verbose=False
        )
        
        # 验证
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss = val_results['loss']
        val_acc = val_results['channel_accuracy']
        val_f1 = val_results['macro_f1']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s) - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
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
    test_results = evaluate(model, test_loader, criterion, device)
    
    print(f"\n测试集结果:")
    print(f"  Loss: {test_results['loss']:.4f}")
    print(f"  Sample Accuracy: {test_results['sample_accuracy']:.4f}")
    print(f"  Channel Accuracy: {test_results['channel_accuracy']:.4f}")
    print(f"  Macro F1: {test_results['macro_f1']:.4f}")
    print(f"  Micro F1: {test_results['micro_f1']:.4f}")
    print(f"  Precision: {test_results['macro_precision']:.4f}")
    print(f"  Recall: {test_results['macro_recall']:.4f}")
    print(f"  AUC: {test_results['auc']:.4f}")
    
    # 每个通道的F1分数
    print("\n每个通道的F1分数:")
    for i, (ch_name, f1) in enumerate(zip(DEEPSOZ_CHANNEL_ORDER, test_results['channel_f1'])):
        region = CHANNEL_TO_REGION.get(ch_name.lower(), 'unknown')
        print(f"  {ch_name.upper():4s} ({region:10s}): {f1:.4f}")
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # 保存测试结果
    results_summary = {
        'metric': ['loss', 'sample_accuracy', 'channel_accuracy', 'macro_f1', 
                   'micro_f1', 'precision', 'recall', 'auc'],
        'value': [test_results['loss'], test_results['sample_accuracy'], 
                  test_results['channel_accuracy'], test_results['macro_f1'],
                  test_results['micro_f1'], test_results['macro_precision'],
                  test_results['macro_recall'], test_results['auc']]
    }
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    # 保存通道级别结果
    channel_results_df = pd.DataFrame({
        'channel': DEEPSOZ_CHANNEL_ORDER,
        'f1_score': test_results['channel_f1'],
        'region': [CHANNEL_TO_REGION.get(ch.lower(), 'unknown') for ch in DEEPSOZ_CHANNEL_ORDER]
    })
    channel_results_df.to_csv(os.path.join(output_dir, 'channel_results.csv'), index=False)
    
    print(f"\n结果已保存至: {output_dir}")
    
    # 生成样本级别可视化报告
    if visualize:
        print("\n生成样本级别可视化报告...")
        try:
            sample_df = evaluate_samples_detailed(
                model, test_loader, criterion, device, DEEPSOZ_CHANNEL_ORDER
            )
            generate_sample_visualization(sample_df, output_dir, DEEPSOZ_CHANNEL_ORDER)
        except Exception as e:
            print(f"生成可视化报告失败: {e}")
            import traceback
            traceback.print_exc()
    
    return model, history, test_results


def main():
    parser = argparse.ArgumentParser(description='通道级别分类训练')
    parser.add_argument('--data-root', type=str, default='.',
                        help='数据根目录')
    parser.add_argument('--manifest', type=str, default='converted_manifest.csv',
                        help='manifest文件路径')
    parser.add_argument('--output-dir', type=str, default='checkpoints/channel',
                        help='输出目录')
    parser.add_argument('--model-type', type=str, default='channel',
                        choices=['channel', 'hybrid', 'transformer'],
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
    parser.add_argument('--pos-weight', type=float, default=3.0,
                        help='正样本权重')
    parser.add_argument('--focal-loss', action='store_true',
                        help='使用Focal Loss')
    parser.add_argument('--no-map-loss', action='store_true',
                        help='不使用Map损失')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--visualize', action='store_true',
                        help='生成样本级别可视化报告')
    
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
    train_channel_classifier(
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        pos_weight=args.pos_weight,
        use_focal_loss=args.focal_loss,
        use_map_loss=not args.no_map_loss,
        device=args.device,
        seed=args.seed,
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()
