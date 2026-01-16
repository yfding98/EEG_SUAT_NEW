# DeepSOZ 私有数据集训练框架

本框架实现了基于STGNN_SOZ_Locator模型的多粒度癫痫发作起始区（SOZ）定位训练，支持三种不同粒度的分类任务。

## 目录结构

```
EEG_SUAT_NEW/DeepSOZ/train/
├── config.py              # 训练配置文件
├── dataset.py             # 数据加载器（支持EDF文件和三种标签粒度）
├── losses.py              # 损失函数模块（参考DeepSOZ设计）
├── model_wrapper.py       # STGNN模型适配器
├── trainer.py             # 训练工具类
├── train_onset_zone.py    # 脑区级别分类训练脚本
├── train_hemi.py          # 半球级别分类训练脚本
├── train_channel.py       # 通道级别分类训练脚本
└── checkpoints/           # 模型检查点保存目录
    ├── onset_zone/
    ├── hemi/
    └── channel/
└── logs/                  # 训练日志保存目录
    ├── onset_zone/
    ├── hemi/
    └── channel/
```

## 核心功能

### 1. 数据加载器 (dataset.py)
- 支持EDF文件读取（多种后端：pyedflib, MNE）
- 自动提取标准19通道（DeepSOZ格式）
- 信号预处理：带通滤波(1.6-30Hz)、幅值裁剪、重采样(200Hz)
- 支持三种标签粒度：
  - onset_zone: 脑区级别多标签分类
  - hemi: 半球级别单标签分类
  - channel: 通道级别多标签分类
- 交叉验证患者划分（按患者ID分层）

### 2. 损失函数 (losses.py)
- 参考DeepSOZ的损失函数设计
- 支持多种损失类型：
  - BCEWithLogitsLoss（带正负样本权重）
  - Focal Loss（处理类别不平衡）
  - Dice Loss（分割任务）
  - DeepSOZ风格的Map Loss（位置定位）
- 组合损失函数支持多任务学习

### 3. 模型适配器 (model_wrapper.py)
- 基于STGNN_SOZ_Locator架构
- 支持三种输出类型：
  - channel: 19通道SOZ定位
  - onset_zone: 5脑区分类
  - hemi: 4半球分类
- 通道到脑区/半球的聚合层
- 多任务模型同时输出三种粒度预测

### 4. 训练脚本

#### 脑区级别分类 (train_onset_zone.py)
```bash
python train_onset_zone.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --pos-weight 2.0 \
    --n-folds 5 \
    --experiment-name onset_zone_exp
```

#### 半球级别分类 (train_hemi.py)
```bash
python train_hemi.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --n-folds 5 \
    --experiment-name hemi_exp
```

#### 通道级别分类 (train_channel.py)
```bash
python train_channel.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --pos-weight 5.0 \
    --n-folds 5 \
    --experiment-name channel_exp
```

## 训练配置 (config.py)

### 数据配置
- EDF文件路径和数据根目录
- 采样率：200Hz（DeepSOZ标准）
- 预处理参数：滤波、标准化等
- 窗口长度：1秒，无重叠

### 模型配置
- 时间特征提取器：因果时间卷积
- 图学习方法：parameter/feature/combined
- 图卷积类型：GCN/GAT
- 分类头：带时间注意力

### 训练配置
- 优化器：AdamW
- 学习率调度：余弦退火
- 混合精度训练
- 梯度裁剪

## 使用方法

### 1. 准备数据
确保你的`detected_ensemble_updated_with_baseline.csv`文件格式正确，包含以下列：
- pt_id, fn, loc (EDF文件路径)
- onset_zone, hemi, region (标签)
- 19个通道列 (fp1, f7, t3, ..., o2)

### 2. 运行训练
```bash
# 脑区级别训练
cd EEG_SUAT_NEW/DeepSOZ/train
python train_onset_zone.py --n-folds 5

# 半球级别训练
python train_hemi.py --n-folds 5

# 通道级别训练
python train_channel.py --n-folds 5
```

### 3. 交叉验证
- 自动按患者ID进行5折交叉验证
- 防止数据泄露
- 每折独立训练和验证

## 模型架构

### STGNN_SOZ_Locator
1. **时间特征提取器**：因果时间卷积，提取高维时间特征
2. **自适应图学习**：学习通道间功能连接
3. **图卷积**：在图结构上交换信息
4. **分类头**：带时间注意力的分类器

### 损失函数设计
参考DeepSOZ的多组件损失：
- 主分类损失（BCE/Focal/Dice）
- Map正类损失（确保SOZ通道预测值高）
- Map负类损失（确保非SOZ通道预测值低）
- Margin损失（确保正负类差距）

## 评估指标

- **多标签任务** (onset_zone, channel)：
  - F1-Macro, F1-Micro
  - Precision, Recall
  - AUC-ROC

- **多分类任务** (hemi)：
  - Accuracy
  - F1-Macro
  - AUC-OVR

## 文件说明

- `config.py`: 统一管理所有配置参数
- `dataset.py`: 数据加载和预处理管道
- `losses.py`: 损失函数定义
- `model_wrapper.py`: 模型架构适配
- `trainer.py`: 训练循环和工具函数
- `train_*.py`: 各粒度训练脚本

## 注意事项

1. 确保EDF文件路径正确且可访问
2. 数据预处理与DeepSOZ保持一致（200Hz, 1.6-30Hz滤波）
3. 建议使用GPU进行训练以提高效率
4. 模型参数和日志自动保存到对应目录
5. 支持中断恢复训练

## 性能优化

- 混合精度训练（AMP）
- 梯度裁剪防止梯度爆炸
- 早停机制防止过拟合
- 学习率调度器优化收敛
