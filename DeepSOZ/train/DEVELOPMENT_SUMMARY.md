# DeepSOZ 私有数据集训练框架 - 开发总结

我们已经成功创建了一个完整的训练框架，用于在私有数据集上训练STGNN_SOZ_Locator模型，支持三种不同粒度的癫痫发作起始区（SOZ）定位分类任务。

## 完成的工作

### 1. 配置管理 (config.py)
- 创建了完整的配置系统，包含数据、模型、训练、损失函数等所有配置项
- 支持灵活的参数调整和实验管理
- 预设了DeepSOZ兼容的参数配置

### 2. 数据加载器 (dataset.py)
- 实现了完整的EDF文件读取和预处理管道
- 支持三种标签粒度：onset_zone（脑区）、hemi（半球）、channel（通道）
- 包含信号预处理：带通滤波、幅值裁剪、重采样、标准化
- 实现了交叉验证数据划分功能

### 3. 损失函数模块 (losses.py)
- 实现了参考DeepSOZ的多组件损失函数
- 支持BCE、Focal、Dice等多种损失类型
- 包含DeepSOZ风格的Map Loss（正类、负类、Margin）
- 为三种任务类型提供了专门的损失函数

### 4. 模型适配器 (model_wrapper.py)
- 基于STGNN_SOZ_Locator架构
- 支持三种输出类型：channel、onset_zone、hemi
- 实现了通道到脑区/半球的聚合层
- 支持多任务联合训练

### 5. 训练工具 (trainer.py)
- 通用的训练循环和验证逻辑
- 支持混合精度训练、梯度裁剪
- 完整的指标计算和日志记录
- 检查点保存和加载功能

### 6. 专用训练脚本
- `train_onset_zone.py`: 脑区级别分类训练
- `train_hemi.py`: 半球级别分类训练  
- `train_channel.py`: 通道级别分类训练
- 支持5折交叉验证和实验管理

## 技术特点

### 与DeepSOZ兼容
- 保持了与DeepSOZ相同的预处理流程（200Hz采样率，1.6-30Hz滤波）
- 使用相同的19通道标准和顺序
- 损失函数设计参考DeepSOZ的多组件策略

### 多粒度训练
- **脑区级别**: 5类多标签分类（frontal, temporal, central, parietal, occipital）
- **半球级别**: 4类单标签分类（L, R, B, U）
- **通道级别**: 19通道多标签分类（SOZ通道定位）

### 模型架构
- 时间特征提取器：因果时间卷积
- 自适应图学习：动态学习通道间连接
- 图卷积：GCN/GAT交换通道信息
- 分类头：带时间注意力的分类器

## 使用方法

### 1. 准备数据
确保`detected_ensemble_updated_with_baseline.csv`文件包含所需列：
- 患者信息：pt_id, fn, loc
- 标签：onset_zone, hemi, region
- 通道标签：fp1, f7, t3, ..., o2

### 2. 运行训练
```bash
# 脑区级别训练
python train_onset_zone.py --n-folds 5

# 半球级别训练
python train_hemi.py --n-folds 5

# 通道级别训练
python train_channel.py --n-folds 5
```

### 3. 结果查看
- 模型检查点保存在 `checkpoints/{task_type}/`
- 训练日志保存在 `logs/{task_type}/`
- 支持中断恢复训练

## 优势

1. **模块化设计**: 各组件高度解耦，便于维护和扩展
2. **DeepSOZ兼容**: 与现有DeepSOZ流程完全兼容
3. **灵活配置**: 通过配置文件轻松调整所有参数
4. **多粒度支持**: 同一框架支持三种不同粒度的分类任务
5. **鲁棒性**: 包含错误处理、数据验证和异常恢复机制

## 未来扩展

1. 支持更多脑区划分方式
2. 集成更多图神经网络架构
3. 添加模型解释性和可视化功能
4. 支持在线学习和增量训练

此框架为DeepSOZ项目在私有数据集上的应用提供了完整的解决方案，能够充分利用STGNN_SOZ_Locator的强大能力进行多粒度SOZ定位研究。
