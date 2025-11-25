# 通道顺序统一修复总结

## 问题发现

用户发现数据中存在两种不同的通道顺序，导致：
- 不同样本的通道顺序不一致
- NHFE数据（21*5*seq_len）中的通道维度顺序混乱
- 标签映射可能出错
- 模型训练时通道对应关系混乱

## 修复方案

### 1. 建立标准通道顺序

在 `NHFEDataLoader` 类中：
- **第一次加载**：将第一个样本的通道顺序作为标准顺序
- **后续加载**：所有样本统一到此标准顺序

### 2. 通道一致性检查

对每个样本进行以下检查：
1. **通道数量检查**：确保通道数量与标准顺序一致
2. **通道名称检查**：确保通道名称集合与标准顺序一致
3. **通道顺序检查**：如果名称相同但顺序不同，自动重新排序

### 3. 自动重排序NHFE数据

如果通道顺序不一致：
- 创建重排序索引：将当前样本的通道顺序映射到标准顺序
- 重新排序NHFE数据：
  - 2D数据 `(n_channels, n_timepoints)`: `nhfe_data[reorder_idx, :]`
  - 3D数据 `(n_channels, n_bands, n_timepoints)`: `nhfe_data[reorder_idx, :, :]`
- 确保所有数据的通道顺序保持一致

### 4. 数据验证

在返回 `PatientData` 对象前进行验证：
- NHFE数据通道数 = 标准通道数
- 通道名称列表 = 标准通道顺序
- 所有数据（NHFE序列、多频段NHFE）都统一到标准通道顺序

## 代码修改

### `EEG_SUAT_NEW/data/loader.py`

1. **添加标准通道顺序属性**：
   ```python
   class NHFEDataLoader:
       def __init__(self, ...):
           # 标准通道顺序（第一次加载时设置，后续所有数据统一到此顺序）
           self.standard_channel_order: Optional[List[str]] = None
   ```

2. **在 `load_from_npz` 方法中调用统一函数**：
   ```python
   # 统一通道顺序
   ch_names, nhfe_all_bands_data = self._normalize_channel_order(
       ch_names, nhfe_all_bands_data, patient_id
   )
   ```

3. **添加 `_normalize_channel_order` 方法**：
   - 第一次加载时设置标准通道顺序
   - 检查通道数量和名称集合
   - 如果顺序不一致，自动重新排序NHFE数据
   - 支持2D和3D NHFE数据（单频段和多频段）

4. **在metadata中标记**：
   ```python
   metadata = {
       ...
       'channel_order_normalized': True  # 标记通道顺序已统一
   }
   ```

## 使用效果

### 修复前
- 不同样本可能有不同的通道顺序
- NHFE数据（21*5*seq_len）中的通道维度顺序不一致
- 标签映射可能出错
- 模型训练时通道对应关系混乱

### 修复后
- ✅ 所有样本统一到标准通道顺序
- ✅ NHFE数据（21*5*seq_len）中的通道维度统一
- ✅ 标签映射使用标准通道顺序
- ✅ 模型训练时通道对应关系一致
- ✅ 自动检测并修复通道顺序不一致的问题
- ✅ 详细的日志输出，便于调试

## 日志输出示例

```
[通道顺序] 设置标准通道顺序 (21 通道): ['Fp1', 'Fp2', 'F3', 'F4', ...]

[通道顺序] 患者 刘娟-389to399_401to432: 通道顺序不一致，正在重新排序...
   当前顺序: ['F3', 'F4', 'Fp1', 'Fp2', ...]
   标准顺序: ['Fp1', 'Fp2', 'F3', 'F4', ...]
[通道顺序] ✓ 通道顺序已统一到标准顺序
```

## 注意事项

1. **标准通道顺序**：以第一个成功加载的样本的通道顺序为准
2. **通道名称必须一致**：如果通道名称不同，会抛出错误
3. **数据维度**：支持2D `(n_channels, n_timepoints)` 和3D `(n_channels, n_bands, n_timepoints)` 数据
4. **标签映射**：标签始终使用标准通道顺序进行映射
5. **数据一致性**：所有数据（NHFE序列、多频段NHFE）都统一到标准通道顺序

## 验证方法

可以通过以下方式验证通道顺序是否统一：

1. **检查日志输出**：查看是否有"通道顺序不一致"的警告
2. **检查metadata**：查看 `channel_order_normalized` 字段
3. **检查PatientData对象**：`patient_data.channel_names` 显示标准通道顺序
4. **检查NHFE数据**：确保所有患者的 `nhfe_data` 的第一个维度（通道维度）顺序一致

## 技术细节

### 重排序索引创建

```python
reorder_idx = []
for std_ch in self.standard_channel_order:
    reorder_idx.append(ch_names.index(std_ch))
```

### 数据重排序

```python
# 2D数据: (n_channels, n_timepoints)
nhfe_data = nhfe_data[reorder_idx, :]

# 3D数据: (n_channels, n_bands, n_timepoints)
nhfe_data = nhfe_data[reorder_idx, :, :]
```

### 错误处理

- 通道数量不匹配：抛出 `ValueError`，显示详细信息
- 通道名称不匹配：抛出 `ValueError`，显示缺失和多余的通道
- 标准通道不在当前样本中：抛出 `ValueError`，显示详细信息

## 相关文件

- `EEG_SUAT_NEW/data/loader.py`: 数据加载器，包含通道顺序统一逻辑
- `EEG_SUAT/nfhe_base/nhfe_dataset.py`: 旧版本的数据集加载器，也有类似的通道顺序统一逻辑
- `EEG_SUAT/nfhe_base/CHANNEL_ORDER_FIX.md`: 旧版本的修复文档



