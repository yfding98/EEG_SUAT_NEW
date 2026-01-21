import pandas as pd
import os
import re

"""
将原来的 E:\output\segment_results_norm下 的 new_data_process.csv 转到
E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\DeepSOZ 下的 detected_ensemble.csv中的格式

"""

# 读取两个 CSV 文件
df_process = pd.read_csv(r'E:\output\segment_results_norm\new_data_process.csv', encoding='utf-8')
# 或者尝试自动检测编码
import chardet
with open(r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\DeepSOZ\converted_manifest_validated.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"文件编码为：{encoding}")
df_ensemble = pd.read_csv(r'E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\DeepSOZ\converted_manifest_validated.csv', encoding=encoding)

# 解析 segments 字段（字符串转为列表）
def parse_segments(seg_str):
    if pd.isna(seg_str) or seg_str.strip() == '':
        return []
    # 清理并解析字符串格式的列表
    seg_str = seg_str.strip().replace(' ', '')
    # 使用正则匹配所有浮点数或整数对
    matches = re.findall(r'\[(\d+\.?\d*),(\d+\.?\d*)\]', seg_str)
    return [(float(start), float(end)) for start, end in matches]

# 从 set_file 提取 loc 对应字段（用于匹配 detected_ensemble.csv 中的 loc）
def extract_loc_from_setfile(set_file):
    # 提取类似 "头皮数据-6例\刘娟\SZ1.edf" 的部分
    parts = set_file.split(os.sep)
    patient_dir = parts[-2]          # 如 "刘娟"
    dataset_group = parts[-3]        # 如 "头皮数据-6例"
    sz_file = parts[-1]              # 如 "SZ1_filtered_3_45_postICA_eye.set"
    # 提取 SZ 编号，如 SZ1
    sz_match = re.search(r'(SZ\d+)', sz_file)
    if not sz_match:
        return None
    sz_base = sz_match.group(1)
    loc = f"{dataset_group}{os.sep}{patient_dir}{os.sep}{sz_base}.edf"
    return loc

# 构建 new_data_process 的映射字典：loc -> (base_line, sz_starts, sz_ends)
process_map = {}
for _, row in df_process.iterrows():
    loc = extract_loc_from_setfile(row['set_file'])
    if loc is None:
        continue
    segments = parse_segments(row['segments'])
    if len(segments) < 2:
        # 至少需要一个基线 + 一个发作段
        continue

    mask_segments = row['mask_segments']
    base_line = segments[0]
    seizure_segments = segments[1:]
    nsz = len(seizure_segments)
    sz_starts = ';'.join([str(s[0]) for s in seizure_segments])
    sz_ends = ';'.join([str(s[1]) for s in seizure_segments])
    base_line_str = f"{base_line[0]},{base_line[1]}"
    process_map[loc] = {
        'sz_starts': sz_starts,
        'sz_ends': sz_ends,
        'base_line': base_line_str,
        'mask_segments': mask_segments,
        'nsz': nsz
    }

# 更新 detected_ensemble.csv
updated_rows = []
for _, row in df_ensemble.iterrows():
    loc = row['loc']
    if loc in process_map:
        update_info = process_map[loc]
        row = row.copy()
        row['sz_starts'] = update_info['sz_starts']
        row['sz_ends'] = update_info['sz_ends']
        row['base_line'] = update_info['base_line']
        row['nsz'] = update_info['nsz']
        row['mask_segments'] = update_info['mask_segments']
        updated_rows.append(row)
    # 如果没有匹配项，则跳过（不保留）

# 转换为 DataFrame 并保存
df_updated = pd.DataFrame(updated_rows)

# 确保 base_line 列存在（即使某些行没有，但这里只保留有匹配的）
df_updated = df_updated.reset_index(drop=True)

# 保存到新文件
output_file = 'detected_ensemble_updated.csv'
df_updated.to_csv(output_file, index=False, encoding='utf-8')

print(f"已成功生成更新后的文件：{output_file}")