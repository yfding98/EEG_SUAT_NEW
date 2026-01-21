"""
脑区映射脚本
根据21通道（19标准 + 2蝶骨）脑区映射规则自动补充converted_manifest.csv文件中的region字段

21通道脑区映射规则:
BRAIN_REGIONS_21 = { 
    'frontal':    ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'central':    ['C3', 'C4', 'Cz'], 
    'temporal':   ['T3', 'T4', 'T5', 'T6', 'Sph-L', 'Sph-R'],  # 或 T7/T8/P7/P8，含蝶骨电极
    'parietal':   ['P3', 'P4', 'Pz'],
    'occipital':  ['O1', 'O2']
}

DEEPSOZ_REGION_MAP = {
    'A': [
        'Fp1', 'Fp2',
        'F3', 'F4', 'F7', 'F8', 'Fz',
        'C3', 'C4', 'Cz',
        'T3', 'T4',   # T7, T8
        'Sph-L', 'Sph-R'  # 蝶骨电极，靠近前颞叶
    ],
    'P': [
        'T5', 'T6',  # P7, P8
        'P3', 'P4', 'Pz',
        'O1', 'O2'
    ]
}

蝶骨电极说明:
- Sph-L (Sphenoidal Left): 左侧蝶骨电极，靠近前颞叶内侧
- Sph-R (Sphenoidal Right): 右侧蝶骨电极，靠近前颞叶内侧
- 由于其解剖位置接近前颞叶，归类到 temporal 脑区和 A (Anterior) 区域

"""

import pandas as pd
import numpy as np
import os
import argparse


# 21通道脑区映射定义（用于onset_zone字段）
# 标准19通道 + 2蝶骨电极 (sph-l, sph-r)
BRAIN_REGIONS_21 = {
    'frontal': ['fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz'],
    'central': ['c3', 'c4', 'cz'],
    'temporal': ['t3', 't4', 't5', 't6', 't7', 't8', 'p7', 'p8', 'sph-l', 'sph-r'],  # 包含替代命名和蝶骨电极
    'parietal': ['p3', 'p4', 'pz'],
    'occipital': ['o1', 'o2']
}

# DeepSOZ脑区映射（用于region字段，与DeepSOZ保持一致）
# A = Anterior（前部）, P = Posterior（后部）
# 蝶骨电极靠近前颞叶，归入A区
DEEPSOZ_REGION_MAP = {
    'A': ['fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz', 'c3', 'c4', 'cz', 't3', 't4', 't7', 't8', 'sph-l', 'sph-r'],
    'P': ['t5', 't6', 'p7', 'p8', 'p3', 'p4', 'pz', 'o1', 'o2']
}

# 反向映射：通道名 -> 脑区（用于onset_zone）
CHANNEL_TO_REGION = {}
for region, channels in BRAIN_REGIONS_21.items():
    for channel in channels:
        CHANNEL_TO_REGION[channel.lower()] = region

# 反向映射：通道名 -> DeepSOZ区域（A/P）
CHANNEL_TO_DEEPSOZ = {}
for region, channels in DEEPSOZ_REGION_MAP.items():
    for channel in channels:
        CHANNEL_TO_DEEPSOZ[channel.lower()] = region


def get_channel_region(channel_name):
    """
    根据通道名称获取对应的脑区
    
    Args:
        channel_name: 通道名称 (如 'fp1', 'T4' 等)
    
    Returns:
        脑区名称 (如 'frontal', 'temporal' 等)
    """
    channel_lower = channel_name.lower().strip()
    return CHANNEL_TO_REGION.get(channel_lower, 'unknown')


def determine_deepsoz_region(row, channel_columns):
    """
    根据CSV行中的通道标记确定DeepSOZ区域（A或P）
    
    DeepSOZ区域定义：
    - A (Anterior): Fp1, Fp2, F3, F4, F7, F8, Fz, C3, C4, Cz, T3, T4
    - P (Posterior): T5, T6, P3, P4, Pz, O1, O2
    
    策略:
    1. 统计A区和P区中被标记为1的通道数量
    2. 返回数量较多的区域
    3. 如果相同，返回'A'（前部优先）
    4. 如果没有任何标记，返回空字符串
    
    Args:
        row: DataFrame的一行
        channel_columns: 通道列名列表
    
    Returns:
        'A', 'P', 或空字符串
    """
    region_counts = {'A': 0, 'P': 0}
    
    for channel in channel_columns:
        try:
            value = row.get(channel, 0)
            if pd.notna(value) and int(value) == 1:
                deepsoz_region = CHANNEL_TO_DEEPSOZ.get(channel.lower(), None)
                if deepsoz_region:
                    region_counts[deepsoz_region] += 1
        except (ValueError, TypeError):
            continue
    
    # 如果没有任何通道被标记
    if region_counts['A'] == 0 and region_counts['P'] == 0:
        return 'U'  # 表示无法确定
    
    # 返回数量较多的区域，相同时返回A
    if region_counts['A'] >= region_counts['P']:
        return 'A'
    else:
        return 'P'


def determine_onset_zone(row, channel_columns):
    """
    确定发作起始区域（onset_zone），使用具体脑区名称
    
    可以是多个脑区，用逗号分隔，如 "frontal,temporal"
    
    Args:
        row: DataFrame的一行
        channel_columns: 通道列名列表
    
    Returns:
        onset_zone字符串，如 "frontal" 或 "frontal,temporal"
    """
    region_counts = {
        'frontal': 0,
        'central': 0,
        'temporal': 0,
        'parietal': 0,
        'occipital': 0
    }
    
    for channel in channel_columns:
        try:
            value = row.get(channel, 0)
            if pd.notna(value) and int(value) == 1:
                region = get_channel_region(channel)
                if region in region_counts:
                    region_counts[region] += 1
        except (ValueError, TypeError):
            continue
    
    # 获取所有有标记的脑区，按标记数量降序排序
    involved_regions = [(r, c) for r, c in region_counts.items() if c > 0]
    involved_regions.sort(key=lambda x: x[1], reverse=True)
    
    if len(involved_regions) == 0:
        return ''  # 无法确定时返回空字符串
    
    # 用逗号分隔多个脑区
    return ','.join([r for r, c in involved_regions])


def process_manifest(input_csv, output_csv=None):
    """
    处理manifest CSV文件，添加region和onset_zone字段
    
    - region: DeepSOZ格式，A（Anterior前部）或 P（Posterior后部）
    - onset_zone: 具体脑区名称，可多个用逗号分隔
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径（默认覆盖输入文件）
    
    Returns:
        处理后的DataFrame
    """
    print(f"正在读取文件: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 确定通道列（21通道：19标准 + 2蝶骨）
    standard_channels = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 
                        'fz', 'cz', 'pz', 'fp2', 'f8', 't4', 't6', 'o2', 
                        'f4', 'c4', 'p4']
    extra_channels = ['sph-l', 'sph-r']  # 蝶骨电极
    all_standard_channels = standard_channels + extra_channels
    
    # 找到实际存在的通道列
    existing_channels = [ch for ch in all_standard_channels if ch in df.columns]
    print(f"找到 {len(existing_channels)} 个通道列: {existing_channels}")
    
    # 为每行计算region和onset_zone
    regions = []
    onset_zones = []
    
    for idx, row in df.iterrows():
        # DeepSOZ格式的region (A/P)
        region = determine_deepsoz_region(row, existing_channels)
        regions.append(region)
        
        # 具体脑区的onset_zone
        onset_zone = determine_onset_zone(row, existing_channels)
        onset_zones.append(onset_zone)
    
    # 添加region列（DeepSOZ格式：A/P）
    df['region'] = regions
    
    # 添加onset_zone列（具体脑区）
    df['onset_zone'] = onset_zones
    
    # 统计分布
    print("\n===== DeepSOZ Region (A/P) 分布 =====")
    print(df['region'].value_counts())
    
    print("\n===== Onset Zone 分布 =====")
    print(df['onset_zone'].value_counts())
    
    # 保存结果
    if output_csv is None:
        output_csv = input_csv
    
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存至: {output_csv}")
    
    return df


def analyze_channel_distribution(df):
    """
    分析各通道的标记分布
    
    Args:
        df: DataFrame
    """
    print("\n===== 通道标记分布分析 =====")
    
    standard_channels = ['fp1', 'f7', 't3', 't5', 'o1', 'f3', 'c3', 'p3', 
                        'fz', 'cz', 'pz', 'fp2', 'f8', 't4', 't6', 'o2', 
                        'f4', 'c4', 'p4']
    external_channels = ['sph-l', 'sph-r']
    all_channels = standard_channels + external_channels
    
    for channel in all_channels:
        if channel in df.columns:
            try:
                count = df[channel].astype(int).sum()
                total = len(df)
                pct = count / total * 100 if total > 0 else 0
                region = get_channel_region(channel)
                print(f"  {channel.upper():4s} ({region:10s}): {count:3d} / {total} ({pct:5.1f}%)")
            except (ValueError, TypeError):
                print(f"  {channel.upper():4s}: 无法统计")


def main():
    parser = argparse.ArgumentParser(description='填充region(A/P)和onset_zone字段')
    parser.add_argument('--input', '-i', type=str, default='converted_manifest.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出CSV文件路径（默认覆盖输入文件）')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='分析通道分布')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果输入路径不是绝对路径，则相对于脚本目录
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    
    if args.output and not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 处理manifest
    df = process_manifest(args.input, args.output)
    
    # 可选：分析通道分布
    if args.analyze:
        analyze_channel_distribution(df)


if __name__ == '__main__':
    main()
