"""
位置信息补充脚本
遍历数据根目录下的SZ1、SZ2、SZ3、SZ4子目录中的.edf文件，
根据文件路径信息补充converted_manifest.csv中的loc字段
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path


def find_edf_files(data_root, search_patterns=None):
    """
    在数据根目录下搜索所有EDF文件
    
    Args:
        data_root: 数据根目录
        search_patterns: 搜索模式列表，默认搜索SZ1-SZ4子目录
    
    Returns:
        dict: {文件标识符: 文件完整路径}
    """
    if search_patterns is None:
        # 默认搜索模式：SZ1, SZ2, SZ3, SZ4 目录
        search_patterns = [
            r'**/SZ*.edf',
            # '**/sz*.edf',  # 搜索所有以SZ开头的EDF文件，不论层级
            # # 也搜索根目录下直接存放的文件
            # '**/*_SZ*.edf',
            # '**/*_sz*.edf',
        ]
    
    edf_files = {}
    
    for pattern in search_patterns:
        full_pattern = os.path.join(data_root, pattern)
        files = glob.glob(full_pattern, recursive=True)
        
        for filepath in files:
            # 获取相对路径
            rel_path = os.path.relpath(filepath, data_root)
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(filepath))[0]

            # 过滤掉 吴斯龙的 SZ2_0001 这种 文件
            if len( filename)>3:
                continue
            
            # 尝试提取患者名和SZ编号作为标识符
            identifier = extract_identifier(filename, filepath)
            
            if identifier:
                edf_files[identifier] = {
                    'full_path': filepath,
                    'rel_path': rel_path,
                    'filename': filename
                }
    
    return edf_files


def extract_identifier(filename, filepath):
    """
    从文件名或路径中提取患者标识符

    支持的命名格式:
    - 患者名_SZ1, 患者名_SZ2 等 (如 蔡方书_SZ1)
    - SZx.edf (如 SZ1.edf, SZ2.edf)
    - 蔡方书-no/SZ1.edf (路径中包含患者信息)

    Args:
        filename: 文件名（不含扩展名）
        filepath: 完整文件路径

    Returns:
        标识符字符串
    """
    import re

    # # 格式1: 文件名中包含 _SZ 或 _sz (如 蔡方书_SZ1)
    # if '_SZ' in filename.upper():
    #     parts = filename.split('_', 1)  # 只分割第一个下划线
    #     if len(parts) >= 2:
    #         patient_name = parts[0]
    #         sz_part = parts[1].upper()
    #         # 确保第二部分以SZ开头
    #         if sz_part.startswith('SZ'):
    #             return f"{patient_name}_{sz_part}"

    # 格式2: 文件名只有SZ编号 (如 SZ1, SZ2)
    if filename.upper().startswith('SZ') and len(filename) >= 3:
        sz_pattern = r'^[Ss][Zz]\d*'  # 匹配 SZ+数字
        if re.match(sz_pattern, filename):
            # 尝试从路径中提取患者名
            parts = filepath.replace('\\', '/').split('/')
            # 找到SZ文件所在目录名作为患者名
            for i in range(len(parts) - 2, 0, -1):
                dir_name = parts[i]
                # 跳过常见的目录名
                if dir_name in ['SZ1', 'SZ2', 'SZ3', 'SZ4', 'sz1', 'sz2', 'sz3', 'sz4', 'data', 'edf', '']:
                    continue
                # 如果目录名看起来像患者名，使用它
                if dir_name and not re.match(r'^[Ss][Zz]\d*$', dir_name):  # 不是SZ开头
                    return f"{dir_name}_{filename.upper()}"

    # 格式3: 路径中包含患者名/SZx/ (如 蔡方书/SZ1.edf)
    parts = filepath.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part.upper().startswith('SZ') and len(part) <= 3:
            # 找到 SZx 目录
            if i > 0:
                # 假设上一级是患者名
                patient_name = parts[i - 1]
                sz_num = part.upper()
                return f"{patient_name}_{sz_num}"

    # 如果以上都没有匹配，返回原始文件名
    return filename


def match_manifest_entry(row, edf_files):
    """
    根据manifest中的pt_id和fn字段匹配对应的EDF文件
    
    匹配规则：
    - pt_id是患者名称（如"蔡方书"）
    - fn是文件标识（如"蔡方书_SZ1"）
    - 需要匹配的EDF文件：路径中包含患者名，文件名为SZx.edf
    - 例如：pt_id="蔡方书", fn="蔡方书_SZ1" -> 匹配 "新-14例\蔡方书-no\SZ1.edf"

    Args:
        row: manifest中的一行数据
        edf_files: EDF文件字典

    Returns:
        匹配的EDF文件信息或None
    """
    import re
    
    pt_id = row.get('pt_id', None)
    fn_value = row.get('fn', None)

    if pd.isna(pt_id) or pd.isna(fn_value):
        return None

    pt_id_str = str(pt_id).strip()
    fn_str = str(fn_value).strip()

    # 从fn字段提取SZ编号，例如从"蔡方书_SZ1"中提取"SZ1"
    sz_part = None
    
    # 尝试多种模式提取SZ编号
    patterns = [
        r'_?(SZ\d+)',           # 匹配 _SZ1, SZ1 等
        r'_(sz\d+)',            # 小写版本
        r'([Ss][Zz]\d+)',       # 任意大小写
    ]
    
    for pattern in patterns:
        match = re.search(pattern, fn_str, re.IGNORECASE)
        if match:
            sz_part = match.group(1).upper()  # 统一转为大写
            break
    
    if not sz_part:
        print(f"  警告: 无法从fn '{fn_str}' 中提取SZ编号")
        return None

    # 遍历edf_files查找匹配
    # 匹配条件：1) 路径中包含患者名(pt_id)  2) 文件名以SZx开头（不区分大小写）
    best_match = None
    best_score = 0
    
    for identifier, info in edf_files.items():
        rel_path = info['rel_path']
        filename = info['filename']  # 不含扩展名
        
        # 检查路径是否包含患者名
        if pt_id_str not in rel_path:
            continue
        
        # 检查文件名是否匹配SZ编号
        # 支持格式: SZ1, SZ1_0001, SZ1-2 等
        filename_upper = filename.upper()
        
        # 精确匹配: 文件名就是SZ编号（如 SZ1）
        if filename_upper == sz_part:
            return info  # 精确匹配，直接返回
        
        # 文件名以SZ编号开头（如 SZ1_0001, SZ1-2）
        if filename_upper.startswith(sz_part):
            # 检查SZ编号后面是否是分隔符或结尾
            rest = filename_upper[len(sz_part):]
            if not rest or rest[0] in ['_', '-', '.', ' ']:
                # 计算匹配分数，精确匹配得分更高
                score = 100 - len(rest)  # 越短越好
                if score > best_score:
                    best_score = score
                    best_match = info
    
    if best_match:
        return best_match
    
    # 备用匹配：更宽松的规则（路径和文件名中都包含患者名和SZ编号）
    for identifier, info in edf_files.items():
        rel_path_lower = info['rel_path'].lower()
        
        # 检查路径中是否同时包含患者名和SZ编号
        if pt_id_str.lower() in rel_path_lower and sz_part.lower() in rel_path_lower:
            return info

    return None


def fill_locations(input_csv, data_root, output_csv=None, relative_path=True):
    """
    填充manifest CSV中的loc字段

    Args:
        input_csv: 输入CSV文件路径
        data_root: EDF文件所在的数据根目录
        output_csv: 输出CSV文件路径
        relative_path: 是否使用相对路径（相对于data_root）

    Returns:
        处理后的DataFrame
    """
    print(f"正在读取manifest文件: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"正在搜索EDF文件，数据目录: {data_root}")
    edf_files = find_edf_files(data_root)
    print(f"找到 {len(edf_files)} 个EDF文件")

    if len(edf_files) > 0:
        print("\n找到的EDF文件示例:")
        for i, (key, value) in enumerate(list(edf_files.items())[:5]):
            print(f"  {key}: {value['rel_path']}")
        if len(edf_files) > 5:
            print(f"  ... 等共 {len(edf_files)} 个文件")

    # 检查必要的列是否存在
    if 'pt_id' not in df.columns or 'fn' not in df.columns:
        print("错误: CSV文件中缺少pt_id或fn列")
        return df

    # 填充loc字段
    matched_count = 0
    unmatched_entries = []

    for idx, row in df.iterrows():
        matched = match_manifest_entry(row, edf_files)

        if matched:
            if relative_path:
                df.at[idx, 'loc'] = matched['rel_path']
            else:
                df.at[idx, 'loc'] = matched['full_path']
            matched_count += 1
        else:
            unmatched_entries.append(f"pt_id: {row.get('pt_id', 'N/A')}, fn: {row.get('fn', 'N/A')}")

    print(f"\n匹配结果:")
    print(f"  成功匹配: {matched_count} / {len(df)}")
    print(f"  未匹配: {len(df) - matched_count}")

    if unmatched_entries and len(unmatched_entries) <= 10:
        print("\n未匹配的条目:")
        for entry in unmatched_entries:
            print(f"  - {entry}")
    elif unmatched_entries:
        print(f"\n未匹配的条目 (显示前10个，共{len(unmatched_entries)}个):")
        for entry in unmatched_entries[:10]:
            print(f"  - {entry}")

    # 保存结果
    if output_csv is None:
        output_csv = input_csv

    # 删掉 loc为 None 的行
    df = df[df['loc'].notnull()]

    df.to_csv(output_csv, index=False,encoding='utf-8')
    print(f"\n结果已保存至: {output_csv}")

    return df
def scan_data_directory(data_root, verbose=True):
    """
    扫描数据目录结构，输出详细报告
    
    Args:
        data_root: 数据根目录
        verbose: 是否输出详细信息
    
    Returns:
        目录结构信息
    """
    print(f"\n===== 数据目录扫描报告 =====")
    print(f"根目录: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"错误: 目录不存在!")
        return None
    
    # 统计各类型文件
    file_counts = {}
    patient_dirs = set()
    sz_dirs = []
    
    for root, dirs, files in os.walk(data_root):
        rel_root = os.path.relpath(root, data_root)
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            file_counts[ext] = file_counts.get(ext, 0) + 1
            
            if ext == '.edf':
                # 检查是否在SZ目录中
                parts = root.replace('\\', '/').split('/')
                for part in parts:
                    if part.upper().startswith('SZ') and len(part) <= 3:
                        sz_dirs.append(os.path.join(rel_root, f))
        
        # 记录可能的患者目录
        for d in dirs:
            if d.upper().startswith('SZ') and len(d) <= 3:
                patient_dirs.add(rel_root if rel_root != '.' else os.path.basename(root))
    
    print(f"\n文件类型统计:")
    for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext or '无扩展名':15s}: {count} 个文件")
    
    print(f"\n发现 {len(patient_dirs)} 个可能的患者目录")
    
    if sz_dirs:
        print(f"\n在SZ目录中找到 {len(sz_dirs)} 个EDF文件")
        if verbose and len(sz_dirs) <= 20:
            for f in sz_dirs:
                print(f"  - {f}")
        elif verbose:
            print(f"  显示前20个:")
            for f in sz_dirs[:20]:
                print(f"  - {f}")
    
    return {
        'file_counts': file_counts,
        'patient_dirs': list(patient_dirs),
        'sz_edf_files': sz_dirs
    }


def create_file_mapping_csv(data_root, output_csv='file_mapping.csv'):
    """
    创建文件映射CSV，记录所有找到的EDF文件
    
    Args:
        data_root: 数据根目录
        output_csv: 输出CSV路径
    """
    edf_files = find_edf_files(data_root)
    
    rows = []
    for identifier, info in edf_files.items():
        rows.append({
            'identifier': identifier,
            'filename': info['filename'],
            'relative_path': info['rel_path'],
            'full_path': info['full_path']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"文件映射已保存至: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='填充manifest中的EDF文件位置信息')
    parser.add_argument('--input', '-i', type=str, default='converted_manifest.csv',
                        help='输入CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出CSV文件路径')
    parser.add_argument('--data-root', '-d', type=str, required=False,
                        help='EDF文件所在的数据根目录')
    parser.add_argument('--absolute-path', action='store_true',
                        help='使用绝对路径而非相对路径')
    parser.add_argument('--scan', '-s', action='store_true',
                        help='仅扫描数据目录，不填充loc字段')
    parser.add_argument('--create-mapping', action='store_true',
                        help='创建文件映射CSV')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果输入路径不是绝对路径，则相对于脚本目录
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    
    if args.output and not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    # 设置默认数据目录
    if args.data_root is None:
        # 默认数据目录可能的位置
        possible_roots = [
            r'E:\DataSet\EEG\EEG dataset_SUAT',  # 常见的数据集位置
            os.path.join(script_dir, 'data'),    # 脚本目录下的data文件夹
            os.path.join(script_dir, '..', 'data'),  # 上级目录的data文件夹
        ]
        
        for root in possible_roots:
            if os.path.exists(root):
                args.data_root = root
                print(f"自动检测到数据目录: {root}")
                break
        
        if args.data_root is None:
            print("错误: 请使用 --data-root 参数指定EDF文件所在的数据根目录")
            print("例如: python fill_edf_locations.py --data-root E:\\DataSet\\EEG\\EEG_dataset_SUAT")
            return
    
    # 仅扫描模式
    if args.scan:
        scan_data_directory(args.data_root)
        return
    
    # 创建文件映射
    if args.create_mapping:
        mapping_output = os.path.join(script_dir, 'file_mapping.csv')
        create_file_mapping_csv(args.data_root, mapping_output)
        return
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 填充loc字段
    fill_locations(
        args.input, 
        args.data_root, 
        args.output, 
        relative_path=not args.absolute_path
    )


if __name__ == '__main__':
    main()
