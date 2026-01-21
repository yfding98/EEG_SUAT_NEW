"""
Manifest数据验证脚本

验证converted_manifest.csv中每行数据的可用性：
1. 判断EDF文件是否存在，能否加载成功
2. 检查数据是否包含标准19通道以及2个额外通道(Sph-L, Sph-R)
3. 与参考CSV文件进行loc字段匹配（可选）
4. 删除异常行后保存为新的CSV文件

Author: EEG_SUAT_NEW Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import logging
from tqdm import tqdm
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入mne用于EDF读取
try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    logger.warning("mne未安装，将使用pyedflib读取EDF文件")

try:
    import pyedflib
    HAS_PYEDFLIB = True
except ImportError:
    HAS_PYEDFLIB = False


# ==============================================================================
# 通道配置
# ==============================================================================

# 标准19通道
STANDARD_19_CHANNELS = [
    'fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8',
    't3', 'c3', 'cz', 'c4', 't4',
    't5', 'p3', 'pz', 'p4', 't6',
    'o1', 'o2'
]

# 蝶骨电极
EXTRA_CHANNELS = ['sph-l', 'sph-r']

# 蝶骨电极别名
SPHENOIDAL_ALIASES = {
    'sp1': 'sph-l', 'sp-l': 'sph-l', 'spl': 'sph-l', 'sphl': 'sph-l',
    'sp2': 'sph-r', 'sp-r': 'sph-r', 'spr': 'sph-r', 'sphr': 'sph-r',
}

# 标准19通道别名（处理不同命名格式）
CHANNEL_ALIASES = {
    't7': 't3', 't8': 't4',  # 新命名到旧命名
    'p7': 't5', 'p8': 't6',
}


# ==============================================================================
# EDF读取函数
# ==============================================================================

def normalize_channel_name(name: str) -> str:
    """标准化通道名称"""
    name_lower = name.lower().strip()
    # 移除常见前缀/后缀
    name_lower = name_lower.replace('eeg ', '').replace(' eeg', '')
    name_lower = name_lower.replace('-ref', '').replace('-le', '').replace('-ar', '')
    name_lower = name_lower.strip()
    
    # 检查蝶骨电极别名
    if name_lower in SPHENOIDAL_ALIASES:
        return SPHENOIDAL_ALIASES[name_lower]
    
    # 检查标准通道别名
    if name_lower in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name_lower]
    
    return name_lower


def read_edf_channels(edf_path: str) -> Tuple[bool, List[str], str]:
    """
    读取EDF文件并获取通道列表
    
    Args:
        edf_path: EDF文件路径
        
    Returns:
        success: 是否成功读取
        channels: 标准化后的通道名称列表
        error_msg: 错误信息（如果失败）
    """
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            ch_names = raw.ch_names
            raw.close()
        elif HAS_PYEDFLIB:
            f = pyedflib.EdfReader(edf_path)
            ch_names = f.getSignalLabels()
            f.close()
        else:
            return False, [], "没有可用的EDF读取库（需要mne或pyedflib）"
        
        # 标准化通道名称
        normalized_channels = [normalize_channel_name(ch) for ch in ch_names]
        return True, normalized_channels, ""
        
    except Exception as e:
        return False, [], str(e)


def check_channels(channels: List[str]) -> Dict:
    """
    检查通道完整性
    
    Args:
        channels: 标准化后的通道名称列表
        
    Returns:
        结果字典，包含通道检查详情
    """
    channels_set = set(channels)
    
    # 检查标准19通道
    found_19 = []
    missing_19 = []
    for ch in STANDARD_19_CHANNELS:
        if ch in channels_set:
            found_19.append(ch)
        else:
            missing_19.append(ch)
    
    # 检查蝶骨电极
    found_extra = []
    missing_extra = []
    for ch in EXTRA_CHANNELS:
        if ch in channels_set:
            found_extra.append(ch)
        else:
            missing_extra.append(ch)
    
    has_all_19 = len(missing_19) == 0
    has_all_extra = len(missing_extra) == 0
    has_all_21 = has_all_19 and has_all_extra
    
    return {
        'found_19_channels': found_19,
        'missing_19_channels': missing_19,
        'found_extra_channels': found_extra,
        'missing_extra_channels': missing_extra,
        'has_all_19': has_all_19,
        'has_all_extra': has_all_extra,
        'has_all_21': has_all_21,
        'n_found_19': len(found_19),
        'n_found_extra': len(found_extra),
    }


# ==============================================================================
# Loc路径标准化
# ==============================================================================

def normalize_loc(loc: str) -> str:
    """
    标准化loc路径用于匹配比较
    
    Args:
        loc: 原始loc路径
        
    Returns:
        标准化后的loc路径（小写、统一分隔符）
    """
    if pd.isna(loc) or not loc:
        return ''
    loc = str(loc).strip().lower()
    # 统一路径分隔符
    loc = loc.replace('\\', '/').replace('//', '/')
    # 移除开头的 ./ 或 /
    while loc.startswith('./'):
        loc = loc[2:]
    while loc.startswith('/'):
        loc = loc[1:]
    return loc


# ==============================================================================
# 主验证函数
# ==============================================================================

def validate_manifest(manifest_path: str,
                      data_roots: List[str],
                      require_extra_channels: bool = False,
                      reference_csv: Optional[str] = None,
                      output_path: Optional[str] = None) -> pd.DataFrame:
    """
    验证manifest CSV文件中的每行数据
    
    Args:
        manifest_path: manifest CSV文件路径
        data_roots: EDF数据根目录列表
        require_extra_channels: 是否要求必须包含蝶骨电极
        reference_csv: 参考CSV文件路径，用于loc字段匹配筛选
        output_path: 输出CSV路径（仅保留有效行）
        
    Returns:
        验证结果DataFrame
    """
    logger.info(f"读取manifest文件: {manifest_path}")
    df = pd.read_csv(manifest_path)
    logger.info(f"共有 {len(df)} 条记录")
    
    # 加载参考CSV文件（如果提供）
    reference_locs = None
    if reference_csv and Path(reference_csv).exists():
        logger.info(f"读取参考CSV文件: {reference_csv}")
        df_ref = pd.read_csv(reference_csv)
        if 'loc' in df_ref.columns:
            # 标准化参考文件中的loc路径
            reference_locs = set(normalize_loc(loc) for loc in df_ref['loc'] if pd.notna(loc))
            logger.info(f"参考文件中共有 {len(reference_locs)} 个唯一loc路径")
        else:
            logger.warning("参考CSV文件中没有loc列，跳过匹配筛选")
    
    # 验证结果记录
    validation_results = []
    valid_indices = []
    
    # 统计
    stats = {
        'total': len(df),
        'loc_not_matched': 0,
        'file_not_found': 0,
        'load_failed': 0,
        'missing_19_channels': 0,
        'missing_extra_channels': 0,
        'valid_19': 0,
        'valid_21': 0,
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="验证数据"):
        loc = row.get('loc', '')
        pt_id = row.get('pt_id', '')
        fn = row.get('fn', '')
        
        result = {
            'index': idx,
            'pt_id': pt_id,
            'fn': fn,
            'loc': loc,
            'loc_matched': True,
            'file_exists': False,
            'load_success': False,
            'has_all_19': False,
            'has_all_extra': False,
            'has_all_21': False,
            'n_found_19': 0,
            'n_found_extra': 0,
            'missing_channels': '',
            'error_msg': '',
            'is_valid': False,
        }
        
        # 0. 检查loc是否在参考文件中（如果提供了参考文件）
        if reference_locs is not None:
            normalized_loc = normalize_loc(loc)
            if normalized_loc not in reference_locs:
                result['loc_matched'] = False
                result['error_msg'] = 'loc不在参考文件中'
                stats['loc_not_matched'] += 1
                validation_results.append(result)
                continue
        
        # 1. 查找EDF文件
        edf_path = None
        if pd.notna(loc) and loc:
            loc = str(loc).strip()
            
            # 检查是否是绝对路径
            if Path(loc).is_absolute() and Path(loc).exists():
                edf_path = loc
            else:
                # 在数据根目录中查找
                for root in data_roots:
                    full_path = Path(root) / loc
                    if full_path.exists():
                        edf_path = str(full_path)
                        break
        
        if edf_path is None:
            result['error_msg'] = '文件不存在'
            stats['file_not_found'] += 1
            validation_results.append(result)
            continue
        
        result['file_exists'] = True
        
        # 2. 尝试加载EDF文件
        success, channels, error_msg = read_edf_channels(edf_path)
        
        if not success:
            result['error_msg'] = f'加载失败: {error_msg}'
            stats['load_failed'] += 1
            validation_results.append(result)
            continue
        
        result['load_success'] = True
        
        # 3. 检查通道完整性
        channel_check = check_channels(channels)
        result['has_all_19'] = channel_check['has_all_19']
        result['has_all_extra'] = channel_check['has_all_extra']
        result['has_all_21'] = channel_check['has_all_21']
        result['n_found_19'] = channel_check['n_found_19']
        result['n_found_extra'] = channel_check['n_found_extra']
        
        missing_all = channel_check['missing_19_channels'] + channel_check['missing_extra_channels']
        result['missing_channels'] = ','.join(missing_all) if missing_all else ''
        
        if not channel_check['has_all_19']:
            result['error_msg'] = f"缺少标准通道: {channel_check['missing_19_channels']}"
            stats['missing_19_channels'] += 1
            validation_results.append(result)
            continue
        
        stats['valid_19'] += 1
        
        if channel_check['has_all_21']:
            stats['valid_21'] += 1
        elif not channel_check['has_all_extra']:
            stats['missing_extra_channels'] += 1
        
        # 判断是否有效
        if require_extra_channels:
            result['is_valid'] = channel_check['has_all_21']
        else:
            result['is_valid'] = channel_check['has_all_19']
        
        if result['is_valid']:
            valid_indices.append(idx)
        
        validation_results.append(result)
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("验证统计:")
    logger.info(f"  总记录数: {stats['total']}")
    if reference_locs is not None:
        logger.info(f"  loc未匹配: {stats['loc_not_matched']}")
    logger.info(f"  文件不存在: {stats['file_not_found']}")
    logger.info(f"  加载失败: {stats['load_failed']}")
    logger.info(f"  缺少标准19通道: {stats['missing_19_channels']}")
    logger.info(f"  缺少蝶骨电极: {stats['missing_extra_channels']}")
    logger.info(f"  有效(19通道): {stats['valid_19']}")
    logger.info(f"  有效(21通道): {stats['valid_21']}")
    logger.info("=" * 60)
    
    # 保存验证结果
    result_df = pd.DataFrame(validation_results)
    result_path = Path(manifest_path).parent / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(result_path, index=False, encoding='utf-8')
    logger.info(f"验证结果已保存至: {result_path}")
    
    # 保存有效行
    if output_path is None:
        output_path = Path(manifest_path).parent / "converted_manifest_validated.csv"
    
    df_valid = df.iloc[valid_indices].copy()
    df_valid.to_csv(output_path, index=False, encoding='utf')
    logger.info(f"有效数据已保存至: {output_path} (共 {len(df_valid)} 条)")
    
    return result_df


# ==============================================================================
# 命令行入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='验证manifest CSV文件中的EDF数据可用性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本验证
  python validate_manifest.py --data-root E:/DataSet/EEG 
  
  # 要求必须包含蝶骨电极
  python validate_manifest.py --manifest converted_manifest.csv --data-root E:/DataSet/EEG --require-extra
  
  # 使用参考CSV文件进行loc匹配筛选
  python validate_manifest.py --data-root E:/DataSet/EEG --reference detected_ensemble_updated_with_baseline.csv
        """
    )
    
    parser.add_argument('--manifest', '-m', type=str, 
                        default='converted_manifest.csv',
                        help='manifest CSV文件路径')
    parser.add_argument('--data-root', '-d', type=str, nargs='+',
                        default=['E:/DataSet/EEG', 'E:/DataSet/EEG/EEG dataset_SUAT'],
                        help='EDF数据根目录 (可指定多个)')
    parser.add_argument('--reference', '-r', type=str, 
                        default='E:\code_learn\SUAT\workspace\EEG-projects\EEG_SUAT_NEW\DeepSOZ\detected_ensemble_updated_with_baseline.csv',
                        help='参考CSV文件路径，用于loc字段匹配筛选（只保留匹配的行）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出CSV文件路径')
    parser.add_argument('--require-extra', action='store_true',
                        help='要求必须包含蝶骨电极(Sph-L, Sph-R)')
    
    args = parser.parse_args()
    
    # 处理manifest路径
    manifest_path = args.manifest
    if not Path(manifest_path).is_absolute():
        # 相对于脚本目录
        script_dir = Path(__file__).parent
        manifest_path = script_dir / manifest_path
    
    if not Path(manifest_path).exists():
        logger.error(f"Manifest文件不存在: {manifest_path}")
        return
    
    # 处理参考CSV路径
    reference_csv = args.reference
    if reference_csv and not Path(reference_csv).is_absolute():
        script_dir = Path(__file__).parent
        reference_csv = script_dir / reference_csv
    
    # 运行验证
    validate_manifest(
        manifest_path=str(manifest_path),
        data_roots=args.data_root,
        require_extra_channels=args.require_extra,
        reference_csv=str(reference_csv) if reference_csv else None,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
