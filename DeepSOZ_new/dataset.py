#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ_new 数据集模块

适配 combined_manifest.csv 格式：
  source, patient_id, edf_path, split, duration,
  sz_start, sz_end, sz_duration, n_seizure_events,
  seizure_type, hemisphere, onset_channels, soz_bipolar,
  FP1_F7, F7_T3, T3_T5, T5_O1,       (左链, 4)
  FP2_F8, F8_T4, T4_T6, T6_O2,       (右链, 4)
  FP1_F3, F3_C3, C3_P3, P3_O1,       (左中心, 4)
  FP2_F4, F4_C4, C4_P4, P4_O2,       (右中心, 4)
  A1_T3, T3_C3, C3_CZ, CZ_C4, C4_T4, T4_A2  (中央链, 6)

source 列区分数据来源：
  'tusz'    — TUSZ 公共数据集
  'private' — 私有数据集

支持三种数据源过滤模式：
  source='tusz'    → 仅用 TUSZ
  source='private' → 仅用私有
  source=None      → 混合使用全部

两阶段 Dataset：
  Stage1Dataset — 返回 (buffers, sz_labels) 用于发作检测预训练
  Stage2Dataset — 返回 (buffers, sz_labels, onset_map) 用于 SOZ 定位微调
                  并按官方 szlocLoader 方式以 onset 为中心截取 45 帧
"""

import hashlib
import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 通道定义
# ─────────────────────────────────────────────────────────────────────────────

# TCP 双极导联 22 通道（manifest 中列名，下划线分隔）
TCP_BIPOLAR_COLUMNS = [
    'FP1_F7', 'F7_T3',  'T3_T5',  'T5_O1',    # 左颞链
    'FP2_F8', 'F8_T4',  'T4_T6',  'T6_O2',    # 右颞链
    'FP1_F3', 'F3_C3',  'C3_P3',  'P3_O1',    # 左中心链
    'FP2_F4', 'F4_C4',  'C4_P4',  'P4_O2',    # 右中心链
    'A1_T3',  'T3_C3',  'C3_CZ',  'CZ_C4', 'C4_T4', 'T4_A2',  # 中央链
]

# 显示名称（横杠分隔）
TCP_BIPOLAR_NAMES = [c.replace('_', '-') for c in TCP_BIPOLAR_COLUMNS]

# 单极 19 通道（官方 DeepSOZ 顺序）
STANDARD_19_UPPER = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3',  'C3',  'CZ', 'C4', 'T4',
    'T5',  'P3',  'PZ', 'P4', 'T6',
    'O1',  'O2',
]
OFFICIAL_19_CHANNELS = [c.lower() for c in STANDARD_19_UPPER]

# 通道名标准化映射
CHANNEL_NAME_MAP = {
    'Fp1': 'FP1', 'Fp2': 'FP2',
    'Fz':  'FZ',  'Cz': 'CZ', 'Pz': 'PZ',
    'T7':  'T3',  'T8': 'T4',
    'P7':  'T5',  'P8': 'T6',
}

# 官方 szloc 截取窗口
SZLOC_WIN_LEN = 45


# ───────────────────���─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _norm_ch(name: str) -> str:
    """标准化通道名"""
    upper = name.upper().strip()
    for pre in ('EEG ', 'EEG-', 'REF-', 'EEG '):
        if upper.startswith(pre):
            upper = upper[len(pre):]
    return CHANNEL_NAME_MAP.get(name, CHANNEL_NAME_MAP.get(upper, upper))


def _load_bipolar_soz(row: pd.Series) -> np.ndarray:
    """从 manifest 行解析 22 通道 TCP 双极 SOZ 标签"""
    soz = np.zeros(len(TCP_BIPOLAR_COLUMNS), dtype=np.float32)
    for i, col in enumerate(TCP_BIPOLAR_COLUMNS):
        val = row.get(col, 0)
        if val != '' and not pd.isna(val):
            try:
                soz[i] = float(val)
            except (ValueError, TypeError):
                pass
    return soz


def _load_19ch_soz(row: pd.Series) -> np.ndarray:
    """从 manifest 行解析 19 通道单极 SOZ 标签（官方方式）。
    如果 manifest 中有 fp1~o2 列则用之；否则从双极映射推断。"""
    soz = np.zeros(19, dtype=np.float32)
    # 先尝试直接读取单极列
    for i, chn in enumerate(OFFICIAL_19_CHANNELS):
        val = row.get(chn, '')
        if val != '' and not pd.isna(val):
            try:
                soz[i] = float(val)
            except (ValueError, TypeError):
                pass
    if soz.sum() > 0:
        return soz
    # 若无单极列，从 onset_channels 文本解析
    onset_str = str(row.get('onset_channels', ''))
    if onset_str and onset_str != 'nan':
        for token in onset_str.replace(',', ';').split(';'):
            name = token.strip().upper()
            name = CHANNEL_NAME_MAP.get(name, name)
            if name in STANDARD_19_UPPER:
                soz[STANDARD_19_UPPER.index(name)] = 1.0
    return soz


def _crop_window(X: np.ndarray, Y: np.ndarray,
                 win_len: int = SZLOC_WIN_LEN,
                 rng: Optional[np.random.Generator] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    官方 szlocLoader.__getitem__ 的截取逻辑：
      szt = Y.argmax(1)
      x   = random [1, 15]
      s   = max(0, szt - x - 15)
      e   = s + 45

    参数：
      X : [Nsz, T, C, L]
      Y : [Nsz, T]
    返回：
      X_crop : [Nsz, win_len, C, L]
      Y_crop : [Nsz, win_len]
    """
    if rng is None:
        rng = np.random.default_rng()

    Nsz, T, C, L = X.shape
    X_out = np.zeros((Nsz, win_len, C, L), dtype=X.dtype)
    Y_out = np.zeros((Nsz, win_len),        dtype=Y.dtype)

    for j in range(Nsz):
        szt = int(Y[j].argmax())
        x   = int(rng.integers(1, 16))
        s   = max(0, szt - x - 15)
        e   = s + win_len
        if e > T:
            e = T
            s = max(0, e - win_len)
        actual = e - s
        X_out[j, :actual] = X[j, s:e]
        Y_out[j, :actual] = Y[j, s:e]

    return X_out, Y_out


# ─────────────────────────────────────────────────────────────────────────────
# 离线预处理 npy 数据集（官方方式）
# ─────────────────────────────────────────────────────────────────────────────

class Stage1Dataset(Dataset):
    """
    Stage-1 发作检测数据集（离线 npy 模式，对应官方 pretrainLoader）

    manifest 必要字段：pt_id, fn, loc (X.npy 路径),
                       sz_starts, sz_ends,
                       fp1~o2 (0/1 通道 SOZ 标签)
    """

    def __init__(
        self,
        data_root: str,
        manifest: List[Dict],
        patient_ids: List,
        normalize: bool = True,
        max_seizures: int = 10,
        seed: int = 42,
    ):
        self.data_root   = data_root
        self.normalize   = normalize
        self.max_seizures = max_seizures
        self.rng = np.random.default_rng(seed)

        self.mnlist = [
            m for m in manifest
            if self._pt_id(m) in patient_ids
        ]
        logger.info(f'Stage1Dataset(npy): {len(self.mnlist)} 条记录  '
                    f'患者: {len(set(self._pt_id(m) for m in self.mnlist))}')

    @staticmethod
    def _pt_id(m: Dict):
        v = m.get('pt_id', '')
        try:
            return json.loads(v) if v != '' else v
        except Exception:
            return v

    def __len__(self) -> int:
        return len(self.mnlist)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mn = self.mnlist[idx]
        xloc = self.data_root + mn['loc']
        yloc = xloc.rsplit('.', 1)[0] + '_label.npy'

        X = np.load(xloc)[:self.max_seizures]
        Y = np.load(yloc)[:self.max_seizures]
        soz = _load_19ch_soz(pd.Series(mn))

        if self.normalize:
            X = (X - X.mean()) / (X.std() + 1e-8)

        return {
            'fn':         mn.get('fn', ''),
            'buffers':    torch.tensor(X, dtype=torch.float32),
            'sz_labels':  torch.tensor(Y, dtype=torch.float32),
            'onset_map':  torch.tensor(soz, dtype=torch.float32),
        }


class Stage2Dataset(Dataset):
    """Stage-2 SOZ 定位数据集（离线 npy，含 onset 截取）"""

    def __init__(
        self,
        data_root: str,
        manifest: List[Dict],
        patient_ids: List,
        normalize: bool = True,
        max_seizures: int = 10,
        win_len: int = SZLOC_WIN_LEN,
        seed: int = 42,
    ):
        self.data_root    = data_root
        self.normalize    = normalize
        self.max_seizures = max_seizures
        self.win_len      = win_len
        self.rng = np.random.default_rng(seed)

        self.mnlist = [
            m for m in manifest
            if Stage1Dataset._pt_id(m) in patient_ids
        ]
        logger.info(f'Stage2Dataset(npy): {len(self.mnlist)} 条记录')

    def __len__(self) -> int:
        return len(self.mnlist)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mn = self.mnlist[idx]
        xloc = self.data_root + mn['loc']
        yloc = xloc.rsplit('.', 1)[0] + '_label.npy'

        X = np.load(xloc)[:self.max_seizures]
        Y = np.load(yloc)[:self.max_seizures]
        soz = _load_19ch_soz(pd.Series(mn))

        X, Y = _crop_window(X, Y, self.win_len, self.rng)

        if self.normalize:
            X = (X - X.mean()) / (X.std() + 1e-8)

        return {
            'fn':         mn.get('fn', ''),
            'buffers':    torch.tensor(X, dtype=torch.float32),
            'sz_labels':  torch.tensor(Y, dtype=torch.float32),
            'onset_map':  torch.tensor(soz, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 在线 EDF 数据集（适配 combined_manifest.csv 格式）
# ─────────────────────────────────────────────────────────────────────────────

def _build_edf_sample_list(
    df: pd.DataFrame,
    data_roots: List[str],
    use_bipolar: bool = True,
) -> List[Dict]:
    """
    从 combined_manifest DataFrame 构建在线加载的样本列表。

    每一行对应一次发作事件（已拆分），直接对应一个 sample。
    """
    samples = []
    for _, row in df.iterrows():
        sz_s = row.get('sz_start', None)
        sz_e = row.get('sz_end', None)
        if pd.isna(sz_s) or pd.isna(sz_e):
            continue
        try:
            sz_start = float(sz_s)
            sz_end   = float(sz_e)
        except (ValueError, TypeError):
            continue
        if sz_end <= sz_start:
            continue

        # 查找 EDF 文件
        edf_path = None
        edf_rel = str(row.get('edf_path', ''))
        if edf_rel and edf_rel != 'nan':
            for root in data_roots:
                p = Path(root) / edf_rel
                if p.exists():
                    edf_path = str(p)
                    break
            # 如果相对路径不行，��索文件名
            if edf_path is None:
                fname = Path(edf_rel).name
                for root in data_roots:
                    for f in Path(root).rglob(fname):
                        edf_path = str(f)
                        break
                    if edf_path:
                        break
        if edf_path is None:
            continue

        # SOZ 标签
        if use_bipolar:
            soz = _load_bipolar_soz(row)
        else:
            soz = _load_19ch_soz(row)

        # 记录持续时间（用于计算 preictal 背景）
        duration = row.get('duration', None)
        if pd.isna(duration) or duration is None or duration == '':
            duration = None
        else:
            duration = float(duration)

        samples.append({
            'pt_id':      str(row.get('patient_id', '')),
            'fn':         edf_rel,
            'source':     str(row.get('source', '')),
            'edf_path':   edf_path,
            'sz_start':   sz_start,
            'sz_end':     sz_end,
            'duration':   duration,
            'onset_map':  soz,
        })
    return samples


def _preprocess_edf(
    edf_path: str, sz_start: float, sz_end: float,
    duration: Optional[float],
    n_channels: int = 19,
    use_bipolar: bool = False,
    n_windows: int = 45,
    target_fs: float = 200.0,
    f_low: float = 1.6, f_high: float = 30.0,
    clip_std: float = 2.0,
    preictal_sec: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 EDF 文件在线提取并预处理一次发作数据。

    提取 [sz_start - preictal_sec, sz_end] 的数据段，
    保证窗口中既有发作前背景又有发作过程。

    返回：
      X : [1, n_windows, n_channels, L]   (Nsz=1)
      Y : [1, n_windows]                  (0=背景, 1=发作)
    """
    from scipy import signal as sig
    from scipy.signal import resample

    # ── 读取 EDF ─────────────────────────────────────────────────────
    errors = []
    data, fs, ch_names = None, None, None
    try:
        import pyedflib
        f = pyedflib.EdfReader(edf_path)
        n  = f.signals_in_file
        ch = list(f.getSignalLabels())
        fs_val = f.getSampleFrequency(0)
        data = np.array([f.readSignal(i) for i in range(n)])
        fs, ch_names = float(fs_val), ch
        f._close()
    except Exception as e:
        errors.append(f'pyedflib: {e}')

    if data is None:
        for enc in ('utf-8', 'latin-1'):
            try:
                import mne
                raw = mne.io.read_raw_edf(edf_path, preload=True,
                                          verbose='ERROR', encoding=enc)
                data, fs = raw.get_data(), float(raw.info['sfreq'])
                ch_names = list(raw.ch_names)
                break
            except Exception as e:
                errors.append(f'mne({enc}): {e}')

    if data is None:
        raise RuntimeError(f'无法读取 EDF {edf_path}: {errors}')

    # ── 通道映射 ─────────────────────────────────────────────────────
    ch_map = {_norm_ch(n): i for i, n in enumerate(ch_names)}

    if use_bipolar:
        # TCP 双极导联：差分
        BIPOLAR_PAIRS = [
            ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
            ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
            ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
            ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
            ('A1', 'T3'),  ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
            ('C4', 'T4'),  ('T4', 'A2'),
        ]
        out = np.zeros((len(BIPOLAR_PAIRS), data.shape[1]))
        for i, (a, b) in enumerate(BIPOLAR_PAIRS):
            ia, ib = ch_map.get(a), ch_map.get(b)
            if ia is not None and ib is not None:
                out[i] = data[ia] - data[ib]
        C = len(BIPOLAR_PAIRS)
    else:
        # 单极 19 通道
        out = np.zeros((19, data.shape[1]))
        for i, chn in enumerate(STANDARD_19_UPPER):
            idx = ch_map.get(chn)
            if idx is not None:
                out[i] = data[idx]
        C = 19

    # ── 带通滤波 + 幅值裁剪 ──────────────────────────────────────────
    nyq = fs / 2.0
    b, a = sig.butter(4, [max(f_low / nyq, 1e-4),
                           min(f_high / nyq, 0.999)], btype='bandpass')
    for i in range(C):
        out[i] = sig.filtfilt(b, a, out[i], method='gust')
        m, s   = out[i].mean(), out[i].std()
        if s > 1e-8:
            out[i] = np.clip(out[i], m - clip_std * s, m + clip_std * s)

    # ── 重采样 ────────────────────────────────────────────────────────
    if abs(fs - target_fs) > 0.5:
        n_new = int(out.shape[1] * target_fs / fs)
        from scipy.signal import resample as rs
        out = np.stack([rs(out[i], n_new) for i in range(C)])
        fs  = target_fs

    # ── 提取时间段：preictal + ictal ──────────────────────────────────
    # 取 [sz_start - preictal_sec, sz_end] 这段
    seg_begin = max(0, sz_start - preictal_sec)
    seg_s = int(seg_begin * fs)
    seg_e = int(min(out.shape[1], sz_end * fs))
    seg   = out[:, seg_s:seg_e]   # [C, N]

    # ── 标准化 ────────────────────────────────────────────────────────
    m, s = seg.mean(), seg.std()
    seg = (seg - m) / (s + 1e-8)

    # ── 分割成 1s 窗口 ────────────────────────────────────────────────
    ws  = int(target_fs)     # 1 秒 = 200 采样点
    n_w = seg.shape[1] // ws
    if n_w == 0:
        pad_len = ws - seg.shape[1]
        seg = np.concatenate([seg, np.zeros((C, pad_len))], axis=1)
        n_w = 1

    windows = seg[:, :n_w * ws].reshape(C, n_w, ws).transpose(1, 0, 2)  # [T, C, ws]

    # ── 统一到 n_windows ──────────────────────────────────────────────
    T = n_windows
    if windows.shape[0] < T:
        pad = np.zeros((T - windows.shape[0], C, ws), dtype=windows.dtype)
        windows = np.concatenate([windows, pad], axis=0)
    else:
        windows = windows[:T]

    # ── 构造帧级发作标签 ──────────────────────────────────────────────
    Y = np.zeros(T, dtype=np.float32)
    # 每帧 i 对应的绝对时间: [seg_begin + i, seg_begin + i + 1)
    for i in range(min(T, n_w)):
        frame_start = seg_begin + i
        frame_end   = seg_begin + i + 1
        # 若此帧与发作时间段有交集 → 标为 1
        if frame_end > sz_start and frame_start < sz_end:
            Y[i] = 1.0

    X = windows[np.newaxis]   # [1, T, C, ws]
    Y = Y[np.newaxis]         # [1, T]
    return X.astype(np.float32), Y.astype(np.float32)


def _compute_cache_key(
    edf_path: str, sz_start: float, sz_end: float,
    duration, n_channels: int, use_bipolar: bool,
    n_windows: int, target_fs: float,
    f_low: float, f_high: float,
) -> str:
    """基于预处理参数计算 SHA-256 哈希，用作缓存文件名。"""
    key_parts = (
        edf_path, sz_start, sz_end, duration,
        n_channels, use_bipolar, n_windows,
        target_fs, f_low, f_high,
    )
    key_str = '|'.join(str(p) for p in key_parts)
    return hashlib.sha256(key_str.encode('utf-8')).hexdigest()


def _preprocess_edf_cached(
    edf_path: str, sz_start: float, sz_end: float,
    duration, n_channels: int, use_bipolar: bool,
    n_windows: int, target_fs: float,
    f_low: float, f_high: float,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    带磁盘缓存的 _preprocess_edf 包装。

    cache_dir=None 时直接调用原函数，无任何额外开销。
    启用时将 (X, Y) 缓存为 .npz 文件，后续直接加载。
    """
    if cache_dir is None:
        return _preprocess_edf(
            edf_path, sz_start, sz_end, duration,
            n_channels=n_channels, use_bipolar=use_bipolar,
            n_windows=n_windows, target_fs=target_fs,
            f_low=f_low, f_high=f_high,
        )

    key = _compute_cache_key(
        edf_path, sz_start, sz_end, duration,
        n_channels, use_bipolar, n_windows,
        target_fs, f_low, f_high,
    )
    cache_path = Path(cache_dir) / f'{key}.npz'

    if cache_path.exists():
        data = np.load(cache_path)
        return data['X'], data['Y']

    # 缓存未命中：执行完整预处理
    X, Y = _preprocess_edf(
        edf_path, sz_start, sz_end, duration,
        n_channels=n_channels, use_bipolar=use_bipolar,
        n_windows=n_windows, target_fs=target_fs,
        f_low=f_low, f_high=f_high,
    )

    # 写入缓存（原子写入：先写带唯一后缀的 .tmp 再 rename，多 worker 安全）
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f'.tmp.{os.getpid()}')
    try:
        np.savez(tmp_path, X=X, Y=Y)
        tmp_path.replace(cache_path)
    except OSError:
        # 另一个 worker 已完成写入，忽略
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return X, Y


class OnlineStage1Dataset(Dataset):
    """
    在线 EDF Stage-1 数据集（适配 combined_manifest.csv）

    支持 source 过滤：'tusz' / 'private' / None（全部）
    支持双极 / 单极通道模式
    """

    def __init__(
        self,
        manifest_path: str,
        data_roots: List[str],
        patient_ids: Optional[List[str]] = None,
        source: Optional[str] = None,
        use_bipolar: bool = True,
        n_windows: int = 45,
        target_fs: float = 200.0,
        f_low: float = 1.6,
        f_high: float = 30.0,
        cache_dir: Optional[str] = None,
    ):
        df = pd.read_csv(manifest_path)

        # source 过滤
        if source is not None:
            df = df[df['source'] == source].reset_index(drop=True)

        # 患者过滤
        if patient_ids is not None:
            df = df[df['patient_id'].isin(patient_ids)].reset_index(drop=True)

        self.use_bipolar = use_bipolar
        self.n_channels  = len(TCP_BIPOLAR_COLUMNS) if use_bipolar else 19
        self.samples     = _build_edf_sample_list(df, data_roots, use_bipolar)
        self.n_windows   = n_windows
        self.target_fs   = target_fs
        self.f_low       = f_low
        self.f_high      = f_high
        self.cache_dir   = cache_dir
        logger.info(f'OnlineStage1Dataset: {len(self.samples)} 样本  '
                    f'source={source}  bipolar={use_bipolar}  '
                    f'n_ch={self.n_channels}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        try:
            X, Y = _preprocess_edf_cached(
                s['edf_path'], s['sz_start'], s['sz_end'],
                s['duration'],
                n_channels=self.n_channels,
                use_bipolar=self.use_bipolar,
                n_windows=self.n_windows,
                target_fs=self.target_fs,
                f_low=self.f_low, f_high=self.f_high,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            logger.error(f"加载失败 {s['fn']}: {e}")
            ws = int(self.target_fs)
            X = np.zeros((1, self.n_windows, self.n_channels, ws), np.float32)
            Y = np.zeros((1, self.n_windows), np.float32)

        return {
            'fn':         s['fn'],
            'pt_id':      s['pt_id'],
            'source':     s['source'],
            'buffers':    torch.from_numpy(X),
            'sz_labels':  torch.from_numpy(Y),
            'onset_map':  torch.from_numpy(s['onset_map']),
        }


class OnlineStage2Dataset(OnlineStage1Dataset):
    """在线 EDF Stage-2 数据集（onset 为中心截取 45 帧）"""

    def __init__(self, *args, win_len: int = SZLOC_WIN_LEN,
                 seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.win_len = win_len
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        X = item['buffers'].numpy()    # [1, T, C, L]
        Y = item['sz_labels'].numpy()  # [1, T]
        X, Y = _crop_window(X, Y, self.win_len, self.rng)
        item['buffers']   = torch.from_numpy(X)
        item['sz_labels'] = torch.from_numpy(Y)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# manifest 工具
# ─────────────────────────────────────────────────────────────────────────────

def read_manifest_csv(path: str, delimiter: str = ',') -> List[Dict]:
    """读取 manifest CSV，返回 list of dict"""
    import csv
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def get_patient_ids(
    manifest_path: str,
    source: Optional[str] = None,
) -> List[str]:
    """
    获取 manifest 中的患者 ID 列表。

    Args:
        manifest_path: CSV 路径
        source: 'tusz' / 'private' / None（全部）
    """
    df = pd.read_csv(manifest_path)
    if source is not None and 'source' in df.columns:
        df = df[df['source'] == source]
    return list(df['patient_id'].unique())


def make_kfold_splits(
    patient_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List, List]]:
    """按患者 K 折划分"""
    rng  = np.random.default_rng(seed)
    pids = np.array(patient_ids)
    rng.shuffle(pids)
    splits, fold_sz = [], len(pids) // n_folds
    for k in range(n_folds):
        s = k * fold_sz
        e = s + fold_sz if k < n_folds - 1 else len(pids)
        val   = list(pids[s:e])
        train = list(pids[np.r_[0:s, e:len(pids)]])
        splits.append((train, val))
    return splits


def make_dataloader(dataset: Dataset, batch_size: int = 1,
                    shuffle: bool = True,
                    num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=False)
