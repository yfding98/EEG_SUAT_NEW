#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Integration Model vs DeepSOZ 推理时间对比

测量两个模型在相同硬件上的推理延迟（包含 warm-up + 多次重复取中位数）。
"""

import argparse
import time
import sys
import os
from pathlib import Path

import numpy as np
import torch

# ── 路径设置 ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TUSZ" / "models"))
sys.path.insert(0, str(ROOT / "DeepSOZ_new"))

# ── 导入模型 ──
from TUSZ.models.integration_model import (
    TimeFilter_LaBraM_BrainNetwork_Integration,
    IntegrationConfig,
)
from DeepSOZ_new.deepsoz_model import DeepSOZLocator, TransformerLSTM


# =====================================================================
# 工具函数
# =====================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """返回模型可训练参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: torch.nn.Module) -> int:
    """返回模型全部参数总数（含冻结）。"""
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


@torch.no_grad()
def benchmark_fn(fn, warmup: int = 10, repeats: int = 50, device: str = "cpu"):
    """
    对 fn() 做 warmup + repeats 次调用，返回延迟统计 (ms)。
    GPU 模式下使用 CUDA events 精确计时。
    """
    use_cuda = device.startswith("cuda")

    # warm-up
    for _ in range(warmup):
        fn()
        if use_cuda:
            torch.cuda.synchronize()

    times = []
    if use_cuda:
        for _ in range(repeats):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            fn()
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))  # ms
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p95_ms": float(np.percentile(arr, 95)),
    }


# =====================================================================
# 构建模型 & 输入
# =====================================================================

def build_integration_model(device: str):
    """构建 Integration 模型（不加载 LaBraM 预训练权重）。"""
    cfg = IntegrationConfig(
        labram_checkpoint="",   # 不加载权重，纯结构
        n_frozen_layers=0,
    )
    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg).to(device).eval()
    return model, cfg


def make_integration_input(cfg: IntegrationConfig, batch_size: int, device: str):
    """生成 Integration 模型的 dummy 输入。"""
    T = int(cfg.patch_len * (cfg.n_pre_patches + cfg.n_post_patches))  # 2000 samples
    x = torch.randn(batch_size, cfg.n_channels, T, device=device)
    onset = torch.zeros(batch_size, device=device)
    start = torch.full((batch_size,), -cfg.n_pre_patches * cfg.patch_len / cfg.fs, device=device)
    return dict(
        x=x,
        seizure_onset_sec=onset,
        window_start_sec=start,
    )


def build_deepsoz_model(device: str):
    """构建 DeepSOZ Stage-2 模型（ctg_11_8）。"""
    model = DeepSOZLocator(n_channels=19).to(device).eval()
    return model


def make_deepsoz_input(batch_size: int, device: str, n_seizures: int = 1,
                       n_windows: int = 45, n_channels: int = 19, samples: int = 200):
    """
    生成 DeepSOZ 的 dummy 输入。
    官方格式: [B, Nsz, T, C, L]
    """
    x = torch.randn(batch_size, n_seizures, n_windows, n_channels, samples, device=device)
    return x


def build_deepsoz_stage1(device: str):
    """构建 DeepSOZ Stage-1 (TransformerLSTM)。"""
    model = TransformerLSTM(n_channels=19, device=device).to(device).eval()
    return model


# =====================================================================
# 主流程
# =====================================================================

def run_benchmark(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退到 CPU")
        device = "cpu"

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print("=" * 72)
    print(f"  Benchmark: Integration Model vs DeepSOZ  |  device={device}")
    print(f"  warmup={args.warmup}  repeats={args.repeats}")
    print("=" * 72)

    # ── 1. 构建模型 ──
    print("\n[1] 构建模型...")
    integration_model, int_cfg = build_integration_model(device)
    deepsoz_stage2 = build_deepsoz_model(device)
    deepsoz_stage1 = build_deepsoz_stage1(device)

    print(f"  Integration Model  — trainable: {format_params(count_parameters(integration_model))}"
          f"  total: {format_params(count_all_parameters(integration_model))}")
    print(f"  DeepSOZ Stage-1    — trainable: {format_params(count_parameters(deepsoz_stage1))}"
          f"  total: {format_params(count_all_parameters(deepsoz_stage1))}")
    print(f"  DeepSOZ Stage-2    — trainable: {format_params(count_parameters(deepsoz_stage2))}"
          f"  total: {format_params(count_all_parameters(deepsoz_stage2))}")

    # ── 2. 按 batch_size 逐一测试 ──
    results = {}
    for bs in batch_sizes:
        print(f"\n{'─' * 72}")
        print(f"  batch_size = {bs}")
        print(f"{'─' * 72}")

        # --- Integration Model ---
        int_inputs = make_integration_input(int_cfg, bs, device)
        int_fn = lambda: integration_model(**int_inputs)
        try:
            int_stats = benchmark_fn(int_fn, warmup=args.warmup, repeats=args.repeats, device=device)
        except Exception as e:
            print(f"  [Integration] ERROR: {e}")
            int_stats = None

        # --- DeepSOZ Stage-1 ---
        ds_input_s1 = make_deepsoz_input(bs, device, n_seizures=1, n_windows=45)
        ds_fn_s1 = lambda: deepsoz_stage1(ds_input_s1)
        try:
            ds_stats_s1 = benchmark_fn(ds_fn_s1, warmup=args.warmup, repeats=args.repeats, device=device)
        except Exception as e:
            print(f"  [DeepSOZ Stage-1] ERROR: {e}")
            ds_stats_s1 = None

        # --- DeepSOZ Stage-2 ---
        ds_input_s2 = make_deepsoz_input(bs, device, n_seizures=1, n_windows=45)
        ds_fn_s2 = lambda: deepsoz_stage2(ds_input_s2)
        try:
            ds_stats_s2 = benchmark_fn(ds_fn_s2, warmup=args.warmup, repeats=args.repeats, device=device)
        except Exception as e:
            print(f"  [DeepSOZ Stage-2] ERROR: {e}")
            ds_stats_s2 = None

        # --- DeepSOZ 两阶段合计 ---
        if ds_stats_s1 and ds_stats_s2:
            ds_combined = {
                k: ds_stats_s1[k] + ds_stats_s2[k]
                for k in ds_stats_s1
            }
        else:
            ds_combined = None

        results[bs] = {
            "integration": int_stats,
            "deepsoz_stage1": ds_stats_s1,
            "deepsoz_stage2": ds_stats_s2,
            "deepsoz_combined": ds_combined,
        }

        # --- 打印结果 ---
        def _print_stats(name: str, stats: dict | None):
            if stats is None:
                print(f"  {name:30s}  — FAILED")
                return
            print(f"  {name:30s}  median={stats['median_ms']:8.2f}ms  "
                  f"mean={stats['mean_ms']:8.2f}ms  "
                  f"std={stats['std_ms']:6.2f}ms  "
                  f"p95={stats['p95_ms']:8.2f}ms")

        _print_stats("Integration Model", int_stats)
        _print_stats("DeepSOZ Stage-1", ds_stats_s1)
        _print_stats("DeepSOZ Stage-2", ds_stats_s2)
        _print_stats("DeepSOZ (S1+S2 合计)", ds_combined)

        if int_stats and ds_combined:
            ratio = ds_combined["median_ms"] / max(int_stats["median_ms"], 1e-9)
            print(f"\n  → DeepSOZ / Integration = {ratio:.2f}x")

    # ── 3. 汇总表格 ──
    print(f"\n{'=' * 72}")
    print("  Summary (median latency, ms)")
    print(f"{'=' * 72}")
    print(f"  {'BS':>4s}  {'Integration':>14s}  {'DeepSOZ S1':>14s}  {'DeepSOZ S2':>14s}  "
          f"{'DeepSOZ Total':>14s}  {'Ratio':>8s}")
    print(f"  {'─' * 4}  {'─' * 14}  {'─' * 14}  {'─' * 14}  {'─' * 14}  {'─' * 8}")
    for bs in batch_sizes:
        r = results[bs]
        int_med = f"{r['integration']['median_ms']:.2f}" if r['integration'] else "N/A"
        s1_med = f"{r['deepsoz_stage1']['median_ms']:.2f}" if r['deepsoz_stage1'] else "N/A"
        s2_med = f"{r['deepsoz_stage2']['median_ms']:.2f}" if r['deepsoz_stage2'] else "N/A"
        tot_med = f"{r['deepsoz_combined']['median_ms']:.2f}" if r['deepsoz_combined'] else "N/A"
        if r['integration'] and r['deepsoz_combined']:
            ratio = f"{r['deepsoz_combined']['median_ms'] / max(r['integration']['median_ms'], 1e-9):.2f}x"
        else:
            ratio = "N/A"
        print(f"  {bs:>4d}  {int_med:>14s}  {s1_med:>14s}  {s2_med:>14s}  {tot_med:>14s}  {ratio:>8s}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Integration vs DeepSOZ inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cpu 或 cuda (default: auto)")
    parser.add_argument("--batch-sizes", dest="batch_sizes", type=str, default="1,2,4,8",
                        help="逗号分隔的 batch sizes (default: 1,2,4,8)")
    parser.add_argument("--warmup", type=int, default=10, help="warm-up 轮次 (default: 10)")
    parser.add_argument("--repeats", type=int, default=50, help="测量重复次数 (default: 50)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
