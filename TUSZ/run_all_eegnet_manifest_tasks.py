#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_TASKS = ["region6", "region9", "hemisphere3"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "combined_manifest.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch launcher for EEGNet manifest-task training"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=DEFAULT_TASKS,
        help="Tasks to run in sequence",
    )
    parser.add_argument(
        "--mode",
        default="full_pipeline",
        choices=["tusz_pretrain", "private_loo_finetune", "full_pipeline"],
        help="Training mode passed to train_eegnet_manifest_tasks.py",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to combined_manifest.csv")
    parser.add_argument("--tusz-data-root", default=r"F:\dataset\TUSZ\v2.0.3\edf")
    parser.add_argument("--private-data-root", default="")
    parser.add_argument("--output-root", required=True, help="Root directory for all task outputs")
    parser.add_argument("--pretrained-ckpt", default="", help="Used when mode=private_loo_finetune")
    parser.add_argument("--pretrained-root", default="", help="Task-wise pretrained checkpoint root; launcher will look for per-task checkpoints under this directory")
    parser.add_argument("--load-backbone-only", action="store_true")
    parser.add_argument("--exclude-montages", default="03_tcp_ar_a")
    parser.add_argument("--min-valid-channels", type=int, default=0)
    parser.add_argument("--soz-only", action="store_true")

    parser.add_argument("--tusz-train-splits", default="train")
    parser.add_argument("--tusz-val-splits", default="dev")
    parser.add_argument("--tusz-test-splits", default="eval")
    parser.add_argument("--private-loo-fold-index", type=int, default=0)
    parser.add_argument("--private-loo-val-offset", type=int, default=1)
    parser.add_argument("--all-loo-folds", action="store_true")

    parser.add_argument("--patch-duration", type=float, default=1.0)
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--pre-onset-sec", type=float, default=5.0)
    parser.add_argument("--post-onset-sec", type=float, default=5.0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrain-patience", type=int, default=8)
    parser.add_argument("--finetune-patience", type=int, default=12)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-decay-patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")

    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--eegnet-f1", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=16)
    parser.add_argument("--eegnet-kernel-length", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel", type=int, default=16)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_task_pretrained_ckpt(args, task: str) -> str:
    if args.pretrained_ckpt:
        return args.pretrained_ckpt
    if not args.pretrained_root:
        return ""

    root = Path(args.pretrained_root)
    candidates = [
        root / task / "stage1_tusz_pretrain" / "best_model.pt",
        root / task / "best_model.pt",
        root / f"{task}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def build_base_command(args, output_dir: Path, task: str) -> List[str]:
    script_path = Path(__file__).resolve().parent / "train_eegnet_manifest_tasks.py"
    pretrained_ckpt = resolve_task_pretrained_ckpt(args, task)
    cmd = [
        sys.executable,
        str(script_path),
        "--task",
        task,
        "--mode",
        args.mode,
        "--manifest",
        args.manifest,
        "--tusz-data-root",
        args.tusz_data_root,
        "--output-dir",
        str(output_dir),
        "--exclude-montages",
        args.exclude_montages,
        "--min-valid-channels",
        str(args.min_valid_channels),
        "--tusz-train-splits",
        args.tusz_train_splits,
        "--tusz-val-splits",
        args.tusz_val_splits,
        "--tusz-test-splits",
        args.tusz_test_splits,
        "--private-loo-fold-index",
        str(args.private_loo_fold_index),
        "--private-loo-val-offset",
        str(args.private_loo_val_offset),
        "--patch-duration",
        str(args.patch_duration),
        "--fs",
        str(args.fs),
        "--pre-onset-sec",
        str(args.pre_onset_sec),
        "--post-onset-sec",
        str(args.post_onset_sec),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--finetune-epochs",
        str(args.finetune_epochs),
        "--pretrain-lr",
        str(args.pretrain_lr),
        "--finetune-lr",
        str(args.finetune_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--pretrain-patience",
        str(args.pretrain_patience),
        "--finetune-patience",
        str(args.finetune_patience),
        "--lr-decay-factor",
        str(args.lr_decay_factor),
        "--lr-decay-patience",
        str(args.lr_decay_patience),
        "--threshold",
        str(args.threshold),
        "--max-pos-weight",
        str(args.max_pos_weight),
        "--seed",
        str(args.seed),
        "--dropout",
        str(args.dropout),
        "--eegnet-f1",
        str(args.eegnet_f1),
        "--eegnet-depth-multiplier",
        str(args.eegnet_depth_multiplier),
        "--eegnet-f2",
        str(args.eegnet_f2),
        "--eegnet-kernel-length",
        str(args.eegnet_kernel_length),
        "--eegnet-separable-kernel",
        str(args.eegnet_separable_kernel),
    ]
    if args.private_data_root:
        cmd.extend(["--private-data-root", args.private_data_root])
    if pretrained_ckpt:
        cmd.extend(["--pretrained-ckpt", pretrained_ckpt])
    if args.load_backbone_only:
        cmd.append("--load-backbone-only")
    if args.soz_only:
        cmd.append("--soz-only")
    if args.all_loo_folds:
        cmd.append("--all-loo-folds")
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)
        cmd = build_base_command(args, task_output, task)
        print("=" * 80)
        print(f"[RUN] task={task} mode={args.mode}")
        print("[CMD] " + subprocess.list2cmdline(cmd))
        print("[OUT] " + str(task_output))
        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[FAIL] task={task} exit_code={result.returncode}")
            if not args.continue_on_error:
                return int(result.returncode)
        else:
            print(f"[OK] task={task}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
