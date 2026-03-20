#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

try:
    from data_preprocess.eeg_pipeline import PipelineConfig
    from models.integration_model import TimeFilter_LaBraM_BrainNetwork_Integration
    from models.train_soz_locator_with_brain_networks import (
        build_soz_datasets,
        collate_fn,
    )
except ImportError:
    from ..data_preprocess.eeg_pipeline import PipelineConfig
    from .integration_model import TimeFilter_LaBraM_BrainNetwork_Integration
    from .train_soz_locator_with_brain_networks import (
        build_soz_datasets,
        collate_fn,
    )


HEMISPHERE_NAMES: Tuple[str, ...] = ("L", "R", "B")
log = logging.getLogger("train_hemisphere_head")


def setup_logging(output_dir: Path) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )


def compute_accuracy(logits: np.ndarray, targets: np.ndarray, ignore_index: int = -100) -> float:
    if len(logits) == 0:
        return 0.0
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(axis=1)
    return float((preds[mask] == targets[mask]).mean())


def compute_class_weight(loader: DataLoader, n_classes: int, ignore_index: int = -100) -> torch.Tensor:
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for batch in loader:
        targets = batch["hemisphere_label"]
        for idx in range(n_classes):
            counts[idx] += float((targets == idx).sum().item())
    total = counts.sum().item()
    weights = torch.zeros_like(counts)
    for idx, count in enumerate(counts.tolist()):
        if count > 0:
            weights[idx] = total / (n_classes * count)
    return weights


def compute_confusion_matrix(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str] = HEMISPHERE_NAMES,
    ignore_index: int = -100,
) -> np.ndarray:
    n_classes = len(class_names)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    if len(logits) == 0:
        return matrix
    preds = logits.argmax(axis=1)
    for pred, target in zip(preds.tolist(), targets.tolist()):
        if target == ignore_index:
            continue
        if 0 <= target < n_classes and 0 <= pred < n_classes:
            matrix[target, pred] += 1
    return matrix


def save_confusion_report(
    logits: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    class_names: Sequence[str] = HEMISPHERE_NAMES,
) -> Tuple[Path, Path]:
    matrix = compute_confusion_matrix(logits, targets, class_names=class_names)
    md_path = output_dir / "hemisphere_confusion_matrix.md"
    csv_path = output_dir / "hemisphere_confusion_matrix.csv"

    lines = [
        "# Hemisphere Confusion Matrix",
        "",
        "| actual \\\\ predicted | " + " | ".join(class_names) + " |",
        "|" + "---|" * (len(class_names) + 1),
    ]
    for row_idx, name in enumerate(class_names):
        values = " | ".join(str(int(v)) for v in matrix[row_idx].tolist())
        lines.append(f"| {name} | {values} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_lines = ["," + ",".join(class_names)]
    for row_idx, name in enumerate(class_names):
        csv_lines.append(name + "," + ",".join(str(int(v)) for v in matrix[row_idx].tolist()))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def freeze_except_hemisphere_head(model: TimeFilter_LaBraM_BrainNetwork_Integration) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hemisphere_head.parameters():
        param.requires_grad = True


def set_model_mode_for_head_training(model: TimeFilter_LaBraM_BrainNetwork_Integration, train_head: bool) -> None:
    model.eval()
    if train_head:
        model.hemisphere_head.train()


def run_epoch(
    model: TimeFilter_LaBraM_BrainNetwork_Integration,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    class_weight: Optional[torch.Tensor],
    desc: str,
) -> Dict[str, object]:
    is_train = optimizer is not None
    set_model_mode_for_head_training(model, train_head=is_train)

    loss_sum = 0.0
    n_batches = 0
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch in iterator:
        x = batch["x"].to(device)
        onset = batch["onset_sec"].to(device)
        start = batch["start_sec"].to(device)
        hemisphere_label = batch["hemisphere_label"].to(device)

        brain_nets = batch.get("brain_nets", None)
        vp_counts = batch.get("valid_patch_counts", None)
        rel_time = batch.get("rel_time", None)
        if brain_nets is not None:
            brain_nets = brain_nets.to(device)
        if vp_counts is not None:
            vp_counts = vp_counts.to(device)
        if rel_time is not None:
            rel_time = rel_time.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(
                x,
                onset,
                start,
                valid_patch_counts=vp_counts,
                brain_networks=brain_nets,
                rel_time=rel_time,
            )
            logits = outputs["hemisphere_logits"]
            valid = hemisphere_label != -100
            if valid.any():
                if class_weight is not None:
                    loss = F.cross_entropy(
                        logits[valid],
                        hemisphere_label[valid],
                        weight=class_weight,
                    )
                else:
                    loss = F.cross_entropy(logits[valid], hemisphere_label[valid])
            else:
                loss = logits.new_zeros(())

        if is_train and valid.any():
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_sum += float(loss.detach().item())
        n_batches += 1
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(hemisphere_label.detach().cpu().numpy())

        iterator.set_postfix(loss=f"{loss.detach().item():.4f}")

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, len(HEMISPHERE_NAMES)))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=np.int64)
    return {
        "loss": loss_sum / max(n_batches, 1),
        "acc": compute_accuracy(logits_np, targets_np),
        "logits": logits_np,
        "targets": targets_np,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 hemisphere-head-only finetuning")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 full model checkpoint, e.g. best_model.pt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tusz-data-root", required=True)
    parser.add_argument("--private-data-root", default=None)
    parser.add_argument("--precomputed-dir", default=None)
    parser.add_argument("--source", choices=["all", "private"], default="private")
    parser.add_argument("--split-strategy", choices=["auto", "random", "private_target"], default="private_target")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--output-mode", choices=["bipolar", "monopolar"], default="monopolar")
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--patch-duration", type=float, default=1.0)
    parser.add_argument("--pre-onset-sec", type=float, default=5.0)
    parser.add_argument("--post-onset-sec", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-class-weight", dest="use_class_weight", action="store_true")
    parser.add_argument("--no-use-class-weight", dest="use_class_weight", action="store_false")
    parser.set_defaults(use_class_weight=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(output_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2, ensure_ascii=False)

    log.info("=== Step 1: Loading datasets ===")
    patch_len = int(args.patch_duration * args.fs)
    n_pre_patches = int(np.ceil(args.pre_onset_sec / args.patch_duration))
    n_post_patches = int(np.ceil(args.post_onset_sec / args.patch_duration))
    n_patches = n_pre_patches + n_post_patches
    pipeline_cfg = PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        n_patches=n_patches,
        patch_len=patch_len,
    )
    train_ds, val_ds, test_ds, split_meta = build_soz_datasets(args=args, pipeline_cfg=pipeline_cfg)
    log.info("  SOZ split strategy: %s", split_meta["strategy"])
    for line in split_meta.get("log_lines", []):
        log.info("  %s", line)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_ds),
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    log.info("=== Step 2: Loading stage-2 model ===")
    model, ckpt = TimeFilter_LaBraM_BrainNetwork_Integration.load_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    model = model.to(device)
    freeze_except_hemisphere_head(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("  Hemisphere-head-only tuning: trainable params=%d/%d", trainable, total)

    class_weight = None
    if args.use_class_weight:
        class_weight = compute_class_weight(train_loader, n_classes=len(HEMISPHERE_NAMES)).to(device)
        log.info("  Hemisphere class weight: %s", class_weight.tolist())

    optimizer = torch.optim.AdamW(
        model.hemisphere_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    best_val_acc = -1.0
    best_ckpt_path = output_dir / "best_model.pt"

    log.info("=== Step 3: Hemisphere-head-only training ===")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            class_weight=class_weight,
            desc=f"hemi-train {epoch + 1}",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            class_weight=None,
            desc="hemi-val",
        )
        scheduler.step()

        log.info(
            "Epoch %03d/%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            args.epochs,
            train_metrics["loss"],
            train_metrics["acc"],
            val_metrics["loss"],
            val_metrics["acc"],
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            model.save_checkpoint(
                str(best_ckpt_path),
                extra={
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
            )
            log.info("  ** New best val_acc=%.4f -> %s", best_val_acc, best_ckpt_path)

    log.info("=== Step 4: Test evaluation ===")
    best_model, best_ckpt = TimeFilter_LaBraM_BrainNetwork_Integration.load_checkpoint(
        str(best_ckpt_path),
        map_location=device,
    )
    best_model = best_model.to(device)
    freeze_except_hemisphere_head(best_model)

    test_metrics = run_epoch(
        model=best_model,
        loader=test_loader,
        device=device,
        optimizer=None,
        scaler=None,
        class_weight=None,
        desc="hemi-test",
    )
    md_path, csv_path = save_confusion_report(
        logits=test_metrics["logits"],
        targets=test_metrics["targets"],
        output_dir=output_dir,
    )

    np.savez(
        str(output_dir / "hemisphere_predictions.npz"),
        logits=test_metrics["logits"],
        targets=test_metrics["targets"],
    )
    report = (
        "# Hemisphere Head Report\n\n"
        f"- Checkpoint: `{args.checkpoint}`\n"
        f"- Source: `{args.source}`\n"
        f"- Split strategy: `{args.split_strategy}`\n"
        f"- Epochs: {args.epochs}\n"
        f"- Best val acc: {best_val_acc:.4f}\n"
        f"- Test acc: {test_metrics['acc']:.4f}\n"
        f"- Confusion markdown: `{md_path.name}`\n"
        f"- Confusion csv: `{csv_path.name}`\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    log.info("Test hemisphere acc: %.4f", test_metrics["acc"])
    log.info("Report saved to %s", output_dir / "report.md")
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
