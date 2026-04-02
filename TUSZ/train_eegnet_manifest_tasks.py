#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from manifest_dataset import (
    ManifestSOZDataset,
    TCP_COL_NAMES,
    _build_region_target,
    _map_hemisphere_label,
    get_region_names,
)

try:
    from data_preprocess.eeg_pipeline import PipelineConfig
except ImportError:
    PipelineConfig = None


log = logging.getLogger("train_eegnet_tasks")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "combined_manifest.csv"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    region_label_mode: str
    hemisphere_label_mode: str
    class_names: Tuple[str, ...]
    selection_metric: str


def build_task_spec(task: str) -> TaskSpec:
    task = str(task).strip().lower()
    if task == "region6":
        return TaskSpec(
            name="region6",
            task_type="multilabel",
            region_label_mode="coarse",
            hemisphere_label_mode="lrb",
            class_names=tuple(get_region_names("coarse")),
            selection_metric="macro_f1_supported",
        )
    if task == "region9":
        return TaskSpec(
            name="region9",
            task_type="multilabel",
            region_label_mode="fine_lateralized",
            hemisphere_label_mode="lrb",
            class_names=tuple(get_region_names("fine_lateralized")),
            selection_metric="macro_f1_supported",
        )
    if task == "hemisphere3":
        return TaskSpec(
            name="hemisphere3",
            task_type="multiclass",
            region_label_mode="coarse",
            hemisphere_label_mode="lrb",
            class_names=("L", "R", "B"),
            selection_metric="macro_f1_supported",
        )
    raise ValueError(f"Unsupported task: {task}")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(raw: str) -> Optional[List[str]]:
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    return items or None


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def make_pipeline_config(args) -> Optional[object]:
    if PipelineConfig is None:
        return None
    patch_len = int(args.patch_duration * args.fs)
    n_pre_patches = int(np.ceil(args.pre_onset_sec / args.patch_duration))
    n_post_patches = int(np.ceil(args.post_onset_sec / args.patch_duration))
    return PipelineConfig(
        target_fs=args.fs,
        pre_onset_sec=args.pre_onset_sec,
        post_onset_sec=args.post_onset_sec,
        n_patches=n_pre_patches + n_post_patches,
        patch_len=patch_len,
    )


def build_manifest_dataset(
    args,
    task_spec: TaskSpec,
    source_filter: str,
    split_filter: Optional[List[str]] = None,
    patient_ids: Optional[List[str]] = None,
) -> ManifestSOZDataset:
    return ManifestSOZDataset(
        manifest_path=args.manifest,
        tusz_data_root=args.tusz_data_root,
        private_data_root=args.private_data_root or None,
        source_filter=source_filter,
        split_filter=split_filter,
        patient_ids=patient_ids,
        soz_only=args.soz_only,
        label_mode="bipolar",
        region_label_mode=task_spec.region_label_mode,
        hemisphere_label_mode=task_spec.hemisphere_label_mode,
        pipeline_cfg=make_pipeline_config(args),
        exclude_montages=parse_csv_list(args.exclude_montages),
        min_valid_channels=args.min_valid_channels,
    )


class EEGTaskDataset(Dataset):
    def __init__(self, manifest_ds: ManifestSOZDataset, task_spec: TaskSpec):
        self.manifest_ds = manifest_ds
        self.task_spec = task_spec
        self.df = manifest_ds.df.reset_index(drop=True).copy()
        self.targets = self._build_targets()
        cfg = getattr(getattr(manifest_ds, "pipeline", None), "cfg", None)
        n_patches = int(getattr(cfg, "n_patches", 20))
        patch_len = int(getattr(cfg, "patch_len", 100))
        self.input_samples = int(n_patches * patch_len)
        self.n_channels = 22

    def _build_targets(self) -> np.ndarray:
        if self.task_spec.task_type == "multilabel":
            if len(self.df) == 0:
                return np.zeros((0, len(self.task_spec.class_names)), dtype=np.float32)
            bipolar = self.df.reindex(columns=TCP_COL_NAMES, fill_value=0).fillna(0).to_numpy(dtype=np.float32, copy=True)
            onset_series = self.df["onset_channels"] if "onset_channels" in self.df.columns else pd.Series([""] * len(self.df))
            targets = [
                _build_region_target(
                    str(onset),
                    bipolar_row,
                    region_label_mode=self.task_spec.region_label_mode,
                )
                for onset, bipolar_row in zip(onset_series.fillna("").tolist(), bipolar)
            ]
            return np.asarray(targets, dtype=np.float32)

        hemi_series = self.df["hemisphere"] if "hemisphere" in self.df.columns else pd.Series([""] * len(self.df))
        targets = [
            _map_hemisphere_label(str(value), mode=self.task_spec.hemisphere_label_mode)
            for value in hemi_series.fillna("").tolist()
        ]
        return np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.manifest_ds)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        x, _, _, _, _, _, _, _ = self.manifest_ds[idx]
        row = self.df.iloc[idx]
        x = x.reshape(x.shape[0], -1).float()
        target_np = self.targets[idx]
        if self.task_spec.task_type == "multilabel":
            target = torch.from_numpy(np.asarray(target_np, dtype=np.float32))
        else:
            target = torch.tensor(int(target_np), dtype=torch.long)
        return {
            "x": x,
            "target": target,
            "row_idx": int(idx),
            "patient_id": str(row.get("patient_id", "")),
            "edf_path": str(row.get("edf_path", "")),
            "source": str(row.get("source", "")),
            "split": str(row.get("split", "")),
        }


def collate_task_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "row_idx": [int(item["row_idx"]) for item in batch],
        "patient_id": [str(item["patient_id"]) for item in batch],
        "edf_path": [str(item["edf_path"]) for item in batch],
        "source": [str(item["source"]) for item in batch],
        "split": [str(item["split"]) for item in batch],
    }


class EEGNetBackbone(nn.Module):
    def __init__(
        self,
        n_channels: int,
        input_samples: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        separable_kernel: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.input_samples = int(input_samples)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_multiplier, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self.feature_dim = self._infer_feature_dim()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_channels, self.input_samples)
            feat = self._forward_features(dummy)
            return int(feat.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_features(x).reshape(x.shape[0], -1)


class EEGNetTaskModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        input_samples: int,
        n_outputs: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        separable_kernel: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone = EEGNetBackbone(
            n_channels=n_channels,
            input_samples=input_samples,
            f1=f1,
            depth_multiplier=depth_multiplier,
            f2=f2,
            kernel_length=kernel_length,
            separable_kernel=separable_kernel,
            dropout=dropout,
        )
        self.head = nn.Linear(self.backbone.feature_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def compute_binary_rows(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    preds = probs >= threshold
    truth = targets >= 0.5
    for idx, name in enumerate(class_names):
        pred_col = preds[:, idx]
        truth_col = truth[:, idx]
        tp = float(np.logical_and(pred_col == 1, truth_col == 1).sum())
        fp = float(np.logical_and(pred_col == 1, truth_col == 0).sum())
        tn = float(np.logical_and(pred_col == 0, truth_col == 0).sum())
        fn = float(np.logical_and(pred_col == 0, truth_col == 1).sum())
        rows.append(
            {
                "class": str(name),
                "support": int(truth_col.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
                "specificity": safe_div(tn, tn + fp),
                "f1": safe_div(2 * tp, 2 * tp + fp + fn),
                "balanced_acc": 0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)),
            }
        )
    return rows


def compute_multilabel_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> Dict[str, object]:
    if logits.size == 0:
        zeros = np.zeros((0, len(class_names)), dtype=np.float32)
        return {
            "probs": zeros,
            "preds": zeros.astype(np.int64),
            "targets_array": zeros.astype(np.int64),
            "element_acc": 0.0,
            "exact_match": 0.0,
            "micro_f1": 0.0,
            "macro_f1_all": 0.0,
            "macro_f1_supported": 0.0,
            "macro_bal_acc_all": 0.0,
            "macro_bal_acc_supported": 0.0,
            "per_class": [],
        }

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)
    truth = (targets >= 0.5).astype(np.int64)
    rows = compute_binary_rows(probs, truth, class_names, threshold=threshold)
    supported_rows = [row for row in rows if row["support"] > 0]
    element_acc = float((preds == truth).mean())
    exact_match = float(np.all(preds == truth, axis=1).mean())
    tp = float(np.logical_and(preds == 1, truth == 1).sum())
    fp = float(np.logical_and(preds == 1, truth == 0).sum())
    fn = float(np.logical_and(preds == 0, truth == 1).sum())
    micro_f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    macro_f1_all = float(np.mean([row["f1"] for row in rows])) if rows else 0.0
    macro_bal_acc_all = float(np.mean([row["balanced_acc"] for row in rows])) if rows else 0.0
    macro_f1_supported = float(np.mean([row["f1"] for row in supported_rows])) if supported_rows else macro_f1_all
    macro_bal_acc_supported = (
        float(np.mean([row["balanced_acc"] for row in supported_rows])) if supported_rows else macro_bal_acc_all
    )
    return {
        "probs": probs,
        "preds": preds,
        "targets_array": truth,
        "element_acc": element_acc,
        "exact_match": exact_match,
        "micro_f1": micro_f1,
        "macro_f1_all": macro_f1_all,
        "macro_f1_supported": macro_f1_supported,
        "macro_bal_acc_all": macro_bal_acc_all,
        "macro_bal_acc_supported": macro_bal_acc_supported,
        "per_class": rows,
    }


def compute_multiclass_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    ignore_index: int = -100,
) -> Dict[str, object]:
    n_classes = len(class_names)
    if logits.size == 0:
        return {
            "probs": np.zeros((0, n_classes), dtype=np.float32),
            "preds": np.zeros((0,), dtype=np.int64),
            "targets_array": np.zeros((0,), dtype=np.int64),
            "accuracy": 0.0,
            "macro_f1_all": 0.0,
            "macro_f1_supported": 0.0,
            "confusion_matrix": np.zeros((n_classes, n_classes), dtype=np.int64),
            "per_class": [],
        }

    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    mask = targets != ignore_index
    logits_valid = logits[mask]
    targets_valid = targets[mask]
    if logits_valid.shape[0] == 0:
        return {
            "probs": np.zeros((0, n_classes), dtype=np.float32),
            "preds": np.zeros((0,), dtype=np.int64),
            "targets_array": np.zeros((0,), dtype=np.int64),
            "accuracy": 0.0,
            "macro_f1_all": 0.0,
            "macro_f1_supported": 0.0,
            "confusion_matrix": np.zeros((n_classes, n_classes), dtype=np.int64),
            "per_class": [],
        }

    shifted = logits_valid - logits_valid.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    preds = probs.argmax(axis=1)
    acc = float((preds == targets_valid).mean())
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(targets_valid.tolist(), preds.tolist()):
        if 0 <= true_idx < n_classes and 0 <= pred_idx < n_classes:
            confusion[true_idx, pred_idx] += 1

    rows: List[Dict[str, float]] = []
    for class_idx, class_name in enumerate(class_names):
        tp = float(confusion[class_idx, class_idx])
        fp = float(confusion[:, class_idx].sum() - tp)
        fn = float(confusion[class_idx, :].sum() - tp)
        support = int(confusion[class_idx, :].sum())
        rows.append(
            {
                "class": str(class_name),
                "support": support,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
                "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            }
        )

    supported_rows = [row for row in rows if row["support"] > 0]
    macro_f1_all = float(np.mean([row["f1"] for row in rows])) if rows else 0.0
    macro_f1_supported = float(np.mean([row["f1"] for row in supported_rows])) if supported_rows else macro_f1_all
    return {
        "probs": probs,
        "preds": preds,
        "targets_array": targets_valid,
        "accuracy": acc,
        "macro_f1_all": macro_f1_all,
        "macro_f1_supported": macro_f1_supported,
        "confusion_matrix": confusion,
        "per_class": rows,
    }


def build_loss(
    task_spec: TaskSpec,
    train_targets: np.ndarray,
    device: torch.device,
    max_pos_weight: float,
) -> Tuple[nn.Module, Dict[str, object]]:
    if task_spec.task_type == "multilabel":
        pos = train_targets.sum(axis=0).astype(np.float32)
        total = float(train_targets.shape[0])
        neg = total - pos
        pos_weight = np.divide(neg, np.clip(pos, 1.0, None))
        pos_weight = np.clip(pos_weight, 1.0, float(max_pos_weight)).astype(np.float32)
        pos_weight[pos <= 0] = 1.0
        summary = {
            "supports": {name: int(count) for name, count in zip(task_spec.class_names, pos.tolist())},
            "pos_weight": {name: float(weight) for name, weight in zip(task_spec.class_names, pos_weight.tolist())},
        }
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device)), summary

    valid = train_targets[train_targets >= 0]
    counts = np.bincount(valid, minlength=len(task_spec.class_names)).astype(np.float32)
    total = float(np.maximum(counts.sum(), 1.0))
    weights = total / np.clip(counts, 1.0, None)
    weights = weights / np.maximum(weights.mean(), 1e-8)
    summary = {
        "supports": {name: int(count) for name, count in zip(task_spec.class_names, counts.tolist())},
        "class_weight": {name: float(weight) for name, weight in zip(task_spec.class_names, weights.tolist())},
    }
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), ignore_index=-100), summary


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_spec: TaskSpec,
    threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    logits_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    meta = {
        "row_idx": [],
        "patient_id": [],
        "edf_path": [],
        "source": [],
        "split": [],
    }

    for batch in loader:
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            if task_spec.task_type == "multilabel":
                loss = criterion(logits, target.float())
            else:
                loss = criterion(logits, target.long())
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        logits_chunks.append(logits.detach().cpu().numpy())
        target_chunks.append(target.detach().cpu().numpy())
        meta["row_idx"].extend(batch["row_idx"])
        meta["patient_id"].extend(batch["patient_id"])
        meta["edf_path"].extend(batch["edf_path"])
        meta["source"].extend(batch["source"])
        meta["split"].extend(batch["split"])

    if logits_chunks:
        logits_np = np.concatenate(logits_chunks, axis=0)
    else:
        logits_np = np.zeros((0, len(task_spec.class_names)), dtype=np.float32)
    if target_chunks:
        targets_np = np.concatenate(target_chunks, axis=0)
    else:
        targets_np = np.zeros((0, len(task_spec.class_names)), dtype=np.float32)

    if task_spec.task_type == "multilabel":
        metrics = compute_multilabel_metrics(logits_np, targets_np, task_spec.class_names, threshold=threshold)
    else:
        metrics = compute_multiclass_metrics(logits_np, targets_np, task_spec.class_names)
    metrics["loss"] = safe_div(total_loss, total_items)
    metrics["metadata"] = meta
    metrics["logits"] = logits_np
    return metrics


def scalar_summary(metrics: Dict[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            result[key] = float(value)
    return result


def build_private_loo_split(
    patient_ids: Sequence[str],
    fold_index: int,
    val_offset: int,
) -> Dict[str, object]:
    ordered = sorted(str(patient_id) for patient_id in patient_ids)
    n_patients = len(ordered)
    if n_patients < 3:
        raise ValueError(f"private_loo requires at least 3 private patients, got {n_patients}")
    test_idx = int(fold_index) % n_patients
    if int(val_offset) <= 0:
        raise ValueError(f"private_loo requires val_offset >= 1, got {val_offset}")
    val_idx = (test_idx + int(val_offset)) % n_patients
    if val_idx == test_idx:
        val_idx = (test_idx + 1) % n_patients
    test_patient = ordered[test_idx]
    val_patient = ordered[val_idx]
    train_patients = [patient_id for patient_id in ordered if patient_id not in {test_patient, val_patient}]
    return {
        "train": train_patients,
        "val": [val_patient],
        "test": [test_patient],
        "fold_index": int(test_idx),
        "n_folds": int(n_patients),
        "val_offset": int(val_offset),
    }


def build_tusz_datasets(args, task_spec: TaskSpec) -> Tuple[EEGTaskDataset, EEGTaskDataset, EEGTaskDataset, Dict[str, object]]:
    train_manifest = build_manifest_dataset(args, task_spec, source_filter="tusz", split_filter=parse_csv_list(args.tusz_train_splits))
    val_manifest = build_manifest_dataset(args, task_spec, source_filter="tusz", split_filter=parse_csv_list(args.tusz_val_splits))
    test_manifest = build_manifest_dataset(args, task_spec, source_filter="tusz", split_filter=parse_csv_list(args.tusz_test_splits))
    split_info = {
        "source": "tusz",
        "train_rows": int(len(train_manifest)),
        "val_rows": int(len(val_manifest)),
        "test_rows": int(len(test_manifest)),
        "train_splits": parse_csv_list(args.tusz_train_splits),
        "val_splits": parse_csv_list(args.tusz_val_splits),
        "test_splits": parse_csv_list(args.tusz_test_splits),
    }
    return EEGTaskDataset(train_manifest, task_spec), EEGTaskDataset(val_manifest, task_spec), EEGTaskDataset(test_manifest, task_spec), split_info


def collect_private_patient_ids(args, task_spec: TaskSpec) -> List[str]:
    manifest_ds = build_manifest_dataset(args, task_spec, source_filter="private", split_filter=None)
    return sorted(str(patient_id) for patient_id in manifest_ds.df["patient_id"].dropna().astype(str).unique().tolist())


def build_private_loo_datasets(
    args,
    task_spec: TaskSpec,
    fold_index: int,
) -> Tuple[EEGTaskDataset, EEGTaskDataset, EEGTaskDataset, Dict[str, object]]:
    patient_ids = collect_private_patient_ids(args, task_spec)
    split = build_private_loo_split(patient_ids, fold_index=fold_index, val_offset=args.private_loo_val_offset)
    train_manifest = build_manifest_dataset(args, task_spec, source_filter="private", split_filter=None, patient_ids=split["train"])
    val_manifest = build_manifest_dataset(args, task_spec, source_filter="private", split_filter=None, patient_ids=split["val"])
    test_manifest = build_manifest_dataset(args, task_spec, source_filter="private", split_filter=None, patient_ids=split["test"])
    split_info = {
        "source": "private",
        "split_strategy": "private_loo",
        **split,
        "train_rows": int(len(train_manifest)),
        "val_rows": int(len(val_manifest)),
        "test_rows": int(len(test_manifest)),
    }
    return EEGTaskDataset(train_manifest, task_spec), EEGTaskDataset(val_manifest, task_spec), EEGTaskDataset(test_manifest, task_spec), split_info


def create_model(args, task_spec: TaskSpec, input_samples: int) -> EEGNetTaskModel:
    return EEGNetTaskModel(
        n_channels=22,
        input_samples=input_samples,
        n_outputs=len(task_spec.class_names),
        f1=args.eegnet_f1,
        depth_multiplier=args.eegnet_depth_multiplier,
        f2=args.eegnet_f2,
        kernel_length=args.eegnet_kernel_length,
        separable_kernel=args.eegnet_separable_kernel,
        dropout=args.dropout,
    )


def load_checkpoint(
    model: EEGNetTaskModel,
    ckpt_path: str,
    load_head: bool,
    device: torch.device,
) -> Dict[str, List[str]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not load_head and "backbone_state" in ckpt:
        missing, unexpected = model.backbone.load_state_dict(ckpt["backbone_state"], strict=False)
        return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    state = ckpt.get("model_state", ckpt)
    if not load_head:
        state = {key: value for key, value in state.items() if not key.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_history(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    pd.DataFrame(history).to_csv(path, index=False)


def save_per_class_metrics(metrics: Dict[str, object], path: Path) -> None:
    rows = metrics.get("per_class") or []
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def save_confusion_matrix(metrics: Dict[str, object], class_names: Sequence[str], path: Path) -> None:
    confusion = metrics.get("confusion_matrix")
    if confusion is None:
        return
    df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    df.to_csv(path)


def save_predictions_csv(
    task_spec: TaskSpec,
    metrics: Dict[str, object],
    path: Path,
) -> None:
    meta = metrics["metadata"]
    probs = metrics["probs"]
    preds = metrics["preds"]
    targets = metrics["targets_array"]
    rows: List[Dict[str, object]] = []
    if task_spec.task_type == "multilabel":
        for idx in range(probs.shape[0]):
            row = {
                "row_idx": int(meta["row_idx"][idx]),
                "patient_id": meta["patient_id"][idx],
                "edf_path": meta["edf_path"][idx],
                "source": meta["source"][idx],
                "split": meta["split"][idx],
                "true_labels": ";".join(name for j, name in enumerate(task_spec.class_names) if int(targets[idx, j]) == 1),
                "pred_labels": ";".join(name for j, name in enumerate(task_spec.class_names) if int(preds[idx, j]) == 1),
            }
            for class_idx, class_name in enumerate(task_spec.class_names):
                row[f"true_{class_name}"] = int(targets[idx, class_idx])
                row[f"pred_{class_name}"] = int(preds[idx, class_idx])
                row[f"prob_{class_name}"] = float(probs[idx, class_idx])
            rows.append(row)
    else:
        for idx in range(probs.shape[0]):
            pred_idx = int(preds[idx])
            true_idx = int(targets[idx])
            row = {
                "row_idx": int(meta["row_idx"][idx]),
                "patient_id": meta["patient_id"][idx],
                "edf_path": meta["edf_path"][idx],
                "source": meta["source"][idx],
                "split": meta["split"][idx],
                "true_index": true_idx,
                "true_label": task_spec.class_names[true_idx] if 0 <= true_idx < len(task_spec.class_names) else "IGNORE",
                "pred_index": pred_idx,
                "pred_label": task_spec.class_names[pred_idx],
            }
            for class_idx, class_name in enumerate(task_spec.class_names):
                row[f"prob_{class_name}"] = float(probs[idx, class_idx])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_report_text(
    args,
    task_spec: TaskSpec,
    stage_name: str,
    split_info: Dict[str, object],
    best_epoch: int,
    train_metrics: Dict[str, object],
    val_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
) -> str:
    lines = [
        "# EEGNet Task Report",
        "",
        f"- Task: `{task_spec.name}`",
        f"- Stage: `{stage_name}`",
        f"- Selection metric: `{task_spec.selection_metric}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Output dir: `{args.output_dir}`",
        "",
        "## Split",
        "",
    ]
    for key, value in split_info.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Summary", "", "| Split | Metric | Value |", "|------|--------|-------|"])

    def append_scalar_rows(split_name: str, metrics: Dict[str, object]) -> None:
        for key, value in scalar_summary(metrics).items():
            lines.append(f"| {split_name} | {key} | {float(value):.6f} |")

    append_scalar_rows("train", train_metrics)
    append_scalar_rows("val", val_metrics)
    append_scalar_rows("test", test_metrics)

    lines.extend(["", "## Test Per-Class", "", "| Class | Support | Precision | Recall | F1 |"])
    lines.append("|-------|---------|-----------|--------|----|")
    for row in test_metrics.get("per_class", []):
        lines.append(
            f"| {row['class']} | {row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )

    if task_spec.task_type == "multiclass" and test_metrics.get("confusion_matrix") is not None:
        confusion = test_metrics["confusion_matrix"]
        lines.extend(["", "## Test Confusion Matrix", ""])
        header = "| true \\ pred | " + " | ".join(task_spec.class_names) + " |"
        sep = "|---------------|" + "|".join(["---"] * len(task_spec.class_names)) + "|"
        lines.append(header)
        lines.append(sep)
        for row_idx, class_name in enumerate(task_spec.class_names):
            row_values = " | ".join(str(int(x)) for x in confusion[row_idx].tolist())
            lines.append(f"| {class_name} | {row_values} |")

    lines.append("")
    return "\n".join(lines)


def train_stage(
    args,
    task_spec: TaskSpec,
    stage_name: str,
    output_dir: Path,
    train_ds: EEGTaskDataset,
    val_ds: EEGTaskDataset,
    test_ds: EEGTaskDataset,
    split_info: Dict[str, object],
    epochs: int,
    lr: float,
    patience: int,
    pretrained_ckpt: str = "",
    load_head: bool = True,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(args, task_spec, input_samples=train_ds.input_samples).to(device)
    if pretrained_ckpt:
        load_info = load_checkpoint(model, pretrained_ckpt, load_head=load_head, device=device)
        log.info(
            "Loaded checkpoint %s (missing=%d unexpected=%d)",
            pretrained_ckpt,
            len(load_info["missing_keys"]),
            len(load_info["unexpected_keys"]),
        )

    criterion, loss_summary = build_loss(
        task_spec,
        train_targets=train_ds.targets,
        device=device,
        max_pos_weight=args.max_pos_weight,
    )
    write_json(output_dir / "loss_setup.json", loss_summary)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_task_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_task_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_task_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_decay_factor,
        patience=max(args.lr_decay_patience, 1),
    )

    best_score = float("-inf")
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    best_ckpt_path = output_dir / "best_model.pt"

    for epoch in range(int(epochs)):
        train_metrics = run_epoch(model, train_loader, criterion, device, task_spec, threshold=args.threshold, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, task_spec, threshold=args.threshold, optimizer=None)
        score = float(val_metrics.get(task_spec.selection_metric, 0.0))
        scheduler.step(score)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_row = {
            "epoch": float(epoch),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
        }
        for key, value in scalar_summary(train_metrics).items():
            history_row[f"train_{key}"] = float(value)
        for key, value in scalar_summary(val_metrics).items():
            history_row[f"val_{key}"] = float(value)
        history.append(history_row)
        log.info(
            "[%s] epoch %03d/%03d lr=%.6g train_loss=%.4f val_loss=%.4f val_%s=%.4f",
            stage_name,
            epoch + 1,
            epochs,
            current_lr,
            float(train_metrics["loss"]),
            float(val_metrics["loss"]),
            task_spec.selection_metric,
            score,
        )

        if score > best_score:
            best_score = score
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone_state": model.backbone.state_dict(),
                    "task": task_spec.name,
                    "class_names": list(task_spec.class_names),
                    "epoch": best_epoch,
                    "selection_metric": task_spec.selection_metric,
                    "best_score": best_score,
                    "split_info": split_info,
                },
                str(best_ckpt_path),
            )
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= int(patience):
                log.info("[%s] early stop triggered at epoch %03d", stage_name, epoch + 1)
                break

    save_history(history, output_dir / "history.csv")
    ckpt = torch.load(str(best_ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    with torch.no_grad():
        train_final = run_epoch(model, train_loader, criterion, device, task_spec, threshold=args.threshold, optimizer=None)
        val_final = run_epoch(model, val_loader, criterion, device, task_spec, threshold=args.threshold, optimizer=None)
        test_final = run_epoch(model, test_loader, criterion, device, task_spec, threshold=args.threshold, optimizer=None)

    metrics_payload = {
        "stage": stage_name,
        "task": task_spec.name,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selection_metric": task_spec.selection_metric,
        "train": scalar_summary(train_final),
        "val": scalar_summary(val_final),
        "test": scalar_summary(test_final),
        "split_info": split_info,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    save_per_class_metrics(test_final, output_dir / "test_per_class_metrics.csv")
    if task_spec.task_type == "multiclass":
        save_confusion_matrix(test_final, task_spec.class_names, output_dir / "test_confusion_matrix.csv")
    save_predictions_csv(task_spec, test_final, output_dir / "test_predictions.csv")
    report = build_report_text(args, task_spec, stage_name, split_info, best_epoch, train_final, val_final, test_final)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "best_checkpoint": str(best_ckpt_path),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "train_metrics": train_final,
        "val_metrics": val_final,
        "test_metrics": test_final,
        "split_info": split_info,
    }


def run_tusz_pretrain(args, task_spec: TaskSpec, output_root: Path) -> Dict[str, object]:
    train_ds, val_ds, test_ds, split_info = build_tusz_datasets(args, task_spec)
    return train_stage(
        args=args,
        task_spec=task_spec,
        stage_name="tusz_pretrain",
        output_dir=output_root,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        split_info=split_info,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        patience=args.pretrain_patience,
        pretrained_ckpt="",
        load_head=True,
    )


def run_private_loo_finetune(args, task_spec: TaskSpec, output_root: Path, pretrained_ckpt: str) -> Dict[str, object]:
    patient_ids = collect_private_patient_ids(args, task_spec)
    fold_indices = list(range(len(patient_ids))) if args.all_loo_folds else [int(args.private_loo_fold_index)]
    summary_rows: List[Dict[str, object]] = []
    last_result: Dict[str, object] = {}

    for fold_index in fold_indices:
        train_ds, val_ds, test_ds, split_info = build_private_loo_datasets(args, task_spec, fold_index=fold_index)
        fold_dir = output_root / f"fold_{int(split_info['fold_index']):03d}_test_{split_info['test'][0]}_val_{split_info['val'][0]}"
        result = train_stage(
            args=args,
            task_spec=task_spec,
            stage_name="private_loo_finetune",
            output_dir=fold_dir,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            split_info=split_info,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            patience=args.finetune_patience,
            pretrained_ckpt=pretrained_ckpt,
            load_head=not args.load_backbone_only,
        )
        test_scalars = scalar_summary(result["test_metrics"])
        val_scalars = scalar_summary(result["val_metrics"])
        summary_row = {
            "fold_index": int(split_info["fold_index"]),
            "test_patient": split_info["test"][0],
            "val_patient": split_info["val"][0],
            "n_train_patients": int(len(split_info["train"])),
            "train_rows": int(split_info["train_rows"]),
            "val_rows": int(split_info["val_rows"]),
            "test_rows": int(split_info["test_rows"]),
            "best_epoch": int(result["best_epoch"]),
            "best_score": float(result["best_score"]),
        }
        for key, value in val_scalars.items():
            summary_row[f"val_{key}"] = float(value)
        for key, value in test_scalars.items():
            summary_row[f"test_{key}"] = float(value)
        summary_rows.append(summary_row)
        last_result = result

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_root / "loo_summary.csv", index=False)
        numeric_cols = [col for col in summary_df.columns if pd.api.types.is_numeric_dtype(summary_df[col])]
        aggregate = {col: float(summary_df[col].mean()) for col in numeric_cols}
        write_json(output_root / "loo_summary_mean.json", aggregate)
    return last_result


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet baselines on combined_manifest tasks")
    parser.add_argument("--task", required=True, choices=["region6", "region9", "hemisphere3"])
    parser.add_argument("--mode", default="full_pipeline", choices=["tusz_pretrain", "private_loo_finetune", "full_pipeline"])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to combined_manifest.csv")
    parser.add_argument("--tusz-data-root", default=r"F:\dataset\TUSZ\v2.0.3\edf")
    parser.add_argument("--private-data-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-ckpt", default="")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    task_spec = build_task_spec(args.task)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", vars(args))

    if PipelineConfig is None:
        log.warning("PipelineConfig is not available in this environment. If EEGPipeline is also missing, ManifestSOZDataset may fall back to zero EEG tensors.")

    if args.mode == "tusz_pretrain":
        result = run_tusz_pretrain(args, task_spec, output_root)
        log.info("Done. Best checkpoint: %s", result["best_checkpoint"])
        return 0

    if args.mode == "private_loo_finetune":
        if not args.pretrained_ckpt:
            log.warning("No --pretrained-ckpt provided; private LOO finetune will start from scratch.")
        run_private_loo_finetune(args, task_spec, output_root, pretrained_ckpt=args.pretrained_ckpt)
        log.info("Done.")
        return 0

    pretrain_dir = output_root / "stage1_tusz_pretrain"
    pretrain_result = run_tusz_pretrain(args, task_spec, pretrain_dir)
    stage2_dir = output_root / "stage2_private_loo"
    run_private_loo_finetune(args, task_spec, stage2_dir, pretrained_ckpt=pretrain_result["best_checkpoint"])
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
