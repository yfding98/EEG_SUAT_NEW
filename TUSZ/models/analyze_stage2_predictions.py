#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


FEATURE_NAMES: Tuple[str, ...] = ("gc", "te", "aec", "wpli")
REGION_NAMES: Tuple[str, ...] = ("FP", "F", "C", "T", "P", "O")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _binary_stats(pred: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    pred_i = pred.astype(np.int64)
    gold_i = gold.astype(np.int64)
    tp = int(np.logical_and(pred_i == 1, gold_i == 1).sum())
    fp = int(np.logical_and(pred_i == 1, gold_i == 0).sum())
    tn = int(np.logical_and(pred_i == 0, gold_i == 0).sum())
    fn = int(np.logical_and(pred_i == 0, gold_i == 1).sum())
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    bal_acc = 0.5 * (recall + specificity)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal_acc,
    }


def summarize_gate_weights(gate_weights: np.ndarray) -> Dict[str, float]:
    gw = np.asarray(gate_weights, dtype=np.float32).reshape(-1)
    return {
        "mean": float(gw.mean()),
        "std": float(gw.std()),
        "min": float(gw.min()),
        "max": float(gw.max()),
        "q10": float(np.quantile(gw, 0.10)),
        "q50": float(np.quantile(gw, 0.50)),
        "q90": float(np.quantile(gw, 0.90)),
        "temporal_dominant_ratio": float((gw > 0.6).mean()),
        "balanced_ratio": float(((gw >= 0.4) & (gw <= 0.6)).mean()),
        "network_dominant_ratio": float((gw < 0.4).mean()),
    }


def summarize_branch_weights(
    branch_weights: np.ndarray,
    valid_patch_counts: np.ndarray | None = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> List[Dict[str, float]]:
    bw = np.asarray(branch_weights, dtype=np.float32)
    if bw.ndim != 3 or bw.shape[-1] != len(feature_names):
        raise ValueError(f"Expected branch_weights with shape [B,P,{len(feature_names)}], got {bw.shape}")

    if valid_patch_counts is not None:
        counts = np.asarray(valid_patch_counts).reshape(-1)
        mask = np.arange(bw.shape[1])[None, :] < counts[:, None]
        valid_bw = bw[mask]
    else:
        valid_bw = bw.reshape(-1, bw.shape[-1])

    rows: List[Dict[str, float]] = []
    for idx, name in enumerate(feature_names):
        col = valid_bw[:, idx]
        rows.append(
            {
                "feature": name,
                "mean": float(col.mean()),
                "std": float(col.std()),
                "q25": float(np.quantile(col, 0.25)),
                "q50": float(np.quantile(col, 0.50)),
                "q75": float(np.quantile(col, 0.75)),
            }
        )
    rows.sort(key=lambda row: row["mean"], reverse=True)
    return rows


def search_region_thresholds(
    region_probs: np.ndarray,
    region_targets: np.ndarray,
    objective: str = "balanced_accuracy",
    region_names: Sequence[str] = REGION_NAMES,
) -> List[Dict[str, float]]:
    probs = np.asarray(region_probs, dtype=np.float32)
    targets = (np.asarray(region_targets, dtype=np.float32) >= 0.5).astype(np.int64)
    thresholds = np.arange(0.05, 1.00, 0.05)
    rows: List[Dict[str, float]] = []
    for idx, name in enumerate(region_names):
        gold = targets[:, idx]
        best = None
        for thr in thresholds:
            pred = probs[:, idx] >= thr
            stats = _binary_stats(pred, gold)
            score = stats[objective]
            if best is None or score > best["score"]:
                best = {
                    "region": name,
                    "threshold": float(thr),
                    "score": float(score),
                    **stats,
                    "support": int(gold.sum()),
                }
        assert best is not None
        rows.append(best)
    return rows


def apply_region_thresholds(
    region_probs: np.ndarray,
    region_targets: np.ndarray,
    thresholds: Sequence[float],
    region_names: Sequence[str] = REGION_NAMES,
) -> List[Dict[str, float]]:
    probs = np.asarray(region_probs, dtype=np.float32)
    targets = (np.asarray(region_targets, dtype=np.float32) >= 0.5).astype(np.int64)
    rows: List[Dict[str, float]] = []
    for idx, name in enumerate(region_names):
        stats = _binary_stats(probs[:, idx] >= thresholds[idx], targets[:, idx])
        rows.append(
            {
                "region": name,
                "threshold": float(thresholds[idx]),
                "support": int(targets[:, idx].sum()),
                **stats,
            }
        )
    return rows


def format_report(
    gate_summary: Dict[str, float] | None,
    branch_rows: List[Dict[str, float]] | None,
    tuned_rows: List[Dict[str, float]] | None,
    applied_rows: List[Dict[str, float]] | None,
    objective: str,
) -> str:
    lines: List[str] = ["# Stage-2 Prediction Analysis", ""]

    if gate_summary is not None:
        lines.extend(
            [
                "## GatedFusion Gate Weights",
                "",
                "`gate_weight -> 1` means Branch A (temporal) dominates; `gate_weight -> 0` means Branch B (brain-network) dominates.",
                "",
                "| Mean | Std | Min | P10 | Median | P90 | Max | Temporal>0.6 | Balanced[0.4,0.6] | Network<0.4 |",
                "|------|-----|-----|-----|--------|-----|-----|---------------|-------------------|-------------|",
                (
                    f"| {gate_summary['mean']:.4f} | {gate_summary['std']:.4f} | {gate_summary['min']:.4f} | "
                    f"{gate_summary['q10']:.4f} | {gate_summary['q50']:.4f} | {gate_summary['q90']:.4f} | "
                    f"{gate_summary['max']:.4f} | {gate_summary['temporal_dominant_ratio']:.4f} | "
                    f"{gate_summary['balanced_ratio']:.4f} | {gate_summary['network_dominant_ratio']:.4f} |"
                ),
                "",
            ]
        )

    if branch_rows is not None:
        lines.extend(
            [
                "## Branch Weights",
                "",
                "Higher mean branch weight means the feature receives more attention inside the B branch.",
                "",
                "| Feature | Mean | Std | Q25 | Median | Q75 |",
                "|---------|------|-----|-----|--------|-----|",
            ]
        )
        for row in branch_rows:
            lines.append(
                f"| {row['feature']} | {row['mean']:.4f} | {row['std']:.4f} | "
                f"{row['q25']:.4f} | {row['q50']:.4f} | {row['q75']:.4f} |"
            )
        lines.append("")

    if tuned_rows is not None:
        lines.extend(
            [
                "## Region Threshold Search (Validation)",
                "",
                f"Objective: `{objective}`",
                "",
                "| Region | Threshold | Support | Precision | Recall | Specificity | F1 | Balanced Acc |",
                "|--------|-----------|---------|-----------|--------|-------------|----|--------------|",
            ]
        )
        for row in tuned_rows:
            lines.append(
                f"| {row['region']} | {row['threshold']:.2f} | {row['support']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | {row['balanced_accuracy']:.4f} |"
            )
        lines.append("")

    if applied_rows is not None:
        lines.extend(
            [
                "## Region Metrics with Per-Class Thresholds (Applied Set)",
                "",
                "| Region | Threshold | Support | Precision | Recall | Specificity | F1 | Balanced Acc |",
                "|--------|-----------|---------|-----------|--------|-------------|----|--------------|",
            ]
        )
        for row in applied_rows:
            lines.append(
                f"| {row['region']} | {row['threshold']:.2f} | {row['support']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | {row['balanced_accuracy']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze gate weights, branch weights, and per-region thresholds.")
    parser.add_argument("--predictions", required=True, help="Path to val/test_predictions.npz")
    parser.add_argument("--val-predictions", default=None, help="Validation predictions NPZ used for threshold search")
    parser.add_argument("--output-dir", default=None, help="Directory for analysis report")
    parser.add_argument(
        "--region-threshold-objective",
        choices=("balanced_accuracy", "f1"),
        default="balanced_accuracy",
        help="Objective used to tune per-region thresholds on validation predictions",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pred_path)
    gate_summary = None
    branch_rows = None
    tuned_rows = None
    applied_rows = None

    if "gate_weights" in data:
        gate_summary = summarize_gate_weights(data["gate_weights"])
    if "branch_weights" in data:
        counts = data["valid_patch_counts"] if "valid_patch_counts" in data else None
        branch_rows = summarize_branch_weights(data["branch_weights"], counts)

    if args.val_predictions:
        val_data = np.load(args.val_predictions)
        tuned_rows = search_region_thresholds(
            region_probs=val_data["region_probs"],
            region_targets=val_data["region_targets"],
            objective=args.region_threshold_objective,
        )
        thresholds = [row["threshold"] for row in tuned_rows]
        applied_rows = apply_region_thresholds(
            region_probs=data["region_probs"],
            region_targets=data["region_targets"],
            thresholds=thresholds,
        )

    report = format_report(
        gate_summary=gate_summary,
        branch_rows=branch_rows,
        tuned_rows=tuned_rows,
        applied_rows=applied_rows,
        objective=args.region_threshold_objective,
    )
    report_path = out_dir / "stage2_prediction_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved analysis report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
