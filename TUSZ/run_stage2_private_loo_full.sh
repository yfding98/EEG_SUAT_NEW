#!/usr/bin/env bash
set -euo pipefail

# Full stage-2 training pipeline:
# 1. Train stage-2 on TUSZ only.
# 2. Run patient-level leave-one-out finetuning on the private dataset,
#    initializing each fold from the TUSZ-only stage-2 checkpoint.

PYTHON_BIN="${PYTHON_BIN:-python}"

MANIFEST="${MANIFEST:-/mnt/hd1/dyf/workspace/EEG_SUAT_NEW/TUSZ/combined_manifest.csv}"
TUSZ_DATA_ROOT="${TUSZ_DATA_ROOT:-/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf}"
PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-/mnt/hd1/dyf/dataset/EEG dataset_SUAT}"
LABRAM_CKPT="${LABRAM_CKPT:-/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-TUSZ/models/runs/stage1_only/best_pretrain_ckpt.pth}"

RUN_TUSZ_STAGE2="${RUN_TUSZ_STAGE2:-1}"
VAL_OFFSET="${VAL_OFFSET:-1}"
SEED_BASE="${SEED_BASE:-42}"

STAGE2_TUSZ_OUTPUT="${STAGE2_TUSZ_OUTPUT:-TUSZ/models/runs/stage2_tusz_only}"
LOO_OUTPUT_BASE="${LOO_OUTPUT_BASE:-TUSZ/models/runs/stage2_private_loo}"

LABRAM_FROZEN_LAYERS="${LABRAM_FROZEN_LAYERS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_PRIVATE="${LR_PRIVATE:-2e-5}"
WORKERS="${WORKERS:-4}"
FREEZE_LABRAM_PRIVATE="${FREEZE_LABRAM_PRIVATE:-1}"

W_REGION="${W_REGION:-0.5}"
W_HEMISPHERE="${W_HEMISPHERE:-0.0}"
W_TRANSITION="${W_TRANSITION:-0.3}"
W_PATTERN="${W_PATTERN:-0.2}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.5}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
GENERALIZED_POS_RATIO_THRESHOLD="${GENERALIZED_POS_RATIO_THRESHOLD:-0.5}"
GENERALIZED_SAMPLE_WEIGHT="${GENERALIZED_SAMPLE_WEIGHT:-0.2}"
BRAIN_NETWORK_FEATURES="${BRAIN_NETWORK_FEATURES:-gc,te,aec,wpli}"
PRIVATE_BALANCED_SAMPLER="${PRIVATE_BALANCED_SAMPLER:-1}"
PRIVATE_PATIENT_WEIGHT_POWER="${PRIVATE_PATIENT_WEIGHT_POWER:-1.0}"
PRIVATE_RARE_CHANNEL_SAMPLER_STRENGTH="${PRIVATE_RARE_CHANNEL_SAMPLER_STRENGTH:-0.5}"
PRIVATE_RARE_CHANNEL_SAMPLER_MAX_BOOST="${PRIVATE_RARE_CHANNEL_SAMPLER_MAX_BOOST:-2.5}"
PRIVATE_SAMPLER_MAX_WEIGHT="${PRIVATE_SAMPLER_MAX_WEIGHT:-4.0}"
PRIVATE_CHANNEL_LOSS_WEIGHT="${PRIVATE_CHANNEL_LOSS_WEIGHT:-1}"
PRIVATE_COMMON_CHANNEL_LOSS_MIN_WEIGHT="${PRIVATE_COMMON_CHANNEL_LOSS_MIN_WEIGHT:-0.5}"
PRIVATE_RARE_CHANNEL_LOSS_MAX_WEIGHT="${PRIVATE_RARE_CHANNEL_LOSS_MAX_WEIGHT:-3.0}"
PRIVATE_ZERO_POSITIVE_CHANNEL_WEIGHT="${PRIVATE_ZERO_POSITIVE_CHANNEL_WEIGHT:-0.2}"
PRIVATE_EEG_AUGMENT="${PRIVATE_EEG_AUGMENT:-1}"
AUGMENT_GAUSSIAN_PROB="${AUGMENT_GAUSSIAN_PROB:-0.4}"
AUGMENT_GAUSSIAN_STD_SCALE="${AUGMENT_GAUSSIAN_STD_SCALE:-0.01}"
AUGMENT_BANDSTOP_PROB="${AUGMENT_BANDSTOP_PROB:-0.25}"
AUGMENT_BANDSTOP_MIN_FREQ="${AUGMENT_BANDSTOP_MIN_FREQ:-45.0}"
AUGMENT_BANDSTOP_MAX_FREQ="${AUGMENT_BANDSTOP_MAX_FREQ:-65.0}"
AUGMENT_BANDSTOP_WIDTH_HZ="${AUGMENT_BANDSTOP_WIDTH_HZ:-2.0}"
AUGMENT_CHANNEL_DROP_PROB="${AUGMENT_CHANNEL_DROP_PROB:-0.15}"
AUGMENT_MAX_CHANNEL_DROPS="${AUGMENT_MAX_CHANNEL_DROPS:-1}"
AUGMENT_LR_MIRROR_PROB="${AUGMENT_LR_MIRROR_PROB:-0.10}"

START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-}"

count_private_patients() {
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys

manifest = sys.argv[1]
patients = set()
with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if str(row.get("source", "")).strip().lower() != "private":
            continue
        patient_id = str(row.get("patient_id", "")).strip()
        if patient_id:
            patients.add(patient_id)
print(len(patients))
PY
}

run_tusz_stage2() {
  echo "[1/2] Training stage-2 on TUSZ only -> ${STAGE2_TUSZ_OUTPUT}"
  "$PYTHON_BIN" TUSZ/models/train_soz_locator_with_brain_networks.py \
    --manifest "$MANIFEST" \
    --tusz-data-root "$TUSZ_DATA_ROOT" \
    --labram-ckpt "$LABRAM_CKPT" \
    --stage-pretrain-ckpt "$STAGE1_CKPT" \
    --source tusz \
    --split-strategy auto \
    --labram-frozen-layers "$LABRAM_FROZEN_LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --w-region "$W_REGION" \
    --w-hemisphere "$W_HEMISPHERE" \
    --w-transition "$W_TRANSITION" \
    --w-pattern "$W_PATTERN" \
    --focal-alpha "$FOCAL_ALPHA" \
    --focal-gamma "$FOCAL_GAMMA" \
    --generalized-pos-ratio-threshold "$GENERALIZED_POS_RATIO_THRESHOLD" \
    --generalized-sample-weight "$GENERALIZED_SAMPLE_WEIGHT" \
    --brain-network-features "$BRAIN_NETWORK_FEATURES" \
    --output-dir "$STAGE2_TUSZ_OUTPUT"
}

run_private_loo_fold() {
  local fold_index="$1"
  local seed="$2"
  local fold_output="${LOO_OUTPUT_BASE}_fold${fold_index}"
  local private_args=()

  if [[ "$FREEZE_LABRAM_PRIVATE" == "1" ]]; then
    private_args+=(--freeze-labram)
  fi
  if [[ "$PRIVATE_BALANCED_SAMPLER" == "1" ]]; then
    private_args+=(--private-balanced-sampler)
  else
    private_args+=(--no-private-balanced-sampler)
  fi
  if [[ "$PRIVATE_CHANNEL_LOSS_WEIGHT" == "1" ]]; then
    private_args+=(--private-channel-loss-weight)
  else
    private_args+=(--no-private-channel-loss-weight)
  fi
  if [[ "$PRIVATE_EEG_AUGMENT" == "1" ]]; then
    private_args+=(--private-eeg-augment)
  else
    private_args+=(--no-private-eeg-augment)
  fi

  echo "[2/2] Running private LOO fold ${fold_index} -> ${fold_output}"
  "$PYTHON_BIN" TUSZ/models/train_soz_locator_with_brain_networks.py \
    --manifest "$MANIFEST" \
    --tusz-data-root "$TUSZ_DATA_ROOT" \
    --private-data-root "$PRIVATE_DATA_ROOT" \
    --labram-ckpt "$LABRAM_CKPT" \
    --init-soz-ckpt "${STAGE2_TUSZ_OUTPUT}/best_model.pt" \
    --source private \
    --split-strategy private_loo \
    --private-loo-fold-index "$fold_index" \
    --private-loo-val-offset "$VAL_OFFSET" \
    --labram-frozen-layers "$LABRAM_FROZEN_LAYERS" \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --lr "$LR_PRIVATE" \
    --seed "$seed" \
    --w-region "$W_REGION" \
    --w-hemisphere "$W_HEMISPHERE" \
    --w-transition "$W_TRANSITION" \
    --w-pattern "$W_PATTERN" \
    --focal-alpha "$FOCAL_ALPHA" \
    --focal-gamma "$FOCAL_GAMMA" \
    --generalized-pos-ratio-threshold "$GENERALIZED_POS_RATIO_THRESHOLD" \
    --generalized-sample-weight "$GENERALIZED_SAMPLE_WEIGHT" \
    --private-patient-weight-power "$PRIVATE_PATIENT_WEIGHT_POWER" \
    --private-rare-channel-sampler-strength "$PRIVATE_RARE_CHANNEL_SAMPLER_STRENGTH" \
    --private-rare-channel-sampler-max-boost "$PRIVATE_RARE_CHANNEL_SAMPLER_MAX_BOOST" \
    --private-sampler-max-weight "$PRIVATE_SAMPLER_MAX_WEIGHT" \
    --private-common-channel-loss-min-weight "$PRIVATE_COMMON_CHANNEL_LOSS_MIN_WEIGHT" \
    --private-rare-channel-loss-max-weight "$PRIVATE_RARE_CHANNEL_LOSS_MAX_WEIGHT" \
    --private-zero-positive-channel-weight "$PRIVATE_ZERO_POSITIVE_CHANNEL_WEIGHT" \
    --augment-gaussian-prob "$AUGMENT_GAUSSIAN_PROB" \
    --augment-gaussian-std-scale "$AUGMENT_GAUSSIAN_STD_SCALE" \
    --augment-bandstop-prob "$AUGMENT_BANDSTOP_PROB" \
    --augment-bandstop-min-freq "$AUGMENT_BANDSTOP_MIN_FREQ" \
    --augment-bandstop-max-freq "$AUGMENT_BANDSTOP_MAX_FREQ" \
    --augment-bandstop-width-hz "$AUGMENT_BANDSTOP_WIDTH_HZ" \
    --augment-channel-drop-prob "$AUGMENT_CHANNEL_DROP_PROB" \
    --augment-max-channel-drops "$AUGMENT_MAX_CHANNEL_DROPS" \
    --augment-lr-mirror-prob "$AUGMENT_LR_MIRROR_PROB" \
    --brain-network-features "$BRAIN_NETWORK_FEATURES" \
    "${private_args[@]}" \
    --output-dir "$fold_output"
}

main() {
  local n_folds
  n_folds="$(count_private_patients)"
  if [[ "$n_folds" -lt 3 ]]; then
    echo "Need at least 3 private patients for private_loo, got: ${n_folds}" >&2
    exit 1
  fi

  if [[ -z "$END_FOLD" ]]; then
    END_FOLD=$((n_folds - 1))
  fi

  echo "Private patient count: ${n_folds}"
  echo "Fold range: ${START_FOLD}..${END_FOLD}"
  echo "Validation offset: ${VAL_OFFSET}"

  if [[ "$RUN_TUSZ_STAGE2" == "1" ]]; then
    run_tusz_stage2
  else
    echo "Skipping TUSZ-only stage-2 training (RUN_TUSZ_STAGE2=${RUN_TUSZ_STAGE2})"
  fi

  if [[ ! -f "${STAGE2_TUSZ_OUTPUT}/best_model.pt" ]]; then
    echo "Missing init checkpoint: ${STAGE2_TUSZ_OUTPUT}/best_model.pt" >&2
    exit 1
  fi

  local fold
  for ((fold=START_FOLD; fold<=END_FOLD; fold++)); do
    run_private_loo_fold "$fold" "$((SEED_BASE + fold))"
  done

  echo "All requested private LOO folds completed."
}

main "$@"
