#!/usr/bin/env bash
set -euo pipefail

# Run stage-2 private leave-one-out finetuning across all private patients,
# starting from an existing TUSZ-only SOZ checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/TUSZ/models/train_soz_locator_with_brain_networks.py}"

MANIFEST="${MANIFEST:-${REPO_ROOT}/TUSZ/combined_manifest.csv}"
TUSZ_DATA_ROOT="${TUSZ_DATA_ROOT:-/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf}"
PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-/mnt/hd1/dyf/dataset/EEG dataset_SUAT}"
LABRAM_CKPT="${LABRAM_CKPT:-/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth}"
INIT_SOZ_CKPT="${INIT_SOZ_CKPT:-}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-}"

LOO_OUTPUT_BASE="${LOO_OUTPUT_BASE:-${REPO_ROOT}/TUSZ/models/runs/stage2_private_loo}"

VAL_OFFSET="${VAL_OFFSET:-1}"
SEED_BASE="${SEED_BASE:-42}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-}"

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

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

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

run_private_loo_fold() {
  local fold_index="$1"
  local seed="$2"
  local fold_output="${LOO_OUTPUT_BASE}_fold${fold_index}"
  local cmd=(
    "$PYTHON_BIN" "$TRAIN_SCRIPT"
    --manifest "$MANIFEST"
    --tusz-data-root "$TUSZ_DATA_ROOT"
    --private-data-root "$PRIVATE_DATA_ROOT"
    --labram-ckpt "$LABRAM_CKPT"
    --init-soz-ckpt "$INIT_SOZ_CKPT"
    --source private
    --split-strategy private_loo
    --private-loo-fold-index "$fold_index"
    --private-loo-val-offset "$VAL_OFFSET"
    --labram-frozen-layers "$LABRAM_FROZEN_LAYERS"
    --finetune-epochs "$FINETUNE_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --workers "$WORKERS"
    --lr "$LR_PRIVATE"
    --seed "$seed"
    --w-region "$W_REGION"
    --w-hemisphere "$W_HEMISPHERE"
    --w-transition "$W_TRANSITION"
    --w-pattern "$W_PATTERN"
    --focal-alpha "$FOCAL_ALPHA"
    --focal-gamma "$FOCAL_GAMMA"
    --generalized-pos-ratio-threshold "$GENERALIZED_POS_RATIO_THRESHOLD"
    --generalized-sample-weight "$GENERALIZED_SAMPLE_WEIGHT"
    --private-patient-weight-power "$PRIVATE_PATIENT_WEIGHT_POWER"
    --private-rare-channel-sampler-strength "$PRIVATE_RARE_CHANNEL_SAMPLER_STRENGTH"
    --private-rare-channel-sampler-max-boost "$PRIVATE_RARE_CHANNEL_SAMPLER_MAX_BOOST"
    --private-sampler-max-weight "$PRIVATE_SAMPLER_MAX_WEIGHT"
    --private-common-channel-loss-min-weight "$PRIVATE_COMMON_CHANNEL_LOSS_MIN_WEIGHT"
    --private-rare-channel-loss-max-weight "$PRIVATE_RARE_CHANNEL_LOSS_MAX_WEIGHT"
    --private-zero-positive-channel-weight "$PRIVATE_ZERO_POSITIVE_CHANNEL_WEIGHT"
    --augment-gaussian-prob "$AUGMENT_GAUSSIAN_PROB"
    --augment-gaussian-std-scale "$AUGMENT_GAUSSIAN_STD_SCALE"
    --augment-bandstop-prob "$AUGMENT_BANDSTOP_PROB"
    --augment-bandstop-min-freq "$AUGMENT_BANDSTOP_MIN_FREQ"
    --augment-bandstop-max-freq "$AUGMENT_BANDSTOP_MAX_FREQ"
    --augment-bandstop-width-hz "$AUGMENT_BANDSTOP_WIDTH_HZ"
    --augment-channel-drop-prob "$AUGMENT_CHANNEL_DROP_PROB"
    --augment-max-channel-drops "$AUGMENT_MAX_CHANNEL_DROPS"
    --augment-lr-mirror-prob "$AUGMENT_LR_MIRROR_PROB"
    --brain-network-features "$BRAIN_NETWORK_FEATURES"
    --output-dir "$fold_output"
  )

  if [[ "$FREEZE_LABRAM_PRIVATE" == "1" ]]; then
    cmd+=(--freeze-labram)
  fi
  if [[ "$PRIVATE_BALANCED_SAMPLER" == "1" ]]; then
    cmd+=(--private-balanced-sampler)
  else
    cmd+=(--no-private-balanced-sampler)
  fi
  if [[ "$PRIVATE_CHANNEL_LOSS_WEIGHT" == "1" ]]; then
    cmd+=(--private-channel-loss-weight)
  else
    cmd+=(--no-private-channel-loss-weight)
  fi
  if [[ "$PRIVATE_EEG_AUGMENT" == "1" ]]; then
    cmd+=(--private-eeg-augment)
  else
    cmd+=(--no-private-eeg-augment)
  fi
  if [[ -n "$PRECOMPUTED_DIR" ]]; then
    cmd+=(--precomputed-dir "$PRECOMPUTED_DIR")
  fi

  echo "[private_loo] fold=${fold_index} seed=${seed} output=${fold_output}"
  "${cmd[@]}"
}

main() {
  require_file "$TRAIN_SCRIPT" "training script"
  require_file "$MANIFEST" "manifest"
  require_dir "$TUSZ_DATA_ROOT" "TUSZ data root"
  require_dir "$PRIVATE_DATA_ROOT" "private data root"
  require_file "$LABRAM_CKPT" "LaBraM checkpoint"
  if [[ -z "$INIT_SOZ_CKPT" ]]; then
    echo "INIT_SOZ_CKPT is required. Set it to your TUSZ-only best_model.pt." >&2
    exit 1
  fi
  require_file "$INIT_SOZ_CKPT" "initial SOZ checkpoint"

  local n_folds
  n_folds="$(count_private_patients)"
  if [[ "$n_folds" -lt 3 ]]; then
    echo "Need at least 3 private patients for private_loo, got: ${n_folds}" >&2
    exit 1
  fi

  if [[ -z "$END_FOLD" ]]; then
    END_FOLD=$((n_folds - 1))
  fi

  echo "Manifest: ${MANIFEST}"
  echo "Init ckpt: ${INIT_SOZ_CKPT}"
  echo "Private patient count: ${n_folds}"
  echo "Fold range: ${START_FOLD}..${END_FOLD}"
  echo "Validation offset: ${VAL_OFFSET}"
  if [[ -n "$PRECOMPUTED_DIR" ]]; then
    echo "Precomputed brain networks: ${PRECOMPUTED_DIR}"
    echo "Note: private EEG and LR-mirror augmentation will be disabled by the trainer."
  fi

  local fold
  for ((fold=START_FOLD; fold<=END_FOLD; fold++)); do
    run_private_loo_fold "$fold" "$((SEED_BASE + fold))"
  done

  echo "All requested private LOO folds completed."
}

main "$@"
