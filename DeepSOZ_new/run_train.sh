#!/usr/bin/env bash
# =============================================================================
# DeepSOZ_new 两阶段训练脚本
#
# 用法：
#   bash run_train.sh [阶段] [数据源] [折]
#
# 阶段（第一个参数）：
#   both   — Stage-1 + Stage-2（默认）
#   1      — 仅 Stage-1
#   2      — 仅 Stage-2
#
# 数据源（第二个参数）：
#   all     — TUSZ + 私有数据混合（默认）
#   tusz    — 仅 TUSZ
#   private — 仅私有数据
#
# 折（第三个参数，可选）：
#   all    — 全部 K 折（默认）
#   0~N-1  — 只训练指定折
#
# 示例：
#   # 仅用 TUSZ 数据，两阶段全流程
#   bash run_train.sh both tusz
#
#   # 仅用私有数据，Stage-2，第 0 折
#   bash run_train.sh 2 private 0
#
#   # 混合训练（通过环境变量指定路径）
#   MANIFEST=/mnt/data/combined_manifest.csv \
#   DATA_ROOTS="/mnt/data/tusz /mnt/data/private" \
#   bash run_train.sh both all
#
#   # TCP 双极导联模式
#   USE_BIPOLAR=1 bash run_train.sh both tusz
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
MANIFEST="${MANIFEST:-/mnt/hd1/dyf/dataset/EEG_dataset_SUAT/combined_manifest.csv}"
DATA_ROOTS="${DATA_ROOTS:-/mnt/hd1/dyf/dataset/EEG_dataset_SUAT}"
DATA_MODE="${DATA_MODE:-online}"

# ── 缓存配置 ──────────────────────────────────────────────────────────────────
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cache}"

# ── 运行设备 ──────────────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ── 参数选择 ──────────────────────────────────────────────────────────────────
STAGE="${1:-both}"                  # both / 1 / 2
SOURCE="${2:-all}"                  # all / tusz / private
FOLD_ARG="${3:-all}"                # all 或具体折号

# ── 导联模式 ──────────────────────────────────────────────────────────────────
USE_BIPOLAR="${USE_BIPOLAR:-0}"     # 1=TCP双极22ch, 0=单极19ch

# ── K 折 ──────────────────────────────────────────────────────────────────────
N_FOLDS=5
SEED=42

# ── 数据预处理参数 ────────────────────────────────────────────────────────────
N_WINDOWS=45
TARGET_FS=200.0
F_LOW=1.6
F_HIGH=30.0

# ── Stage-1 官方超参 ──────────────────────────────────────────────────────────
STAGE1_LR=1e-5
STAGE1_EPOCHS=30
STAGE1_PATIENCE=10
STAGE1_BATCH=1

# ── Stage-2 官方超参 ──────────────────────────────────────────────────────────
STAGE2_LR=1e-4
STAGE2_EPOCHS=50
STAGE2_PATIENCE=15
STAGE2_BATCH=1

# ── 模型结构（官方默认） ──────────────────────────────────────────────────────
TF_DROPOUT=0.15
CNN_DROPOUT=0.15
GRU_DROPOUT=0.0

# ── Stage-2 损失权重（官方 szloc_train.py） ──────────────────────────────────
CHN_SZ_WEIGHT=1.0
TOT_SZ_WEIGHT=1.0
ATTN_MAP_W_POS=2.0
ATTN_MAP_W_NEG=1.0
ATTN_MAP_W_MARGIN=1.0
CHN_MAP_W_POS=2.0
CHN_MAP_W_NEG=1.0
CHN_MAP_W_MARGIN=1.0

# ── 通用训练 ──────────────────────────────────────────────────────────────────
GRAD_CLIP=1.0

# 自动设置输出目录名
if [[ "${SOURCE}" == "all" ]]; then
    SRC_TAG="mixed"
else
    SRC_TAG="${SOURCE}"
fi
if [[ "${USE_BIPOLAR}" == "1" ]]; then
    CH_TAG="bipolar22"
else
    CH_TAG="mono19"
fi
OUTPUT_DIR="${SCRIPT_DIR}/runs/${SRC_TAG}_${CH_TAG}_${STAGE}"
EXP_PREFIX="deepsoz_${SRC_TAG}"

echo "============================================================"
echo "  DeepSOZ_new 训练"
echo "  阶段:      ${STAGE}"
echo "  数据源:    ${SOURCE}"
echo "  导联:      ${CH_TAG}"
echo "  折:        ${FOLD_ARG}"
echo "  manifest:  ${MANIFEST}"
echo "  data:      ${DATA_ROOTS}"
echo "  data_mode: ${DATA_MODE}"
echo "  cache:     ${CACHE_DIR}"
echo "  device:    ${DEVICE}"
echo "  output:    ${OUTPUT_DIR}"
echo "============================================================"

# ── 构建公共参数 ──────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --manifest      "${MANIFEST}"
    --data-roots    ${DATA_ROOTS}
    --data-mode     "${DATA_MODE}"
    --stage         "${STAGE}"
    --n-folds       "${N_FOLDS}"
    --seed          "${SEED}"
    --n-windows     "${N_WINDOWS}"
    --target-fs     "${TARGET_FS}"
    --f-low         "${F_LOW}"
    --f-high        "${F_HIGH}"
    --stage1-lr     "${STAGE1_LR}"
    --stage1-epochs "${STAGE1_EPOCHS}"
    --stage1-patience "${STAGE1_PATIENCE}"
    --stage1-batch  "${STAGE1_BATCH}"
    --stage2-lr     "${STAGE2_LR}"
    --stage2-epochs "${STAGE2_EPOCHS}"
    --stage2-patience "${STAGE2_PATIENCE}"
    --stage2-batch  "${STAGE2_BATCH}"
    --tf-dropout    "${TF_DROPOUT}"
    --cnn-dropout   "${CNN_DROPOUT}"
    --gru-dropout   "${GRU_DROPOUT}"
    --chn-sz-weight "${CHN_SZ_WEIGHT}"
    --tot-sz-weight "${TOT_SZ_WEIGHT}"
    --attn-map-w-pos    "${ATTN_MAP_W_POS}"
    --attn-map-w-neg    "${ATTN_MAP_W_NEG}"
    --attn-map-w-margin "${ATTN_MAP_W_MARGIN}"
    --chn-map-w-pos     "${CHN_MAP_W_POS}"
    --chn-map-w-neg     "${CHN_MAP_W_NEG}"
    --chn-map-w-margin  "${CHN_MAP_W_MARGIN}"
    --grad-clip     "${GRAD_CLIP}"
    --num-workers   "${NUM_WORKERS}"
    --cache-dir     "${CACHE_DIR}"
    --device        "${DEVICE}"
    --output-dir    "${OUTPUT_DIR}"
    --exp-prefix    "${EXP_PREFIX}"
)

# ── source 过滤 ──────────────────────────────────────────────────────────────
if [[ "${SOURCE}" != "all" ]]; then
    COMMON_ARGS+=(--source "${SOURCE}")
fi

# ── 双极导联 ─────────────────────────────────────────────────────────────────
if [[ "${USE_BIPOLAR}" == "1" ]]; then
    COMMON_ARGS+=(--use-bipolar)
fi

# ── Stage-1 checkpoint ────────────────────────────────────────────────────────
if [[ "${STAGE}" == "2" && -n "${STAGE1_CKPT:-}" ]]; then
    COMMON_ARGS+=(--stage1-ckpt "${STAGE1_CKPT}")
fi

# ── 指定折 ───────────────────────────────────────────────────────────────────
if [[ "${FOLD_ARG}" != "all" ]]; then
    COMMON_ARGS+=(--fold "${FOLD_ARG}")
fi

# ── 执行训练 ──────────────────────────────────────────────────────────────────
python "${SCRIPT_DIR}/train.py" "${COMMON_ARGS[@]}"

echo ""
echo "训练完成！结果保存在: ${OUTPUT_DIR}"
