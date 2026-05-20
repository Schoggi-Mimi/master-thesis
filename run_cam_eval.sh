#!/bin/bash
#SBATCH --job-name=cam_eval
#SBATCH --output=logs/cam_eval_%x_%j.out
#SBATCH --error=logs/cam_eval_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mail-user=choekyel.nyungmartsang@students.unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gratis

REPO_DIR="$HOME/projects/master-thesis"
cd "$REPO_DIR"

mkdir -p logs

module load Anaconda3
eval "$(conda shell.bash hook)"

# Some conda activate scripts on UBELIX reference unset variables.
# Temporarily disable nounset so activation does not fail with:
# MKL_INTERFACE_LAYER: unbound variable
set +u
conda activate thesis
set -u

# -----------------------------------------------------------------------------
# CAM metric evaluation wrapper
# -----------------------------------------------------------------------------
# Recommended usage from repo root:
#
# Base CE, final block:
#   JOB_NAME=base_last \
#   CHECKPOINT=external/checkpoints3/checkpoint-best-base.pth \
#   OUT_DIR=external/metrics/metrics_base_last_100_topk3 \
#   TARGET_BLOCK_INDEX=-1 \
#   sbatch run_cam_eval.sh
#
# Base CE, block -4:
#   JOB_NAME=base_block4 \
#   CHECKPOINT=external/checkpoints3/checkpoint-best-base.pth \
#   OUT_DIR=external/metrics/metrics_base_block4_100_topk3 \
#   TARGET_BLOCK_INDEX=-4 \
#   sbatch run_cam_eval.sh
#
# HA 0.75, final block:
#   JOB_NAME=ha075_last \
#   CHECKPOINT=external/checkpoints3/checkpoint-best-HA075.pth \
#   OUT_DIR=external/metrics/metrics_ha075_last_100_topk3 \
#   TARGET_BLOCK_INDEX=-1 \
#   sbatch run_cam_eval.sh
#
# HA 0.75, block -4:
#   JOB_NAME=ha075_block4 \
#   CHECKPOINT=external/checkpoints3/checkpoint-best-HA075.pth \
#   OUT_DIR=external/metrics/metrics_ha075_block4_100_topk3 \
#   TARGET_BLOCK_INDEX=-4 \
#   sbatch run_cam_eval.sh
# -----------------------------------------------------------------------------

JOB_NAME=${JOB_NAME:-cam_eval}

CSV=${CSV:-data/HAM10000/ham_test_for_cam.csv}
IMAGE_COL=${IMAGE_COL:-image_rel_path}
IMG_DIR=${IMG_DIR:-data/HAM10000}
GT_COL=${GT_COL:-gt_label}
MASK_ROOT=${MASK_ROOT:-data/HAM10000}
MASK_COL=${MASK_COL:-mask_rel_path}

CHECKPOINT=${CHECKPOINT:-external/checkpoints3/checkpoint-best-base.pth}
CHECKPOINT_MODEL_TYPE=${CHECKPOINT_MODEL_TYPE:-panderm}
CLASS_PRESET=${CLASS_PRESET:-ham}

OUT_DIR=${OUT_DIR:-external/metrics/metrics_${JOB_NAME}_100_topk3}
NUM_SAMPLES=${NUM_SAMPLES:-100}
DEVICE=${DEVICE:-cuda}
COMPARE_MODE=${COMPARE_MODE:-gt_topk_non_target}
TOPK_COMPARE=${TOPK_COMPARE:-3}
PERTURBATION_STEPS=${PERTURBATION_STEPS:-0.1}
TARGET_BLOCK_INDEX=${TARGET_BLOCK_INDEX:--1}
METHODS=${METHODS:-gradcam_target,gradcam_reference,gradcam_diff,finercam}

USE_SEG_GATE=${USE_SEG_GATE:-0}
SEG_GATE_BG_KEEP=${SEG_GATE_BG_KEEP:-0.05}

echo "================================================================================"
echo "CAM evaluation job: $JOB_NAME"
echo "================================================================================"
echo "CSV: $CSV"
echo "CHECKPOINT: $CHECKPOINT"
echo "CHECKPOINT_MODEL_TYPE: $CHECKPOINT_MODEL_TYPE"
echo "OUT_DIR: $OUT_DIR"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "TARGET_BLOCK_INDEX: $TARGET_BLOCK_INDEX"
echo "METHODS: $METHODS"
echo "USE_SEG_GATE: $USE_SEG_GATE"
echo "================================================================================"

CMD=(
  python -m scripts.eval_cam_metrics_panderm
  --csv "$CSV"
  --image_col "$IMAGE_COL"
  --img_dir "$IMG_DIR"
  --gt_col "$GT_COL"
  --checkpoint "$CHECKPOINT"
  --checkpoint_model_type "$CHECKPOINT_MODEL_TYPE"
  --class_preset "$CLASS_PRESET"
  --out_dir "$OUT_DIR"
  --num_samples "$NUM_SAMPLES"
  --device "$DEVICE"
  --compare_mode "$COMPARE_MODE"
  --topk_compare "$TOPK_COMPARE"
  --mask_root "$MASK_ROOT"
  --mask_col "$MASK_COL"
  --perturbation_steps "$PERTURBATION_STEPS"
  --methods "$METHODS"
  --target_block_index "$TARGET_BLOCK_INDEX"
)

if [[ "$USE_SEG_GATE" == "1" ]]; then
  CMD+=(
    --use_seg_gate
    --seg_gate_bg_keep "$SEG_GATE_BG_KEEP"
  )
fi

printf 'Running command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"