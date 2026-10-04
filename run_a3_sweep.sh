#!/bin/bash
set -euo pipefail
# =============================================================================
# ALIGNMENT THREE. Train on the supervised images ONLY.
#
# A1 and A2 keep the full 6350 image train split for the cross entropy loss.
# A3 removes it. Train becomes the 45 or 46 supervised images.
#
# WHY
#   1. Tests the foundation model claim directly. Does PanDerm hold
#      classification on 45 images.
#   2. Matches A1 and A2 on supervision VOLUME. In A1 the HA loss draws from
#      every train row carrying a lesion mask, roughly 6350. In A2 the feature
#      loss draws from 65. Any A1 vs A2 difference is confounded by that.
#      Here every arm sees the same 45.
#
# ARMS
#   a3ce  no alignment. CONTROL and the most important arm. If classification
#         collapses here the CAM is the gradient of a broken logit and neither
#         alignment arm can be interpreted.
#   a3ha  lesion mask target, Tschandl. A1 matched on volume.
#   a3ft  feature mask target, Florentia, both channels. A2 matched on volume.
#
# NAMING. The worker builds its output name from EXPERIMENT_TAG and the loss
# parameters. The CE and HA branches do NOT include the fold, so without a
# per fold EXPERIMENT_TAG all five folds write to one directory and overwrite
# each other. That is what happened on the first attempt.
# The feat branch hardcodes an "a2_" prefix and ignores EXPERIMENT_TAG, but it
# does include the fold, so a3ft survived under a misleading name.
#
# Anchors reduced 40 to 20. The 40 anchor version gave a 4 to 1 patch budget
# favouring nevus, over correcting an annotated set already near balanced.
#
# Warm started from ha0, the UNALIGNED model, same as A2.
#
# SCHEDULE. 45 images at batch 8 with drop_last=True gives 5 steps per epoch.
# 60 epochs is 300 steps against 1320 for A1 and A2. Lower LR because the
# weighted sampler shows each of the 12 melanoma images roughly 112 times.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WORKER="$REPO_ROOT/run_panderm_full_finetune_ha.sh"
[[ -f "$WORKER" ]] || WORKER="$REPO_ROOT/scripts/run_panderm_full_finetune_ha.sh"
[[ -f "$WORKER" ]] || { echo "worker script not found"; exit 1; }

GPU_MODEL=${GPU_MODEL:-rtx4090}
QOS=${QOS:-job_gratis}
PARTITION=${PARTITION:-gpu}
CPUS=${CPUS:-4}
MEM=${MEM:-32G}
WALLTIME=${WALLTIME:-1:30:00}

BATCH_SIZE=${BATCH_SIZE:-8}
FEAT_BATCH_SIZE=${FEAT_BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-60}
LR=${LR:-5e-6}
POOLING=mean
POOLING_TAG=gap
DAL_LAMBDA=0.0
NB_CLASSES=2
HA_LOSS_TYPE=${HA_LOSS_TYPE:-paper_dice}
HA_FP_WEIGHT=${HA_FP_WEIGHT:-1.0}
HA_START_EPOCH=${HA_START_EPOCH:-0}
LAM=${LAM:-5.0}

DATA_DIR="$REPO_ROOT/data/HAM10000/mel_nv"
ROOT_PATH="$REPO_ROOT/data/HAM10000"
OUTPUT_ROOT="$REPO_ROOT/outputs/alignment3"
INIT_CHECKPOINT="$REPO_ROOT/external/checkpoints5/checkpoint-best-gap-ha0.pth"
CHECKPOINT_KEEP_DIR="$REPO_ROOT/external/checkpoints_a3"
WANDB_PROJECT=master-thesis-mel-nv

ARMS=(${ARMS_OVERRIDE:-a3ce a3ha a3ft})
FOLDS=(${FOLDS_OVERRIDE:-0 1 2 3 4})
FORCE=${FORCE:-0}

# ---- batch size guard. drop_last=True on a 45 image split -------------------
[[ "$BATCH_SIZE" -le 40 ]] || {
  echo "BATCH_SIZE ${BATCH_SIZE} too large. The train loader uses drop_last=True"
  echo "and the split has roughly 45 images, so this yields zero batches."
  exit 1; }

[[ -f "$INIT_CHECKPOINT" ]] || { echo "MISSING $INIT_CHECKPOINT"; exit 1; }
[[ -d "$ROOT_PATH" ]]       || { echo "MISSING $ROOT_PATH"; exit 1; }
for K in "${FOLDS[@]}"; do
  [[ -f "$DATA_DIR/a3_fold${K}.csv"         ]] || { echo "MISSING a3_fold${K}.csv"; exit 1; }
  [[ -f "$DATA_DIR/a3_fold${K}_heldout.csv" ]] || { echo "MISSING a3_fold${K}_heldout.csv"; exit 1; }
  N_TRAIN=$(awk -F, 'NR>1 && $0 ~ /,train,/ {c++} END{print c+0}' "$DATA_DIR/a3_fold${K}.csv")
  echo "  fold ${K} train rows ${N_TRAIN}"
  [[ "$N_TRAIN" -lt 100 ]] || { echo "fold ${K} has ${N_TRAIN} train rows, expected ~45."; exit 1; }
done
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_KEEP_DIR"

# ---- predict the checkpoint name the worker will write ----------------------
predict_tag () {
  local ARM=$1 K=$2 ETAG=$3
  case "$ARM" in
    a3ce) echo "${ETAG}_${POOLING_TAG}_ce_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}" ;;
    a3ha) echo "${ETAG}_${POOLING_TAG}_ha${LAM}_start${HA_START_EPOCH}_${HA_LOSS_TYPE}_fp${HA_FP_WEIGHT}_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}" ;;
    a3ft) local FT; FT=$(echo "$LAM" | sed 's/\./p/')
          echo "a2_${POOLING_TAG}_feat${FT}_both_fold${K}_ep${EPOCHS}_lr${LR}" ;;
  esac
}

# ---- collision check BEFORE submitting anything -----------------------------
declare -A SEEN
COLLIDE=0
for ARM in "${ARMS[@]}"; do
for K in "${FOLDS[@]}"; do
  TAG=$(predict_tag "$ARM" "$K" "${ARM}fold${K}")
  if [[ -n "${SEEN[$TAG]:-}" ]]; then
    echo "COLLISION  ${ARM} fold ${K} and ${SEEN[$TAG]} both write ${TAG}"
    COLLIDE=1
  fi
  SEEN[$TAG]="${ARM} fold ${K}"
  CK="$CHECKPOINT_KEEP_DIR/checkpoint-best-${TAG}.pth"
  if [[ -f "$CK" && "$FORCE" != "1" ]]; then
    echo "EXISTS     ${ARM} fold ${K}  ->  $(basename "$CK")"
  fi
done; done
[[ "$COLLIDE" -eq 0 ]] || { echo; echo "name collisions detected, nothing submitted"; exit 1; }

echo
echo "alignment three sweep"
echo "  repo    $REPO_ROOT"
echo "  worker  $WORKER"
echo "  gpu ${GPU_MODEL}  batch ${BATCH_SIZE}/${FEAT_BATCH_SIZE}  epochs ${EPOCHS}  lr ${LR}"
echo "  arms    ${ARMS[*]}   folds ${FOLDS[*]}"
echo "  jobs    $(( ${#ARMS[@]} * ${#FOLDS[@]} ))"
echo

for ARM in "${ARMS[@]}"; do
for K in "${FOLDS[@]}"; do

    case "$ARM" in
      a3ce) HA_L=0.0 ; FEAT_L=0.0 ; RULE=both ;;
      a3ha) HA_L=$LAM; FEAT_L=0.0 ; RULE=both ;;
      a3ft) HA_L=0.0 ; FEAT_L=$LAM; RULE=both ;;
      *) echo "unknown arm $ARM"; exit 1 ;;
    esac

    ETAG="${ARM}fold${K}"
    TAG=$(predict_tag "$ARM" "$K" "$ETAG")
    RUN_NAME="${ARM}__gap__fold${K}__ep${EPOCHS}"

    JID=$(EXPERIMENT_TAG="${ETAG}" \
      BATCH_SIZE=${BATCH_SIZE} FEAT_BATCH_SIZE=${FEAT_BATCH_SIZE} \
      EPOCHS=${EPOCHS} LR=${LR} POOLING=${POOLING} \
      HA_LAMBDA=${HA_L} DAL_LAMBDA=${DAL_LAMBDA} \
      HA_LOSS_TYPE=${HA_LOSS_TYPE} HA_FP_WEIGHT=${HA_FP_WEIGHT} \
      HA_START_EPOCH=${HA_START_EPOCH} \
      FEAT_LAMBDA=${FEAT_L} FEAT_RULE=${RULE} FOLD=${K} \
      CSV_PATH="$DATA_DIR/a3_fold${K}.csv" \
      FEAT_VAL_CSV="$DATA_DIR/a3_fold${K}_heldout.csv" \
      ROOT_PATH="$ROOT_PATH" NB_CLASSES=${NB_CLASSES} \
      INIT_CHECKPOINT="$INIT_CHECKPOINT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" CHECKPOINT_KEEP_DIR="$CHECKPOINT_KEEP_DIR" \
      WANDB_NAME=${RUN_NAME} WANDB_PROJECT=${WANDB_PROJECT} \
      WANDB_MODE=online DEBUG_BATCHES=0 \
      sbatch --parsable \
        --job-name="${RUN_NAME}" \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --gres="gpu:${GPU_MODEL}:1" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${WALLTIME}" \
        "$WORKER")

    echo "  ${JID}  ${ARM} fold ${K}  ha=${HA_L} feat=${FEAT_L}"
    echo "          -> checkpoint-best-${TAG}.pth"
    sleep 1

done; done

echo
echo "watch:  squeue -u \$USER -o '%.10i %.42j %.8T %.10M %R'"
echo "verify: ls -1 $CHECKPOINT_KEEP_DIR/*.pth | wc -l    # expect 15"