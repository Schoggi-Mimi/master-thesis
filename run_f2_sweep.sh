#!/bin/bash
set -euo pipefail
# =============================================================================
# F2. Shelley's design. Two alignment terms trained jointly.
#
#   L = CE + alpha * HA1 + beta * HA2
#
#   HA1  Tschandl lesion mask, target class map, full train split
#   HA2  Florentia concept masks, BOTH class maps, annotated images only
#
# beta = 0 is the control. Foundation init, 10 epochs, lr 1e-5, weighted
# sampler, recall monitor. Only beta changes across the sweep.
#
# RTX 4090. Batch 32 for EVERY beta. checkpoints5 ha5 used batch 64, so the
# beta = 0 run in this sweep is the control, not the old ha5 checkpoint.
#
# Anchors keep their CE row and lose their HA2 target. HA1 now holds the
# nevus channel on the lesion, so anchors only added imbalance.
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
WALLTIME=${WALLTIME:-8:00:00}

BATCH_SIZE=${BATCH_SIZE:-32}
FEAT_BATCH_SIZE=${FEAT_BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-5}
POOLING=mean
POOLING_TAG=gap
HA_LAMBDA=${HA_LAMBDA:-5.0}
HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0
DAL_LAMBDA=0.0
NB_CLASSES=2
RULE=both

DATA_DIR="$REPO_ROOT/data/HAM10000/mel_nv"
ROOT_PATH="$REPO_ROOT/data/HAM10000"
OUTPUT_ROOT="$REPO_ROOT/outputs/f2"
CHECKPOINT_KEEP_DIR="$REPO_ROOT/external/checkpoints_f2"
LOG_DIR="$REPO_ROOT/logs"
WANDB_PROJECT=master-thesis-mel-nv

BETAS=(${BETAS_OVERRIDE:-0.0 1.0 5.0 20.0})
FOLDS=(${FOLDS_OVERRIDE:-0 1 2 3 4})
FORCE=${FORCE:-0}

# ---- guards ------------------------------------------------------------------
if [[ "$GPU_MODEL" != "h100" && "$BATCH_SIZE" -gt 32 ]]; then
  echo "GPU ${GPU_MODEL} with BATCH_SIZE ${BATCH_SIZE} will likely run out of memory."
  exit 1
fi
[[ -d "$ROOT_PATH" ]] || { echo "MISSING $ROOT_PATH"; exit 1; }
for K in "${FOLDS[@]}"; do
  [[ -f "$DATA_DIR/f2_fold${K}.csv"         ]] || { echo "MISSING f2_fold${K}.csv"; exit 1; }
  [[ -f "$DATA_DIR/f2_fold${K}_heldout.csv" ]] || { echo "MISSING f2_fold${K}_heldout.csv"; exit 1; }
  N_TRAIN=$(awk -F, 'NR>1 && $0 ~ /,train,/ {c++} END{print c+0}' "$DATA_DIR/f2_fold${K}.csv")
  echo "  fold ${K} train rows ${N_TRAIN}"
  [[ "$N_TRAIN" -gt 6000 ]] || { echo "fold ${K} has ${N_TRAIN} train rows, expected ~6350."; exit 1; }
done
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_KEEP_DIR" "$LOG_DIR"

# ---- predict the checkpoint name the worker will write ----------------------
predict_tag () {
  local B=$1 K=$2 ETAG=$3
  if [[ "$B" == "0.0" || "$B" == "0" ]]; then
    echo "${ETAG}_${POOLING_TAG}_ha${HA_LAMBDA}_start${HA_START_EPOCH}_${HA_LOSS_TYPE}_fp${HA_FP_WEIGHT}_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}"
  else
    local FT; FT=$(echo "$B" | sed 's/\./p/')
    echo "${ETAG}_${POOLING_TAG}_ha${HA_LAMBDA}_feat${FT}_${RULE}_fold${K}_ep${EPOCHS}_lr${LR}"
  fi
}

# ---- collision check BEFORE submitting anything -----------------------------
declare -A SEEN
COLLIDE=0
for B in "${BETAS[@]}"; do
for K in "${FOLDS[@]}"; do
  BT=$(echo "$B" | sed 's/\./p/')
  TAG=$(predict_tag "$B" "$K" "f2b${BT}fold${K}")
  if [[ -n "${SEEN[$TAG]:-}" ]]; then
    echo "COLLISION  beta ${B} fold ${K} and ${SEEN[$TAG]} both write ${TAG}"
    COLLIDE=1
  fi
  SEEN[$TAG]="beta ${B} fold ${K}"
  if [[ -f "$CHECKPOINT_KEEP_DIR/checkpoint-best-${TAG}.pth" && "$FORCE" != "1" ]]; then
    echo "EXISTS     beta ${B} fold ${K}"
  fi
done; done
[[ "$COLLIDE" -eq 0 ]] || { echo "name collisions detected, nothing submitted"; exit 1; }

echo
echo "F2 sweep   alpha ${HA_LAMBDA}   betas ${BETAS[*]}   folds ${FOLDS[*]}"
echo "  gpu ${GPU_MODEL}  batch ${BATCH_SIZE}/${FEAT_BATCH_SIZE}  epochs ${EPOCHS}  lr ${LR}"
echo "  init foundation weights, no warm start"
echo "  jobs $(( ${#BETAS[@]} * ${#FOLDS[@]} ))"
echo

for B in "${BETAS[@]}"; do
for K in "${FOLDS[@]}"; do

    BT=$(echo "$B" | sed 's/\./p/')
    ETAG="f2b${BT}fold${K}"
    TAG=$(predict_tag "$B" "$K" "$ETAG")
    RUN_NAME="f2__gap__ha${HA_LAMBDA}__beta${BT}__fold${K}"

    JID=$(EXPERIMENT_TAG="${ETAG}" \
      BATCH_SIZE=${BATCH_SIZE} FEAT_BATCH_SIZE=${FEAT_BATCH_SIZE} \
      EPOCHS=${EPOCHS} LR=${LR} POOLING=${POOLING} \
      HA_LAMBDA=${HA_LAMBDA} HA_LOSS_TYPE=${HA_LOSS_TYPE} \
      HA_FP_WEIGHT=${HA_FP_WEIGHT} HA_START_EPOCH=${HA_START_EPOCH} \
      DAL_LAMBDA=${DAL_LAMBDA} \
      FEAT_LAMBDA=${B} FEAT_RULE=${RULE} FOLD=${K} \
      CSV_PATH="$DATA_DIR/f2_fold${K}.csv" \
      FEAT_VAL_CSV="$DATA_DIR/f2_fold${K}_heldout.csv" \
      ROOT_PATH="$ROOT_PATH" NB_CLASSES=${NB_CLASSES} \
      INIT_CHECKPOINT="" \
      OUTPUT_ROOT="$OUTPUT_ROOT" CHECKPOINT_KEEP_DIR="$CHECKPOINT_KEEP_DIR" \
      WANDB_NAME=${RUN_NAME} WANDB_PROJECT=${WANDB_PROJECT} \
      WANDB_MODE=online DEBUG_BATCHES=${DEBUG_BATCHES:-0} \
      sbatch --parsable \
        --job-name="${RUN_NAME}" \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --gres="gpu:${GPU_MODEL}:1" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${WALLTIME}" \
        --output="${LOG_DIR}/f2_%j.out" \
        --error="${LOG_DIR}/f2_%j.err" \
        "$WORKER")

    echo "  ${JID}  beta ${B} fold ${K}  ->  checkpoint-best-${TAG}.pth"
    sleep 1

done; done

echo
echo "watch:   squeue -u \$USER -o '%.10i %.42j %.8T %.10M %R'"
echo "logs:    $LOG_DIR/f2_<JID>.out"
echo "verify:  ls -1 $CHECKPOINT_KEEP_DIR/*.pth | wc -l    # expect $(( ${#BETAS[@]} * ${#FOLDS[@]} ))"