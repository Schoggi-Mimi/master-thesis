#!/bin/bash
set -euo pipefail
# =============================================================================
# ALIGNMENT TWO. Feature mask supervision, Florentia annotations.
#
# ARMS
#   both    MEL and NV channels both supervised. Empty targets pushed to zero,
#           never filled with a complement. The hypothesis.
#   single  MEL channel only, NV unsupervised. CONTROL. Tests whether single
#           channel supervision alone reproduces the alignment one collapse.
#
# Warm started from ha0, the UNALIGNED model, so a failure cannot be confused
# with a failure to undo an existing collapse.
#
# PRIMARY READOUT
#   heldout_feat_class_corr_in_lesion. Alignment one reaches +0.97.
#   Success needs it low AND heldout_feat_dead_channel_frac near zero.
#   A dead channel is not a differential explanation.
#
# All paths absolute. The worker script cds to scripts/, this one does not,
# so relative paths would resolve against different directories.
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
WALLTIME=${WALLTIME:-2:00:00}

BATCH_SIZE=${BATCH_SIZE:-48}
FEAT_BATCH_SIZE=${FEAT_BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-5}
POOLING=mean
HA_LAMBDA=0.0
DAL_LAMBDA=0.0
NB_CLASSES=2


DATA_DIR="$REPO_ROOT/data/HAM10000/mel_nv"
ROOT_PATH="$REPO_ROOT/data/HAM10000"
OUTPUT_ROOT="$REPO_ROOT/outputs/alignment2"
INIT_CHECKPOINT="$REPO_ROOT/external/checkpoints5/checkpoint-best-gap-ha0.pth"
CHECKPOINT_KEEP_DIR="$REPO_ROOT/external/checkpoints_a2"
WANDB_PROJECT=master-thesis-mel-nv

RULES=(${RULES_OVERRIDE:-both single})
LAMBDAS=(${LAMBDAS_OVERRIDE:-1.0 5.0})
FOLDS=(${FOLDS_OVERRIDE:-0 1 2 3 4})

[[ -f "$INIT_CHECKPOINT" ]] || { echo "MISSING $INIT_CHECKPOINT"; exit 1; }
[[ -d "$ROOT_PATH" ]]       || { echo "MISSING $ROOT_PATH"; exit 1; }
for K in "${FOLDS[@]}"; do
  [[ -f "$DATA_DIR/a2_fold${K}.csv"         ]] || { echo "MISSING a2_fold${K}.csv"; exit 1; }
  [[ -f "$DATA_DIR/a2_fold${K}_heldout.csv" ]] || { echo "MISSING a2_fold${K}_heldout.csv"; exit 1; }
done
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_KEEP_DIR"

echo "alignment two sweep"
echo "  repo   $REPO_ROOT"
echo "  worker $WORKER"
echo "  gpu ${GPU_MODEL}  batch ${BATCH_SIZE}/${FEAT_BATCH_SIZE}  epochs ${EPOCHS}  lr ${LR}"
echo "  jobs $(( ${#RULES[@]} * ${#LAMBDAS[@]} * ${#FOLDS[@]} ))"
echo

for RULE in "${RULES[@]}"; do
for FL in "${LAMBDAS[@]}"; do
for K in "${FOLDS[@]}"; do

    FT=$(echo "$FL" | sed 's/\./p/')
    RUN_NAME="a2__gap__${RULE}__feat${FT}__fold${K}__ep${EPOCHS}"

    JID=$(BATCH_SIZE=${BATCH_SIZE} FEAT_BATCH_SIZE=${FEAT_BATCH_SIZE} \
      EPOCHS=${EPOCHS} LR=${LR} POOLING=${POOLING} \
      HA_LAMBDA=${HA_LAMBDA} DAL_LAMBDA=${DAL_LAMBDA} \
      FEAT_LAMBDA=${FL} FEAT_RULE=${RULE} FOLD=${K} \
      CSV_PATH="$DATA_DIR/a2_fold${K}.csv" \
      FEAT_VAL_CSV="$DATA_DIR/a2_fold${K}_heldout.csv" \
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

    echo "  ${JID}  ${RUN_NAME}"
    sleep 1

done; done; done

echo
echo "watch:  squeue -u \$USER -o '%.10i %.42j %.8T %.10M %R'"
echo "logs:   $REPO_ROOT/logs/panderm_ha_<JID>.out"