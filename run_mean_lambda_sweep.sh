#!/bin/bash
set -euo pipefail

EPOCHS=10
POOLING=mean
HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0
DAL_LAMBDA=0.0
LR=1e-5
NB_CLASSES=2
BATCH_SIZE=64
CSV_PATH=../data/HAM10000/mel_nv/ham_mel_nv_clean.csv
ROOT_PATH=../data/HAM10000
OUTPUT_ROOT=../outputs/ha_sweep
WANDB_PROJECT=master-thesis-mel-nv

# Canonical lambda grid
HA_LAMBDA_VALUES=(0.0 0.1 0.25 0.5 1.0 3.0 5.0 7.0 11.0 17.0)

echo "Launching MEAN pooling lambda sweep"

for HA_LAMBDA in "${HA_LAMBDA_VALUES[@]}"; do

    # Format: sweep_ha__mean__l{lambda}__ep{epochs}__lr{lr}
    # Double underscore separates fields. Lambda uses 'p' for decimal point.
    LAMBDA_TAG=$(echo "$HA_LAMBDA" | sed 's/\./p/')
    RUN_NAME="sweep_ha__mean__l${LAMBDA_TAG}__ep${EPOCHS}__lr1e5"
    RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"

    echo "  Submitting: ${RUN_NAME}"

    BATCH_SIZE=${BATCH_SIZE} \
    EPOCHS=${EPOCHS} \
    HA_LAMBDA=${HA_LAMBDA} \
    HA_START_EPOCH=${HA_START_EPOCH} \
    HA_FP_WEIGHT=${HA_FP_WEIGHT} \
    HA_LOSS_TYPE=${HA_LOSS_TYPE} \
    DAL_LAMBDA=${DAL_LAMBDA} \
    POOLING=${POOLING} \
    LR=${LR} \
    CSV_PATH=${CSV_PATH} \
    ROOT_PATH=${ROOT_PATH} \
    NB_CLASSES=${NB_CLASSES} \
    OUTPUT_DIR=${RUN_OUTPUT} \
    WANDB_NAME=${RUN_NAME} \
    WANDB_PROJECT=${WANDB_PROJECT} \
    sbatch run_panderm_full_finetune_ha.sh

    sleep 2
done

echo "All MEAN pooling jobs submitted."