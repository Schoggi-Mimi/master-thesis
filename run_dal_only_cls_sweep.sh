#!/bin/bash
set -euo pipefail

EPOCHS=10
POOLING=cls
HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0
HA_LAMBDA=0.0
DAL_MODE=all_classes
DAL_TOPK=2
LR=1e-5
NB_CLASSES=2
BATCH_SIZE=64
CSV_PATH=../data/HAM10000/mel_nv/ham_mel_nv_clean.csv
ROOT_PATH=../data/HAM10000
OUTPUT_ROOT=../outputs/dal_only_sweep
WANDB_PROJECT=master-thesis-mel-nv

DAL_LAMBDA_VALUES=(0.0 0.1 0.5 1.0 2.0 5.0)

echo "Launching DAL-only sweep — CLS model — HA completely off"

for DAL_LAMBDA in "${DAL_LAMBDA_VALUES[@]}"; do

    LAMBDA_TAG=$(echo "$DAL_LAMBDA" | sed 's/\./p/')
    RUN_NAME="sweep_dalonly__cls__ha0__dall${LAMBDA_TAG}__ep${EPOCHS}__lr1e5"

    echo "  Submitting: ${RUN_NAME}"

    BATCH_SIZE=${BATCH_SIZE} \
    EPOCHS=${EPOCHS} \
    HA_LAMBDA=${HA_LAMBDA} \
    HA_START_EPOCH=${HA_START_EPOCH} \
    HA_FP_WEIGHT=${HA_FP_WEIGHT} \
    HA_LOSS_TYPE=${HA_LOSS_TYPE} \
    DAL_LAMBDA=${DAL_LAMBDA} \
    DAL_MODE=${DAL_MODE} \
    DAL_TOPK=${DAL_TOPK} \
    POOLING=${POOLING} \
    LR=${LR} \
    CSV_PATH=${CSV_PATH} \
    ROOT_PATH=${ROOT_PATH} \
    NB_CLASSES=${NB_CLASSES} \
    OUTPUT_ROOT=${OUTPUT_ROOT} \
    WANDB_NAME=${RUN_NAME} \
    WANDB_PROJECT=${WANDB_PROJECT} \
    sbatch run_panderm_full_finetune_ha.sh

    sleep 2
done

echo "All CLS DAL-only sweep jobs submitted."