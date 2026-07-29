#!/bin/bash
set -euo pipefail

EPOCHS=10
POOLING=mean
HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0
HA_LAMBDA=5.0
DAL_MODE=all_classes
DAL_TOPK=2
LR=1e-5
NB_CLASSES=2
BATCH_SIZE=64
CSV_PATH=../data/HAM10000/mel_nv/ham_mel_nv_clean.csv
ROOT_PATH=../data/HAM10000
OUTPUT_ROOT=../outputs/dal_sweep
INIT_CHECKPOINT=../external/checkpoints5/checkpoint-best-gap-ha5.pth
WANDB_PROJECT=master-thesis-mel-nv

# DAL lambda grid — 0.0 is HA-only baseline for direct comparison
DAL_LAMBDA_VALUES=(0.0 0.1 0.5 1.0 2.0 5.0)

echo "Launching DAL sweep — GAP model — HA lambda fixed at ${HA_LAMBDA}"

for DAL_LAMBDA in "${DAL_LAMBDA_VALUES[@]}"; do

    LAMBDA_TAG=$(echo "$DAL_LAMBDA" | sed 's/\./p/')
    RUN_NAME="sweep_dal__gap__hal5__dall${LAMBDA_TAG}__ep${EPOCHS}__lr1e5"

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
    INIT_CHECKPOINT=${INIT_CHECKPOINT} \
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

echo "All DAL sweep jobs submitted."