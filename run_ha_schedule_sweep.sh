#!/bin/bash

HA_START_EPOCHS=(0 3 5 7)
HA_LAMBDAS=(0.1 0.5 1.0 2.0 5.0)

for START in "${HA_START_EPOCHS[@]}"; do
  for LAMBDA in "${HA_LAMBDAS[@]}"; do
    echo "Submitting HA sweep: lambda=${LAMBDA}, start_epoch=${START}"
    HA_LAMBDA="$LAMBDA" \
    HA_START_EPOCH="$START" \
    DAL_LAMBDA=0.0 \
    WANDB_NAME="ha${LAMBDA}_start${START}" \
    sbatch run_panderm_full_finetune_ha.sh
  done
done