#!/bin/bash
set -euo pipefail

# =============================================================================
# DAL sweep, second pass.
#
# Different question from the first sweep. That one asked whether DAL improves
# spatial alignment against lesion masks. It does not.
#
# This one asks whether DAL restores CLASS DISCRIMINATIVENESS, which is what
# L_DAL was designed for and which was never measured.
#
# Primary readout is the correlation between the MEL and NV attention maps.
# Currently r = 0.968 at DAL = 0. Target is a value near 0 while MEL union
# AUC stays above 0.85 and test AUC-ROC stays above 0.93.
# =============================================================================

GPU_MODEL=${GPU_MODEL:-h100}
QOS=${QOS:-job_gratis}
PARTITION=${PARTITION:-gpu}
CPUS=${CPUS:-4}
MEM=${MEM:-32G}
WALLTIME=${WALLTIME:-6:00:00}

EPOCHS=10
POOLING=mean
HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0
DAL_MODE=all_classes
DAL_TOPK=2
LR=1e-5
NB_CLASSES=2
BATCH_SIZE=64
CSV_PATH=../data/HAM10000/mel_nv/ham_mel_nv_clean.csv
ROOT_PATH=../data/HAM10000
OUTPUT_ROOT=../outputs/dal_disc_sweep
INIT_CHECKPOINT=../external/checkpoints5/checkpoint-best-gap-ha5.pth
CHECKPOINT_KEEP_DIR=../external/checkpoints_dal
WANDB_PROJECT=master-thesis-mel-nv

# "HA_LAMBDA:DAL_LAMBDA" pairs
RUNS=(
  # already done
  # "5.0:0.0"
  # "5.0:0.1"
  # "5.0:0.5"
  # "5.0:1.0"
  # "5.0:2.0"
  # "5.0:5.0"
  # "0.0:2.0"

  # missing control. DAL at 0.001 rather than 0.0 so the HA code path
  # stays active and the separability metrics are still logged.
  "0.0:0.001"
)

echo "DAL discriminativeness sweep, GPU ${GPU_MODEL}"

for PAIR in "${RUNS[@]}"; do
    HA_LAMBDA="${PAIR%%:*}"
    DAL_LAMBDA="${PAIR##*:}"

    HA_TAG=$(echo "$HA_LAMBDA"  | sed 's/\./p/')
    DAL_TAG=$(echo "$DAL_LAMBDA" | sed 's/\./p/')
    RUN_NAME="disc__gap__ha${HA_TAG}__dal${DAL_TAG}__ep${EPOCHS}"

    JID=$(BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} \
      HA_LAMBDA=${HA_LAMBDA} HA_START_EPOCH=${HA_START_EPOCH} \
      HA_FP_WEIGHT=${HA_FP_WEIGHT} HA_LOSS_TYPE=${HA_LOSS_TYPE} \
      DAL_LAMBDA=${DAL_LAMBDA} DAL_MODE=${DAL_MODE} DAL_TOPK=${DAL_TOPK} \
      INIT_CHECKPOINT=${INIT_CHECKPOINT} POOLING=${POOLING} LR=${LR} \
      CSV_PATH=${CSV_PATH} ROOT_PATH=${ROOT_PATH} NB_CLASSES=${NB_CLASSES} \
      OUTPUT_ROOT=${OUTPUT_ROOT} CHECKPOINT_KEEP_DIR=${CHECKPOINT_KEEP_DIR} \
      WANDB_NAME=${RUN_NAME} WANDB_PROJECT=${WANDB_PROJECT} \
      sbatch --parsable \
        --job-name="${RUN_NAME}" \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --gres="gpu:${GPU_MODEL}:1" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${WALLTIME}" \
        run_panderm_full_finetune_ha.sh)

    echo "  ${JID}  ${RUN_NAME}  HA=${HA_LAMBDA} DAL=${DAL_LAMBDA}"
    sleep 2
done

echo "submitted. watch: squeue --me -o '%.10i %.40j %.10T %.20S %.25R'"