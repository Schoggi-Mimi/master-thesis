#!/bin/bash
set -euo pipefail

# =============================================================================
# HA alpha ablation, rerun for the manuscript ablation table.
#
# Why rerun. The original sweep predates the evaluate_ha logging patch, so
# val_attn_class_pearson, val_attn_class_cosine and val_token_class_cosine
# were never recorded. Those are now the primary readouts for the
# discriminativeness finding.
#
# All runs train from the FOUNDATION checkpoint, not warm started. Each is
# an independent alpha setting, matching how the original sweep was done.
#
# alpha=0 uses --force_explanation_path so it runs on the same paired-mask
# dataset, the same augmentation and the same evaluate_ha as every other
# arm. Without it the script silently switches to a different augmentation
# pipeline, which would confound the alpha=0 versus alpha>0 comparison.
#
# Deliverables
#   main table      alpha 0 and alpha 5
#   ablation table  all alpha values
# =============================================================================

GPU_MODEL=${GPU_MODEL:-h100}
QOS=${QOS:-job_gratis}
PARTITION=${PARTITION:-gpu}
CPUS=${CPUS:-4}
MEM=${MEM:-32G}
WALLTIME=${WALLTIME:-6:00:00}

# ---- must match the original sweep. verify from wandb config first ---------
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-0}
# ---------------------------------------------------------------------------

HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0
DAL_LAMBDA=0.0
DAL_MODE=all_classes
DAL_TOPK=2
NB_CLASSES=2

CSV_PATH=../data/HAM10000/mel_nv/ham_mel_nv_clean.csv
ROOT_PATH=../data/HAM10000
OUTPUT_ROOT=../outputs/ha_alpha_ablation
CHECKPOINT_KEEP_DIR=../external/checkpoints_alpha
WANDB_PROJECT=master-thesis-mel-nv

# no INIT_CHECKPOINT. every run starts from the foundation weights.

POOLINGS=(${POOLINGS:-mean})          # "mean" or "mean cls"
ALPHAS=(${ALPHAS:-0 0.1 0.25 0.5 1 3 5 7 11 17})

echo "HA alpha ablation"
echo "  poolings : ${POOLINGS[*]}"
echo "  alphas   : ${ALPHAS[*]}"
echo "  jobs     : $(( ${#POOLINGS[@]} * ${#ALPHAS[@]} ))"
echo "  epochs ${EPOCHS}  lr ${LR}  batch ${BATCH_SIZE}  seed ${SEED}"
echo

for POOLING in "${POOLINGS[@]}"; do
  if [[ "$POOLING" == "mean" ]]; then TAG=gap; else TAG=cls; fi

  for HA_LAMBDA in "${ALPHAS[@]}"; do
      HA_TAG=$(echo "$HA_LAMBDA" | sed 's/\./p/')
      RUN_NAME="abl__${TAG}__ha${HA_TAG}__ep${EPOCHS}__s${SEED}"

      # alpha=0 needs the forced path so augmentation and logging match
      if [[ "$HA_LAMBDA" == "0" || "$HA_LAMBDA" == "0.0" ]]; then
        FORCE=1
      else
        FORCE=0
      fi

      JID=$(BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} SEED=${SEED} \
        HA_LAMBDA=${HA_LAMBDA} HA_START_EPOCH=${HA_START_EPOCH} \
        HA_FP_WEIGHT=${HA_FP_WEIGHT} HA_LOSS_TYPE=${HA_LOSS_TYPE} \
        DAL_LAMBDA=${DAL_LAMBDA} DAL_MODE=${DAL_MODE} DAL_TOPK=${DAL_TOPK} \
        FORCE_EXPLANATION_PATH=${FORCE} \
        POOLING=${POOLING} LR=${LR} \
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

      echo "  ${JID}  ${RUN_NAME}  pool=${POOLING} alpha=${HA_LAMBDA} force=${FORCE}"
      sleep 2
  done
done

echo
echo "watch: squeue --me -o '%.10i %.40j %.10T %.20S %.25R'"