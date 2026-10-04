#!/bin/bash
set -euo pipefail
# =============================================================================
# ALIGNMENT ONE, rerun on the alignment two folds.
#
# WHY. The original A1 checkpoints were trained with all 34 annotated images
# in the TEST split. A2 trains with 26 of them in TRAIN. Any A1 versus A2
# difference therefore confounds target type with data split.
#
# This rerun holds the split fixed. Same fold CSVs, same held out images,
# same 987 image test set, same ha0 warm start. The only thing that differs
# from A2 is the target: Tschandl lesion mask, ground truth class channel
# only, every training image.
#
# EXPERIMENT_TAG carries the fold because the A1 branch of LOSS_TAG in the
# worker script has no fold field. Without it all five runs collide.
#
# These runs log no heldout_feat_* metrics because a2_active is false. That
# is expected. All headline numbers come from the 14c evaluation notebook,
# which uses one evaluation path for every checkpoint.
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
WALLTIME=${WALLTIME:-3:00:00}

BATCH_SIZE=${BATCH_SIZE:-48}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-5}
POOLING=mean
NB_CLASSES=2
HA_LOSS_TYPE=paper_dice
HA_FP_WEIGHT=1.0
HA_START_EPOCH=0

DATA_DIR="$REPO_ROOT/data/HAM10000/mel_nv"
ROOT_PATH="$REPO_ROOT/data/HAM10000"
OUTPUT_ROOT="$REPO_ROOT/outputs/alignment2"
INIT_CHECKPOINT="$REPO_ROOT/external/checkpoints5/checkpoint-best-gap-ha0.pth"
CHECKPOINT_KEEP_DIR="$REPO_ROOT/external/checkpoints_a2"
WANDB_PROJECT=master-thesis-mel-nv

# matched to the A2 lambda grid so the comparison is like for like
HA_LAMBDAS=(${HA_LAMBDAS_OVERRIDE:-1.0 5.0})
FOLDS=(${FOLDS_OVERRIDE:-0 1 2 3 4})

[[ -f "$INIT_CHECKPOINT" ]] || { echo "MISSING $INIT_CHECKPOINT"; exit 1; }
for K in "${FOLDS[@]}"; do
  [[ -f "$DATA_DIR/a2_fold${K}.csv" ]] || { echo "MISSING a2_fold${K}.csv"; exit 1; }
done
mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_KEEP_DIR"

echo "alignment one, fold matched rerun"
echo "  gpu ${GPU_MODEL}  batch ${BATCH_SIZE}  epochs ${EPOCHS}  lr ${LR}"
echo "  ha lambdas: ${HA_LAMBDAS[*]}"
echo "  jobs $(( ${#HA_LAMBDAS[@]} * ${#FOLDS[@]} ))"
echo

for HL in "${HA_LAMBDAS[@]}"; do
for K in "${FOLDS[@]}"; do

    HT=$(echo "$HL" | sed 's/\./p/')
    TAG="a1fold${K}"
    RUN_NAME="a1__gap__ha${HT}__fold${K}__ep${EPOCHS}"

    JID=$(BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} LR=${LR} POOLING=${POOLING} \
      HA_LAMBDA=${HL} HA_START_EPOCH=${HA_START_EPOCH} \
      HA_LOSS_TYPE=${HA_LOSS_TYPE} HA_FP_WEIGHT=${HA_FP_WEIGHT} \
      DAL_LAMBDA=0.0 FEAT_LAMBDA=0.0 FOLD=${K} \
      EXPERIMENT_TAG=${TAG} \
      CSV_PATH="$DATA_DIR/a2_fold${K}.csv" \
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

done; done

echo
echo "watch: squeue -u \$USER -o '%.10i %.42j %.8T %.10M %R'"