#!/bin/bash
#SBATCH --job-name=panderm_ha
#SBATCH --output=logs/panderm_ha_%j.out
#SBATCH --error=logs/panderm_ha_%j.err
#SBATCH --time=6:00:00
#SBATCH --mail-user=choekyel.nyungmartsang@students.unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gratis


REPO_ROOT="$HOME/projects/master-thesis"
REPO_DIR="$REPO_ROOT/scripts"
cd "$REPO_DIR"

# Optional local config file for W&B or cluster-specific settings.
# Keep this file out of git.
if [[ -f "$REPO_ROOT/config/local.env" ]]; then
  source "$REPO_ROOT/config/local.env"
fi

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

# Toggle the explanation losses here.
# You can either edit the defaults below or override them from the command line.
#
# Examples:
#   DAL only, weak:
#     HA_LAMBDA=0.0 DAL_LAMBDA=0.1 sbatch run_panderm_full_finetune_ha.sh
#   HA + DAL, weak:
#     HA_LAMBDA=1.0 DAL_LAMBDA=0.1 sbatch run_panderm_full_finetune_ha.sh
#   DAL only, paper-like strength:
#     HA_LAMBDA=0.0 DAL_LAMBDA=1.0 sbatch run_panderm_full_finetune_ha.sh
#   Top-k DAL instead of all classes:
#     HA_LAMBDA=1.0 DAL_LAMBDA=0.1 DAL_MODE=topk_non_target DAL_TOPK=3 sbatch run_panderm_full_finetune_ha.sh
#
# Current default if you simply run:
#   sbatch run_panderm_full_finetune_ha.sh
#
# For binary MEL/NV fine-tuning, override at submission time, for example:
#   CSV_PATH=../data/HAM10000/mel_nv/ham_mel_nv_clean.csv \
#   NB_CLASSES=2 EXPERIMENT_TAG=mel_nv_clean POOLING=mean \
#   OUTPUT_ROOT=../outputs/mel_nv sbatch run_panderm_full_finetune_ha.sh
#
# PATCH 8: BATCH_SIZE is now controllable from the environment.
# Example pilot run: BATCH_SIZE=8 EPOCHS=1 sbatch run_panderm_full_finetune_ha.sh
BATCH_SIZE=${BATCH_SIZE:-64}
POOLING=${POOLING:-mean}
HA_LAMBDA=${HA_LAMBDA:-0.5}
HA_START_EPOCH=${HA_START_EPOCH:-0}
HA_FP_WEIGHT=${HA_FP_WEIGHT:-1.0}
DAL_LAMBDA=${DAL_LAMBDA:-0.0}
DAL_MODE=${DAL_MODE:-all_classes}
DAL_TOPK=${DAL_TOPK:-3}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-5}
MIXUP=${MIXUP:-0.0}
CUTMIX=${CUTMIX:-0.0}
SMOOTHING=${SMOOTHING:-0.0}

HA_LOSS_TYPE=${HA_LOSS_TYPE:-paper_dice}
CHECKPOINT_KEEP_DIR=${CHECKPOINT_KEEP_DIR:-../external/checkpoints4}
if [[ "$POOLING" == "cls" ]]; then
  POOLING_TAG="cls"
elif [[ "$POOLING" == "mean" ]]; then
  POOLING_TAG="gap"
else
  echo "Unsupported POOLING=$POOLING. Use POOLING=cls or POOLING=mean."
  exit 1
fi

CSV_PATH=${CSV_PATH:-../data/HAM10000/ham_segmentation_overlap.csv}
ROOT_PATH=${ROOT_PATH:-../data/HAM10000}
IMAGE_KEY=${IMAGE_KEY:-image_rel_path}
MASK_KEY=${MASK_KEY:-mask_rel_path}
NB_CLASSES=${NB_CLASSES:-7}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-foundation}
OUTPUT_ROOT=${OUTPUT_ROOT:-../outputs}

WANDB_PROJECT=${WANDB_PROJECT:-master-thesis-mel-nv}

if [[ "$HA_LAMBDA" == "0.0" || "$HA_LAMBDA" == "0" ]]; then
  if [[ "$DAL_LAMBDA" == "0.0" || "$DAL_LAMBDA" == "0" ]]; then
    LOSS_TAG="${EXPERIMENT_TAG}_${POOLING_TAG}_ce_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}"
  else
    LOSS_TAG="${EXPERIMENT_TAG}_${POOLING_TAG}_dal${DAL_LAMBDA}_${DAL_MODE}_top${DAL_TOPK}_start${HA_START_EPOCH}_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}"
  fi
else
  if [[ "$DAL_LAMBDA" == "0.0" || "$DAL_LAMBDA" == "0" ]]; then
    LOSS_TAG="${EXPERIMENT_TAG}_${POOLING_TAG}_ha${HA_LAMBDA}_start${HA_START_EPOCH}_${HA_LOSS_TYPE}_fp${HA_FP_WEIGHT}_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}"
  else
    LOSS_TAG="${EXPERIMENT_TAG}_${POOLING_TAG}_ha${HA_LAMBDA}_start${HA_START_EPOCH}_${HA_LOSS_TYPE}_fp${HA_FP_WEIGHT}_dal${DAL_LAMBDA}_${DAL_MODE}_top${DAL_TOPK}_classes${NB_CLASSES}_ep${EPOCHS}_lr${LR}"
  fi
fi

WANDB_NAME=${WANDB_NAME:-${LOSS_TAG}}

export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_NAME="$WANDB_NAME"
export WANDB_ENTITY=${WANDB_ENTITY:-choekyel-nyungmartsang-university-of-bern}
export WANDB_MODE=${WANDB_MODE:-online}

python -m run_panderm_full_finetune_ha \
  --panderm-classification-dir ../external/PanDerm/classification \
  --csv-path "$CSV_PATH" \
  --root-path "$ROOT_PATH" \
  --image-key "$IMAGE_KEY" \
  --mask-key "$MASK_KEY" \
  --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
  --output-dir "$OUTPUT_ROOT/panderm_${LOSS_TAG}" \
  --model PanDerm_Base_FT \
  --nb-classes "$NB_CLASSES" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay 0.05 \
  --warmup-epochs 1 \
  --layer-decay 0.65 \
  --drop-path 0.2 \
  --update-freq 1 \
  --weights \
  --monitor recall \
  --pooling "$POOLING" \
  --device cuda \
  --num-workers 4 \
  --mixup "$MIXUP" \
  --cutmix "$CUTMIX" \
  --smoothing "$SMOOTHING" \
  --ha-lambda "$HA_LAMBDA" \
  --ha-start-epoch "$HA_START_EPOCH" \
  --ha-loss-type "$HA_LOSS_TYPE" \
  --ha-fp-weight "$HA_FP_WEIGHT" \
  --dal-lambda "$DAL_LAMBDA" \
  --dal-mode "$DAL_MODE" \
  --dal-topk "$DAL_TOPK" \
  --init-checkpoint "${INIT_CHECKPOINT:-}" \
  --debug-batches 0 \
  --wandb-name "$WANDB_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE" \
  --disable-color-jitter \
  --disable-amp

BEST_CKPT="$OUTPUT_ROOT/panderm_${LOSS_TAG}/checkpoint-best.pth"
if [[ -f "$BEST_CKPT" ]]; then
  mkdir -p "$CHECKPOINT_KEEP_DIR"
  KEEP_CKPT="$CHECKPOINT_KEEP_DIR/checkpoint-best-${LOSS_TAG}.pth"
  cp "$BEST_CKPT" "$KEEP_CKPT"
  echo "Copied best checkpoint to: $KEEP_CKPT"
else
  echo "WARNING: best checkpoint not found at: $BEST_CKPT"
fi