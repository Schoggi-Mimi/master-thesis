#!/bin/bash
#SBATCH --job-name=panderm_ha
#SBATCH --output=logs/panderm_ha_%j.out
#SBATCH --error=logs/panderm_ha_%j.err
#SBATCH --time=12:00:00
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
HA_LAMBDA=${HA_LAMBDA:-0.5}
HA_START_EPOCH=${HA_START_EPOCH:-0}
DAL_LAMBDA=${DAL_LAMBDA:-0.0}
DAL_MODE=${DAL_MODE:-all_classes}
DAL_TOPK=${DAL_TOPK:-3}

WANDB_PROJECT=${WANDB_PROJECT:-master-thesis-panderm-ha}
WANDB_NAME=${WANDB_NAME:-ha${HA_LAMBDA}_start${HA_START_EPOCH}_dal${DAL_LAMBDA}}

if [[ "$HA_LAMBDA" == "0.0" || "$HA_LAMBDA" == "0" ]]; then
  LOSS_TAG="dal${DAL_LAMBDA}_start${HA_START_EPOCH}"
else
  LOSS_TAG="ha${HA_LAMBDA}_start${HA_START_EPOCH}_dal${DAL_LAMBDA}"
fi

export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_NAME="$WANDB_NAME"
export WANDB_MODE=${WANDB_MODE:-online}

python -m run_panderm_full_finetune_ha \
  --panderm-classification-dir ../external/PanDerm/classification \
  --csv-path ../data/HAM10000/ham_segmentation_overlap.csv \
  --root-path ../data/HAM10000 \
  --image-key image_rel_path \
  --mask-key mask_rel_path \
  --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
  --init-checkpoint ../outputs/panderm_full_finetune/ham_base_nomix_weighted/checkpoint-best.pth \
  --output-dir ../outputs/panderm_${LOSS_TAG} \
  --model PanDerm_Base_FT \
  --nb-classes 7 \
  --batch-size 64 \
  --epochs 10 \
  --lr 1e-5 \
  --weight-decay 0.05 \
  --warmup-epochs 1 \
  --layer-decay 0.65 \
  --drop-path 0.2 \
  --update-freq 1 \
  --weights \
  --monitor recall \
  --device cuda \
  --num-workers 4 \
  --ha-lambda "$HA_LAMBDA" \
  --ha-start-epoch "$HA_START_EPOCH" \
  --ha-loss-type paper_dice \
  --dal-lambda "$DAL_LAMBDA" \
  --dal-mode "$DAL_MODE" \
  --dal-topk "$DAL_TOPK" \
  --debug-batches 0 \
  --wandb-name "$WANDB_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --disable-amp