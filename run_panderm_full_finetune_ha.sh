#!/bin/bash
#SBATCH --job-name=panderm_ft_ha_ham
#SBATCH --output=logs/panderm_ft_ha_ham_%j.out
#SBATCH --error=logs/panderm_ft_ha_ham_%j.err
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

REPO_DIR="$HOME/projects/master-thesis/scripts"
cd "$REPO_DIR"

mkdir -p ../logs
mkdir -p ../outputs/panderm_full_finetune/ham_ha

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

# nvidia-smi || true
# python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python -m run_panderm_full_finetune_ha \
  --panderm-classification-dir ../external/PanDerm/classification \
  --csv-path ../data/HAM10000/ham_segmentation_overlap.csv \
  --root-path ../data/HAM10000 \
  --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
  --output-dir ../outputs/panderm_full_finetune/ham_ha \
  --model PanDerm_Base_FT \
  --nb-classes 7 \
  --batch-size 64 \
  --epochs 6 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 1 \
  --layer-decay 0.65 \
  --drop-path 0.2 \
  --update-freq 1 \
  --weights \
  --monitor recall \
  --wandb-name panderm_full_finetune_ham_ha \
  --device cuda \
  --image-key image_rel_path \
  --mask-key mask_rel_path \
  --num-workers 4 \
  --ha-lambda 1.0 \
  --ha-fp-weight 1.0 \
  --init-checkpoint ../outputs/panderm_full_finetune/ham/checkpoint-best.pth \
  --debug-batches 0  \
  --ha-loss-type paper_dice