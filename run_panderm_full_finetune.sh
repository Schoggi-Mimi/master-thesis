#!/bin/bash
#SBATCH --job-name=panderm_ft_official_ham
#SBATCH --output=logs/panderm_ft_official_ham_%j.out
#SBATCH --error=logs/panderm_ft_official_ham_%j.err
#SBATCH --time=12:00:00
#SBATCH --mail-user=choekyel.nyungmartsang@students.unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gratis

REPO_DIR="$HOME/projects/master-thesis/scripts"
cd "$REPO_DIR"

mkdir -p ../logs
mkdir -p ../outputs/panderm_full_finetune/ham

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

nvidia-smi || true

python -m run_panderm_full_finetune_official.py \
  --panderm-classification-dir ../external/PanDerm/classification \
  --csv-path ../data/HAM10000/HAM10000.csv \
  --root-path ../data/HAM10000/images \
  --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
  --output-dir ../outputs/panderm_full_finetune/ham \
  --model PanDerm_Base_FT \
  --nb-classes 7 \
  --batch-size 128 \
  --epochs 50 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  --layer-decay 0.65 \
  --drop-path 0.2 \
  --update-freq 1 \
  --weights \
  --monitor recall \
  --wandb-name panderm_full_finetune_ham \
  --wandb-mode disabled \
  --device cuda
