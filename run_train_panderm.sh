#!/bin/bash
#SBATCH --job-name=panderm
#SBATCH --output=logs/panderm_smoke_%j.out
#SBATCH --error=logs/panderm_smoke_%j.err
#SBATCH --time=12:00:00
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
mkdir -p ../outputs/panderm_base/checkpoints ../outputs/panderm_base/metrics ../outputs/panderm_base/figures ../outputs/panderm_base/predictions

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

nvidia-smi || true

python train_panderm.py \
    --split-dir ../data/processed/splits \
    --output-dir ../outputs/panderm_base \
    --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
    --image-size 224 \
    --batch-size 16 \
    --num-workers 2 \
    --base-epochs 20 \
    --ft-epochs 15 \
    --base-lr 5e-5 \
    --ft-lr 2e-5 \
    --weight-decay 0.05 \
    --label-smoothing 0.05
