#!/bin/bash
#SBATCH --job-name=ham_bcn_baseline
#SBATCH --output=logs/ham_bcn_baseline_%j.out
#SBATCH --error=logs/ham_bcn_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Adjust for your cluster:
# module load python/3.10 cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

set -euo pipefail

echo "Job started on: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Start time: $(date)"

mkdir -p logs
mkdir -p ../outputs/checkpoints ../outputs/metrics ../outputs/figures ../outputs/predictions

nvidia-smi || true

python train_baseline.py   --split-dir ../data/processed/splits   --output-dir ../outputs   --model-name convnext_tiny   --image-size 224   --batch-size 32   --num-workers 4   --base-epochs 8   --ft-epochs 6   --base-lr 3e-4   --ft-lr 1e-4   --weight-decay 1e-4   --label-smoothing 0.05

echo "End time: $(date)"
