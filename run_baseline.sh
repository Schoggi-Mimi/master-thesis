#!/bin/bash
#SBATCH --job-name=thesis_smoke
#SBATCH --output=logs/thesis_smoke_%j.out
#SBATCH --error=logs/thesis_smoke_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gratis

echo "============================================================"
echo "Job started on: $(hostname)"
echo "Start time: $(date)"
echo "Working directory before cd: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "============================================================"

REPO_DIR="$HOME/projects/master-thesis/scripts"
cd "$REPO_DIR"

mkdir -p ../logs
mkdir -p ../outputs/checkpoints ../outputs/metrics ../outputs/figures ../outputs/predictions

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

nvidia-smi || true

python train_baseline.py \
  --split-dir ../data/processed/splits \
  --output-dir ../outputs \
  --model-name convnext_tiny \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 4 \
  --base-epochs 8 \
  --ft-epochs 6 \
  --base-lr 3e-4 \
  --ft-lr 1e-4 \
  --weight-decay 1e-4 \
  --label-smoothing 0.05

echo "============================================================"
echo "End time: $(date)"
echo "Job finished"
echo "============================================================"