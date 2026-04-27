#!/bin/bash
#SBATCH --job-name=panderm_seg_ham
#SBATCH --output=/storage/homefs/cn21m021/projects/master-thesis/logs/panderm_seg_ham_%j.out
#SBATCH --error=/storage/homefs/cn21m021/projects/master-thesis/logs/panderm_seg_ham_%j.err
#SBATCH --time=24:00:00
#SBATCH --mail-user=choekyel.nyungmartsang@students.unibe.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gratis

PROJECT_DIR="$HOME/projects/master-thesis"
SEG_DIR="$PROJECT_DIR/external/PanDerm/segmentation"
LOG_DIR="$PROJECT_DIR/logs"
OUT_DIR="$PROJECT_DIR/outputs/ham_seg_full_10ep"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

cd "$SEG_DIR"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dermseg

echo "============================================================"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PWD:  $(pwd)"
echo "============================================================"

python run.py \
  --model cae_seg \
  --dataset HAM10000 \
  --parent_path "$PROJECT_DIR/data" \
  --csv_path HAM10000/ham_segmentation_overlap.csv \
  --save_name "$OUT_DIR" \
  --log_name ham_seg_full_10ep \
  --batch_size 4 \
  --test_batch_size 4 \
  --epoch 10 \
  --lr 1e-4 \
  --pretrained "$PROJECT_DIR/external/weights/panderm_ll_data6_checkpoint-499.pth" \
  --weight_decay 1e-4 \
  --size 224 \
  --gpu 0 \
  --workers 4  \
  --no_wandb