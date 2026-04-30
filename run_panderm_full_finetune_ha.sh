#!/bin/bash
#SBATCH --job-name=panderm_ha_lam1_from_nomix_weighted
#SBATCH --output=logs/panderm_ha_lam1_from_nomix_weighted_%j.out
#SBATCH --error=logs/panderm_ha_lam1_from_nomix_weighted_%j.err
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

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

python -m run_panderm_full_finetune_ha \
  --panderm-classification-dir ../external/PanDerm/classification \
  --csv-path ../data/HAM10000/ham_segmentation_overlap.csv \
  --root-path ../data/HAM10000 \
  --image-key image_rel_path \
  --mask-key mask_rel_path \
  --pretrained-checkpoint ../external/weights/panderm_bb_data6_checkpoint-499.pth \
  --init-checkpoint ../outputs/panderm_full_finetune/ham_base_nomix_weighted/checkpoint-best.pth \
  --output-dir ../outputs/panderm_HA_lam1_from_nomix_weighted \
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
  --ha-lambda 1.0 \
  --ha-loss-type paper_dice \
  --debug-batches 0 \
  --disable-amp