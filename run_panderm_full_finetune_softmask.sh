#!/bin/bash
#SBATCH --job-name=panderm_ft_softmask_base_bg095
#SBATCH --output=logs/panderm_ft_softmask_base_bg095_%j.out
#SBATCH --error=logs/panderm_ft_softmask_base_bg095_%j.err
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

REPO_DIR="$HOME/projects/master-thesis/external/PanDerm/classification"
cd "$REPO_DIR"

mkdir -p "$HOME/projects/master-thesis/logs"
mkdir -p "$HOME/projects/master-thesis/outputs/panderm_softmask"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

python run_class_finetuning_softmask.py \
  --model PanDerm_Base_FT \
  --batch_size 16 \
  --epochs 20 \
  --update_freq 1 \
  --input_size 224 \
  --drop_path 0.2 \
  --lr 1e-4 \
  --layer_decay 0.65 \
  --warmup_epochs 2 \
  --smoothing 0.0 \
  --csv_path "$HOME/projects/master-thesis/data/HAM10000/ham_segmentation_overlap.csv" \
  --root_path "$HOME/projects/master-thesis/data/HAM10000" \
  --image_key image_rel_path \
  --mask_key mask_rel_path \
  --nb_classes 7 \
  --pretrained_checkpoint "$HOME/projects/master-thesis/external/weights/panderm_bb_data6_checkpoint-499.pth" \
  --output_dir "$HOME/projects/master-thesis/outputs/panderm_softmask_base_bg095_weighted_nomix" \
  --bg_strength 0.95 \
  --weights \
  --disable_amp