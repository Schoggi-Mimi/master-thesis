#!/bin/bash
#SBATCH --job-name=panderm_ft_from_seg_ham
#SBATCH --output=logs/panderm_ft_from_seg_ham_%j.out
#SBATCH --error=logs/panderm_ft_from_seg_ham_%j.err
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
mkdir -p "$HOME/projects/master-thesis/outputs/panderm_from_seg/ham"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python run_class_finetuning_from_seg.py \
  --csv_path "$HOME/projects/master-thesis/data/HAM10000/ham_segmentation_overlap.csv" \
  --root_path "$HOME/projects/master-thesis/data/HAM10000" \
  --image_key image_rel_path \
  --seg_pretrained_checkpoint "$HOME/projects/master-thesis/outputs/ham_seg_full_10ep/436/model_best_0.ckpt" \
  --output_dir "$HOME/projects/master-thesis/outputs/panderm_from_seg/ham" \
  --model PanDerm_Large_FT \
  --nb_classes 7 \
  --batch_size 8 \
  --epochs 6 \
  --lr 5e-4 \
  --weight_decay 0.05 \
  --warmup_epochs 1 \
  --layer_decay 0.65 \
  --drop_path 0.2 \
  --update_freq 8 \
  --num_workers 4 \
  --weights \
  --monitor recall \
  --wandb_name panderm_from_seg_ham \
  --device cuda