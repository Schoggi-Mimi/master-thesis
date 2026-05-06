#!/bin/bash
#SBATCH --job-name=cam_eval
#SBATCH --output=logs/cam_eval_%j.out
#SBATCH --error=logs/cam_eval_%j.err
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

REPO_DIR="$HOME/projects/master-thesis"
cd "$REPO_DIR"

# mkdir -p logs

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate thesis

python -m scripts.eval_cam_metrics_panderm \
  --csv data/HAM10000/ham_test_for_cam.csv \
  --image_col image_rel_path \
  --img_dir data/HAM10000 \
  --checkpoint external/checkpoints/checkpoint-best-seggate.pth \
  --checkpoint_model_type seggate \
  --use_seg_gate \
  --seg_gate_bg_keep 0.05 \
  --class_preset ham \
  --out_dir outputs/metrics_seggate_100_topk3_final \
  --num_samples 100 \
  --device cuda \
  --compare_mode gt_topk_non_target \
  --topk_compare 3 \
  --mask_root data/HAM10000 \
  --mask_col mask_rel_path \
  --perturbation_steps 0.1 \
  --methods gate_weighted_gradcam_target,gate_weighted_gradcam_reference,gate_weighted_gradcam_diff,gate_weighted_finercam,pred_seg_gate


# python -m scripts.eval_cam_metrics_panderm \
#   --csv data/HAM10000/ham_test_for_cam.csv \
#   --image_col image_rel_path \
#   --img_dir data/HAM10000 \
#   --checkpoint external/checkpoints/checkpoint-best-dal-ha.pth \
#   --checkpoint_model_type panderm \
#   --class_preset ham \
#   --out_dir outputs/metrics_dal_ha_100_topk3_final \
#   --num_samples 100 \
#   --device cuda \
#   --compare_mode gt_topk_non_target \
#   --topk_compare 3 \
#   --mask_root data/HAM10000 \
#   --mask_col mask_rel_path \
#   --perturbation_steps 0.1 \
#   --methods gradcam_target,gradcam_reference,gradcam_diff,finercam