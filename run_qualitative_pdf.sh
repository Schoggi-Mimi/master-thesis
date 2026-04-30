#!/bin/bash
set -euo pipefail

# Run locally from the repository root:
#   bash run_qualitative_pdf.sh
#
# This combines already-generated CAM panel PNGs into one comparison PDF.
# It does not need a GPU or SLURM.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

python -m scripts.make_qualitative_comparison_pdf \
  --csv data/HAM10000/ham_test_cam_qualitative_stratified_10.csv \
  --image_col image_rel_path \
  --gt_col gt_label \
  --out_pdf outputs/qual/qualitative_comparison_gt_topk3_baseline_ha_seggate.pdf \
  --experiments_json_path configs/qualitative_gt_topk3_baseline_ha_seggate.json \
  --num_samples 10 \
  --missing_policy placeholder