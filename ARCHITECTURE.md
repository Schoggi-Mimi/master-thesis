# Architecture and Design Decisions (Current Pipeline)

## Goal
Generate comparison-based CAM explanations for dermoscopic images, especially for confusing differential diagnosis pairs (e.g., MEL vs NV).

## Key Idea: “Diff Target”
Instead of explaining class A alone, we often explain:
- **score = logit(A) − logit(B)**

This yields a heatmap of regions that push the model toward **A more than B**.

---

## Main Pipeline

### 1) Script entry point
**scripts/generate_finer_cam.py**
- Reads CLI args
- Loads model checkpoint (ISIC7 EfficientNet-B4)
- Preprocesses images
- Chooses classes A/B (top2 or fixed)
- Calls CAM generator
- Saves panel PNG + JSON metadata

### 2) Model loading
**src/models/isic7_loader.py**
- Builds EfficientNet-B4
- Replaces final classifier head to 7 classes
- Loads checkpoint safely (handles state_dict wrapper + module. prefix)
- Sets eval mode and moves to device

### 3) CAM logic
**src/cam/diff_cam.py**
- Defines targets:
  - ClassifierOutputTarget(A): score = logit(A)
  - LogitDiffTarget(A,B): score = logit(A) − logit(B)
- Computes CAM triplet:
  - CAM(A), CAM(B), CAM(diff)
- Supports:
  - GradCAM, LayerCAM, FinerCAM (depends on installed grad-cam package)

---

## Data / Outputs

### Input data
- data/isic2018/images_val/*.jpg
- data/isic2018/val_gt.csv (one-hot GT columns + image id)
- data/isic2018/subsets/*.csv (curated image lists)

### Outputs
- outputs/.../*.png (3-panel CAM images)
- outputs/.../*_meta.json (A/B, probs, settings, paths)

---

## Why this structure
- Scripts keep experiment control simple (CLI-based).
- `src/` keeps reusable logic clean: model loading and CAM computation.
- Subset generation is separated into its own script for reproducibility.