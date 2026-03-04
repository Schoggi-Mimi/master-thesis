# Master Thesis: Fine-Grained Grounding for Differential Diagnosis in Dermoscopic Images (ISIC 2018)

The main goal is **fine-grained grounding**: highlighting image regions that push the model toward class **A** more than **B** (e.g., MEL vs NV).

## Repo Structure (current)

```
master-thesis/
├── scripts/
│   ├── generate_finer_cam.py        # main CAM generation script (CLI)
│   └── make_subsets.py              # create subset CSVs from GT + predictions
├── src/
│   ├── cam/
│   │   └── diff_cam.py              # CAM logic: GradCAM/LayerCAM/FinerCAM + diff targets
│   └── models/
│       ├── isic7_loader.py          # loads EfficientNet-B4 7-class checkpoint
│       └── siim_loader.py           # older/legacy loader (optional)
├── data/
│   └── isic2018/
│       ├── images_val/              # validation images (*.jpg)
│       ├── val_gt.csv               # GT CSV (one-hot columns + image id)
│       └── subsets/                 # curated subset CSVs (generated)
├── external/
│   └── weights/
│       └── isic7_last_effnetb4.pth  # checkpoint weights
└── outputs/
└── isic7_cam/                   # generated CAM panels + json metadata
```