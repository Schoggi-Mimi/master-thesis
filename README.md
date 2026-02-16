# Master Thesis: Fine-Grained Dermoscopic Classification (ISIC 2018)

Code and experiments for fine-grained grounding in dermoscopic skin-lesion classification (ISIC 2018). Implements comparison-based CAM methods to highlight discriminative features between clinically similar diagnoses (e.g., melanoma vs nevus) and evaluates interpretability with quantitative + qualitative metrics.

## 📁 Folder Structure

```
master-thesis/
├── configs/                      # Configuration files
│   ├── paths.yaml               # Path configurations
│   ├── model.yaml               # Model configurations
│   └── eval.yaml                # Evaluation configurations
│
├── src/                         # Source code
│   ├── data/                    # Data loading and processing
│   │   ├── __init__.py
│   │   └── isic_dataset.py      # ISIC 2018 dataset loader
│   ├── models/                  # Model wrappers
│   │   ├── __init__.py
│   │   └── siim_inference.py    # SIIM-ISIC model inference wrapper
│   ├── eval/                    # Evaluation metrics
│   │   ├── __init__.py
│   │   └── cam_metrics.py       # CAM evaluation (confidence drop, deletion)
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── config.py            # Configuration utilities
│       └── visualization.py     # Visualization utilities
│
├── scripts/                     # Executable scripts
│   ├── run_inference.py         # Run model inference
│   ├── generate_finer_cam.py    # Generate Finer-CAM heatmaps/overlays
│   ├── generate_differential_cam.py  # Generate differential CAMs
│   └── evaluate_cams.py         # Evaluate CAM quality
│
├── notebooks/                   # Jupyter notebooks (for exploration)
│
├── data/                        # Data directory (create manually)
│   └── isic2018/               # ISIC 2018 dataset
│       ├── images/             # Image files
│       ├── labels.csv          # All labels
│       ├── train.csv           # Training split
│       ├── val.csv             # Validation split
│       └── test.csv            # Test split
│
├── external/                    # External resources (create manually)
│   ├── siim/                   # SIIM-ISIC pretrained model
│   │   ├── model.pth           # Model checkpoint
│   │   └── config.yaml         # Model config (optional)
│   └── finer_cam/              # Finer-CAM repository
│       └── ...                 # Finer-CAM code
│
├── outputs/                     # Generated outputs
│   ├── heatmaps/               # CAM heatmaps
│   ├── overlays/               # CAM overlays on images
│   ├── differential_cams/      # Differential CAM visualizations
│   └── eval_results/           # Evaluation results (JSON)
│
├── checkpoints/                 # Model checkpoints (if training)
│
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data Directory

Create the data directory and organize ISIC 2018 dataset:

```bash
mkdir -p data/isic2018/images
```

Place your ISIC 2018 images in `data/isic2018/images/` and create CSV files:
- `labels.csv`: All image labels (columns: image_name, MEL, NV, BCC, AKIEC, BKL, DF, VASC)
- `train.csv`: Training split
- `val.csv`: Validation split  
- `test.csv`: Test split

### 3. Prepare External Resources

Create directories for external models:

```bash
mkdir -p external/siim
mkdir -p external/finer_cam
```

Place your SIIM-ISIC pretrained model at `external/siim/model.pth`.

Optionally, clone Finer-CAM repository into `external/finer_cam/` for enhanced CAM generation.

### 4. Configure Paths

Update `configs/paths.yaml` if your paths differ from the defaults.

## 📊 ISIC 2018 Classes

The dataset includes 7 skin lesion types:
- **MEL**: Melanoma
- **NV**: Melanocytic nevus
- **BCC**: Basal cell carcinoma
- **AKIEC**: Actinic keratosis / Intraepithelial carcinoma
- **BKL**: Benign keratosis
- **DF**: Dermatofibroma
- **VASC**: Vascular lesion

## 🔧 Usage

### Run Inference

Run model inference on the test set:

```bash
python scripts/run_inference.py \
    --data-split test \
    --batch-size 16 \
    --output-file outputs/predictions_test.csv
```

Options:
- `--data-split`: Data split to use (train/val/test)
- `--batch-size`: Batch size for inference
- `--output-file`: Output CSV file for predictions

### Generate CAM Heatmaps and Overlays

Generate CAM visualizations using Finer-CAM (or Grad-CAM fallback):

```bash
python scripts/generate_finer_cam.py \
    --data-split test \
    --num-samples 100 \
    --colormap jet \
    --alpha 0.5
```

Options:
- `--data-split`: Data split to process (train/val/test)
- `--num-samples`: Number of samples to process (None for all)
- `--colormap`: Colormap for visualization (jet/viridis/plasma)
- `--alpha`: Overlay transparency (0-1)

Output:
- `outputs/heatmaps/{image}_{class}_heatmap.png`: CAM heatmap
- `outputs/heatmaps/{image}_{class}_overlay.png`: CAM overlay on image

### Generate Differential CAMs

Generate differential CAMs to highlight discriminative features:

#### Top-2 Mode (highlight differences between top-2 predicted classes):

```bash
python scripts/generate_differential_cam.py \
    --data-split test \
    --mode top2 \
    --num-samples 50 \
    --colormap RdBu_r \
    --alpha 0.5
```

#### Class-vs-Class Mode (compare specific class pairs):

```bash
python scripts/generate_differential_cam.py \
    --data-split test \
    --mode class_vs_class \
    --class-pairs MEL-NV BCC-AKIEC BKL-DF \
    --num-samples 50 \
    --colormap RdBu_r \
    --alpha 0.5
```

Options:
- `--mode`: top2 or class_vs_class
- `--class-pairs`: Pairs of classes to compare (format: CLASS1-CLASS2)
- `--colormap`: Use diverging colormap (RdBu_r recommended)

Output:
- `outputs/differential_cams/{image}_diff_{class1}_vs_{class2}_*.png`

### Evaluate CAM Quality

Evaluate CAMs using confidence drop and deletion metrics:

```bash
python scripts/evaluate_cams.py \
    --data-split test \
    --cam-method gradcam \
    --metrics confidence_drop deletion \
    --batch-size 16 \
    --num-samples 500
```

Options:
- `--cam-method`: CAM method name (gradcam/finer_cam)
- `--metrics`: Metrics to compute (confidence_drop, deletion, insertion)
- `--use-saved-cams`: Use pre-saved CAM heatmaps instead of generating
- `--num-samples`: Number of samples to evaluate

Output:
- `outputs/eval_results/{method}_{split}_evaluation.json`

Metrics:
- **Confidence Drop**: Measures drop in confidence when masking CAM-highlighted regions
- **Deletion**: Area under curve when progressively deleting important pixels
- **Insertion**: Area under curve when progressively inserting important pixels

## 📝 Configuration Files

### `configs/paths.yaml`
- Defines all data paths, external resources, and output directories
- Update this file to match your directory structure

### `configs/model.yaml`
- Model architecture settings (backbone, input size, etc.)
- Inference parameters (batch size, device)
- CAM generation settings

### `configs/eval.yaml`
- Evaluation metric configurations
- Differential CAM settings (class pairs)
- Visualization settings

## 🔬 Evaluation Metrics

### Confidence Drop
Measures how much the model's confidence drops when regions highlighted by the CAM are masked. Higher drop indicates better CAM quality (CAM highlights truly important regions).

### Deletion
Measures the area under the curve when progressively deleting pixels in order of CAM importance. Lower AUC indicates better CAM (confidence drops faster).

### Insertion  
Measures the area under the curve when progressively inserting pixels from black baseline. Higher AUC indicates better CAM (confidence increases faster).

## 📚 References

- **ISIC 2018 Challenge**: [https://challenge.isic-archive.com/](https://challenge.isic-archive.com/)
- **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **SIIM-ISIC Competition**: [https://www.kaggle.com/c/siim-isic-melanoma-classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)

## 📄 License

This project is for research purposes as part of a master's thesis.

## 🤝 Contributing

This is a research project. For questions or suggestions, please open an issue.

## 📧 Contact

For inquiries about this research project, please contact the repository owner.
