#!/bin/bash
# Setup script for ISIC 2018 research repository

echo "=========================================="
echo "ISIC 2018 Research Repository Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/isic2018/images
mkdir -p external/siim
mkdir -p external/finer_cam
mkdir -p outputs/heatmaps
mkdir -p outputs/overlays
mkdir -p outputs/differential_cams
mkdir -p outputs/eval_results
mkdir -p checkpoints
mkdir -p notebooks
echo "✓ Directories created"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place ISIC 2018 dataset in data/isic2018/"
echo "   - Images in data/isic2018/images/"
echo "   - CSV files: train.csv, val.csv, test.csv"
echo ""
echo "2. Place SIIM-ISIC model in external/siim/"
echo "   - Model checkpoint: external/siim/model.pth"
echo ""
echo "3. (Optional) Clone Finer-CAM to external/finer_cam/"
echo ""
echo "4. Update configs/paths.yaml if needed"
echo ""
echo "5. Run inference:"
echo "   python scripts/run_inference.py --data-split test"
echo ""
echo "See README.md for detailed usage instructions."
echo ""
