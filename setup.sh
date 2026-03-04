#!/bin/bash
set -e

echo "=========================================="
echo "Setup (Current Thesis Pipeline)"
echo "=========================================="
echo ""

echo "Python:"
python --version
echo ""

echo "Creating directory structure..."
mkdir -p data/isic2018/images_val
mkdir -p data/isic2018/subsets
mkdir -p external/weights
mkdir -p outputs
mkdir -p notebooks
echo "✓ Directories created"
echo ""

if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install -r requirements.txt
  echo "✓ Dependencies installed"
else
  echo "No requirements.txt found. Install manually if needed:"
  echo "  pip install torch torchvision numpy pandas opencv-python Pillow efficientnet_pytorch grad-cam"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1) Put validation images into: data/isic2018/images_val/"
echo "2) Put val_gt.csv into:       data/isic2018/val_gt.csv"
echo "3) Put checkpoint into:       external/weights/isic7_last_effnetb4.pth"
echo ""
echo "Run:"
echo "  python verify_setup.py"