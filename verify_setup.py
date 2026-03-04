"""
Quick verification script to check repo structure + key imports.

Run:
  python verify_setup.py
"""

from __future__ import annotations

import os
import sys


def check_file(path: str, description: str) -> bool:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_directory(path: str, description: str) -> bool:
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_import(module: str, description: str) -> bool:
    try:
        __import__(module)
        print(f"✓ Import OK: {description} ({module})")
        return True
    except Exception as e:
        print(f"✗ Import FAIL: {description} ({module}) -> {type(e).__name__}: {e}")
        return False


def main() -> None:
    print("=" * 80)
    print("Repository Verification (Current Pipeline)")
    print("=" * 80)
    print()

    checks = []

    # Core scripts
    print("Scripts:")
    checks.append(check_file("scripts/generate_finer_cam.py", "CAM generation script"))
    checks.append(check_file("scripts/make_subsets.py", "Subset creation script"))
    print()

    # Core src modules
    print("Core source files:")
    checks.append(check_file("src/models/isic7_loader.py", "ISIC7 EfficientNet loader"))
    checks.append(check_file("src/cam/diff_cam.py", "CAM logic (diff targets)"))
    print()

    # Expected data folders (may be empty depending on setup)
    print("Data / output folders (existence only):")
    checks.append(check_directory("data/isic2018", "ISIC2018 data root"))
    checks.append(check_directory("data/isic2018/subsets", "Subset CSV folder"))
    checks.append(check_directory("external/weights", "Checkpoint folder"))
    checks.append(check_directory("outputs", "Outputs folder"))
    print()

    # Optional but helpful files
    print("Optional files:")
    check_file("data/isic2018/val_gt.csv", "Validation GT CSV (val_gt.csv)")
    check_file("external/weights/isic7_last_effnetb4.pth", "ISIC7 checkpoint")
    print()

    # Imports
    print("Python imports:")
    checks.append(check_import("torch", "PyTorch"))
    checks.append(check_import("torchvision", "Torchvision"))
    checks.append(check_import("pandas", "Pandas"))
    checks.append(check_import("numpy", "NumPy"))
    checks.append(check_import("cv2", "OpenCV"))
    checks.append(check_import("PIL", "Pillow"))
    checks.append(check_import("efficientnet_pytorch", "efficientnet_pytorch"))
    checks.append(check_import("pytorch_grad_cam", "grad-cam (pytorch-grad-cam)"))
    print()

    ok = all(checks)
    print("=" * 80)
    print("OK" if ok else "Some checks failed")
    print("=" * 80)


if __name__ == "__main__":
    main()