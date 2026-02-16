"""
Quick verification script to check repository structure.
Run this after setup to ensure everything is in place.
"""

import os
import sys


def check_file(path, description):
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_directory(path, description):
    """Check if a directory exists."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def main():
    print("="*80)
    print("Repository Structure Verification")
    print("="*80)
    print()
    
    all_checks = []
    
    # Check config files
    print("Configuration Files:")
    all_checks.append(check_file("configs/paths.yaml", "Paths config"))
    all_checks.append(check_file("configs/model.yaml", "Model config"))
    all_checks.append(check_file("configs/eval.yaml", "Eval config"))
    print()
    
    # Check source files
    print("Source Code:")
    all_checks.append(check_file("src/data/isic_dataset.py", "ISIC dataloader"))
    all_checks.append(check_file("src/models/siim_inference.py", "SIIM model wrapper"))
    all_checks.append(check_file("src/eval/cam_metrics.py", "CAM metrics"))
    all_checks.append(check_file("src/utils/config.py", "Config utilities"))
    all_checks.append(check_file("src/utils/visualization.py", "Visualization utils"))
    print()
    
    # Check scripts
    print("Scripts:")
    all_checks.append(check_file("scripts/run_inference.py", "Inference script"))
    all_checks.append(check_file("scripts/generate_finer_cam.py", "Finer-CAM script"))
    all_checks.append(check_file("scripts/generate_differential_cam.py", "Differential CAM script"))
    all_checks.append(check_file("scripts/evaluate_cams.py", "Evaluation script"))
    print()
    
    # Check directories (created by setup)
    print("Directories (created by setup):")
    all_checks.append(check_directory("outputs/heatmaps", "Heatmaps output"))
    all_checks.append(check_directory("outputs/overlays", "Overlays output"))
    all_checks.append(check_directory("outputs/differential_cams", "Differential CAMs output"))
    all_checks.append(check_directory("outputs/eval_results", "Eval results output"))
    print()
    
    # Check directories (manual setup required)
    print("Directories (manual setup - see README):")
    check_directory("data/isic2018", "ISIC 2018 data directory")
    check_directory("external/siim", "SIIM model directory")
    check_directory("external/finer_cam", "Finer-CAM directory (optional)")
    print()
    
    # Check other files
    print("Other Files:")
    all_checks.append(check_file("requirements.txt", "Requirements file"))
    all_checks.append(check_file("README.md", "README"))
    all_checks.append(check_file("setup.sh", "Setup script"))
    print()
    
    # Summary
    print("="*80)
    total = len(all_checks)
    passed = sum(all_checks)
    print(f"Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ All checks passed! Repository structure is complete.")
        return 0
    else:
        print(f"✗ {total - passed} check(s) failed. Please run setup.sh or create missing items.")
        return 1
    
    print("="*80)


if __name__ == "__main__":
    sys.exit(main())
