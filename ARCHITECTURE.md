# Architecture and Design Decisions

## Overview

This repository implements a research framework for analyzing dermoscopic skin lesion images from the ISIC 2018 challenge dataset. The focus is on generating and evaluating Class Activation Maps (CAMs) to understand model decision-making, particularly for differential diagnosis between clinically similar conditions.

## Design Principles

### 1. Modularity
- **Separation of concerns**: Data loading, model inference, CAM generation, and evaluation are separate modules
- **Reusability**: Core components can be used independently or composed together
- **Extensibility**: Easy to add new CAM methods, metrics, or model architectures

### 2. Configuration-Driven
- All paths, hyperparameters, and settings are externalized to YAML configs
- Easy to switch between different experimental setups without code changes
- Version control friendly - track experimental configs alongside code

### 3. Research-Focused
- Command-line scripts for common workflows
- Jupyter notebooks for interactive exploration
- Comprehensive metrics for quantitative evaluation
- Rich visualizations for qualitative assessment

## Architecture Components

### Data Layer (`src/data/`)

**`isic_dataset.py`**: PyTorch Dataset implementation for ISIC 2018
- Handles image loading and preprocessing
- Supports train/val/test splits via CSV files
- Flexible label format (one-hot or class index)
- Standard ImageNet normalization for pretrained models
- Optional data augmentation for training

**Design Decisions**:
- Uses PyTorch's native Dataset/DataLoader for efficiency and compatibility
- Separates transforms from dataset class for flexibility
- Supports both single-image and batch processing

### Model Layer (`src/models/`)

**`siim_inference.py`**: Wrapper for SIIM-ISIC pretrained models
- Loads model checkpoints with multiple format support
- Provides unified inference interface
- Enables feature extraction for CAM generation
- Handles hook registration for gradient-based methods

**Design Decisions**:
- Model-agnostic wrapper - works with various architectures (ResNet, EfficientNet, etc.)
- Separates model loading from inference logic
- Provides both high-level (predict) and low-level (forward_with_features) APIs
- Automatic device management (CPU/GPU)

### Evaluation Layer (`src/eval/`)

**`cam_metrics.py`**: Quantitative CAM evaluation metrics
- **Confidence Drop**: Measures importance of CAM-highlighted regions
- **Deletion**: Progressive removal of important pixels
- **Insertion**: Progressive addition of important pixels

**Design Decisions**:
- Metrics are model-agnostic (work with any PyTorch model)
- Batched processing for efficiency
- Returns detailed results for analysis (not just summary statistics)
- Follows established evaluation protocols from literature

### Utilities Layer (`src/utils/`)

**`config.py`**: Configuration management
- Load YAML configs
- Ensure directory structure
- Centralized path management

**`visualization.py`**: Visualization utilities
- CAM normalization and colormapping
- Overlay generation
- Comparison plots
- Export to various formats

**Design Decisions**:
- Pure utility functions (no state)
- Support for multiple colormaps
- Consistent visual style across outputs

### Scripts Layer (`scripts/`)

Command-line entry points for common workflows:
1. **`run_inference.py`**: Batch inference with result saving
2. **`generate_finer_cam.py`**: CAM generation and visualization
3. **`generate_differential_cam.py`**: Differential CAM analysis
4. **`evaluate_cams.py`**: Quantitative metric computation

**Design Decisions**:
- CLI-first design for reproducibility
- Progress bars for long-running operations
- Results saved in standard formats (CSV, JSON, PNG)
- Extensive command-line options for flexibility

## CAM Methods

### Grad-CAM (Baseline)
Standard gradient-based CAM:
1. Forward pass to get feature maps
2. Backward pass for target class
3. Weight feature maps by gradient importance
4. Apply ReLU and normalize

### Finer-CAM (Extension)
Enhanced CAM with refinement:
- Placeholder for Finer-CAM integration
- Falls back to Grad-CAM if not available
- Designed for future enhancement

### Differential CAM (Novel)
Comparison-based CAM analysis:
- **Top-2 mode**: Compare top-2 predicted classes
- **Class-vs-class mode**: Compare specific diagnostic pairs
- Highlights discriminative features between classes
- Useful for fine-grained diagnosis (e.g., melanoma vs nevus)

## Evaluation Metrics

### Why These Metrics?

1. **Confidence Drop**: Validates that CAM highlights truly important regions
   - High drop = good CAM (masking important regions reduces confidence)
   - Widely used in XAI literature

2. **Deletion**: Measures CAM faithfulness
   - Faster confidence drop = better CAM
   - Area under curve metric

3. **Insertion**: Complementary to deletion
   - Faster confidence increase = better CAM
   - Tests if CAM identifies sufficient information

### Metric Design
- All metrics use the same model that generated predictions
- Avoid introducing new uncertainty
- Results comparable across different CAM methods
- Efficient batched implementation

## Configuration System

### Three-tier Configuration

1. **`paths.yaml`**: Data and model locations
   - Easy to adapt to different machines/clusters
   - Centralized path management
   - Class label definitions

2. **`model.yaml`**: Model and inference settings
   - Architecture specifications
   - Batch size, device selection
   - CAM method parameters

3. **`eval.yaml`**: Evaluation and visualization
   - Metric configurations
   - Differential CAM settings
   - Visualization parameters

### Benefits
- Change experimental setup without code changes
- Track configurations in version control
- Easy to share experimental protocols
- Validate configurations (YAML schema)

## Data Organization

### Assumed Structure
```
data/isic2018/
  images/          # All images
  labels.csv       # Full dataset labels
  train.csv        # Training split
  val.csv          # Validation split
  test.csv         # Test split
```

### CSV Format
Expected columns:
- `image_name`: Image filename (with or without .jpg)
- Class columns: Either one-hot encoded or single `class` column
- Additional metadata optional

### External Resources
```
external/siim/        # Pretrained model
external/finer_cam/   # Optional Finer-CAM implementation
```

## Output Organization

Structured output directory:
```
outputs/
  heatmaps/          # Individual CAM heatmaps
  overlays/          # CAM overlays on images
  differential_cams/  # Differential visualizations
  eval_results/       # Evaluation metrics (JSON)
```

### Naming Convention
- `{image_name}_{class}_heatmap.png`
- `{image_name}_{class}_overlay.png`
- `{image_name}_diff_{class1}_vs_{class2}_*.png`
- `{method}_{split}_evaluation.json`

## Extensibility Points

### Adding New CAM Methods
1. Implement CAM generation function
2. Update `generate_finer_cam.py` with new method
3. Add method-specific configs if needed

### Adding New Metrics
1. Implement metric class in `cam_metrics.py`
2. Follow existing interface (model, compute method)
3. Update `evaluate_cams.py` to include new metric

### Adding New Models
1. Create new wrapper in `src/models/`
2. Follow `SIIMModelWrapper` interface
3. Update configs to support new architecture

### Adding New Visualizations
1. Add utility function to `visualization.py`
2. Update scripts to use new visualization
3. Consider adding to notebook examples

## Best Practices

### For Reproducibility
1. Use configuration files for all experiments
2. Save configs alongside results
3. Use random seeds where applicable
4. Document data splits and preprocessing

### For Efficiency
1. Use DataLoader with multiple workers
2. Batch processing where possible
3. GPU acceleration for inference
4. Pre-generate CAMs for repeated evaluation

### For Code Quality
1. Type hints for function signatures
2. Docstrings for public APIs
3. Modular, testable functions
4. Consistent naming conventions

## Future Enhancements

### Planned Features
1. Full Finer-CAM integration
2. Additional CAM methods (LayerCAM, ScoreCAM)
3. Adversarial evaluation metrics
4. Interactive visualization dashboard
5. Multi-GPU support for large-scale evaluation

### Research Directions
1. Attention-based differential analysis
2. Temporal CAM evolution during training
3. Human-in-the-loop CAM refinement
4. CAM-guided data augmentation

## References

### Key Papers
- Grad-CAM: Selvaraju et al., ICCV 2017
- ISIC Dataset: Codella et al., 2018
- CAM Evaluation: Chattopadhay et al., WACV 2018

### Related Work
- Score-CAM, LayerCAM, XGrad-CAM
- Attention mechanisms in medical imaging
- Explainable AI for healthcare

## Conclusion

This architecture balances:
- **Research flexibility**: Easy to experiment with new ideas
- **Engineering rigor**: Modular, testable, maintainable
- **User accessibility**: Clear APIs, documentation, examples
- **Performance**: Efficient batching, GPU support

The result is a solid foundation for research on interpretable dermoscopic image classification.
