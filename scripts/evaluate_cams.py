"""
Evaluate CAMs

Script to evaluate CAM quality using confidence drop and deletion metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.isic_dataset import create_dataloader
from src.models.siim_inference import SIIMModelWrapper
from src.eval.cam_metrics import ConfidenceDropMetric, DeletionMetric, InsertionMetric
from src.utils.config import load_all_configs, ensure_directories


def load_cam_heatmap(heatmap_path):
    """
    Load a saved CAM heatmap.
    
    Args:
        heatmap_path: Path to heatmap image
        
    Returns:
        CAM as numpy array
    """
    heatmap_img = Image.open(heatmap_path).convert('L')  # Grayscale
    heatmap = np.array(heatmap_img).astype(np.float32) / 255.0
    return heatmap


def generate_gradcam_batch(model_wrapper, images, target_classes, target_layer):
    """
    Generate Grad-CAM for a batch of images.
    
    Args:
        model_wrapper: SIIM model wrapper
        images: Batch of images [B, C, H, W]
        target_classes: Target class indices [B]
        target_layer: Layer to compute CAM from
        
    Returns:
        CAMs as tensor [B, H, W]
    """
    batch_size = images.shape[0]
    cams = []
    
    # Register hooks
    model_wrapper.register_hooks(target_layer)
    
    for i in range(batch_size):
        image = images[i:i+1]
        target_class = target_classes[i].item()
        
        # Forward pass
        image.requires_grad = True
        logits = model_wrapper.model(image)
        
        # Backward pass
        model_wrapper.model.zero_grad()
        target = logits[0, target_class]
        target.backward()
        
        # Compute CAM
        gradients = model_wrapper.gradients[0].cpu().numpy()
        features = model_wrapper.features[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * features, axis=0)
        cam = np.maximum(cam, 0)
        
        cams.append(cam)
    
    cams = np.stack(cams)
    return torch.tensor(cams, device=images.device)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CAM quality')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Path to configs directory')
    parser.add_argument('--data-split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate')
    parser.add_argument('--cam-method', type=str, default='gradcam',
                       choices=['gradcam', 'finer_cam'],
                       help='CAM method to evaluate')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['confidence_drop', 'deletion'],
                       choices=['confidence_drop', 'deletion', 'insertion'],
                       help='Metrics to compute')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--use-saved-cams', action='store_true',
                       help='Use pre-saved CAM heatmaps instead of generating')
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    configs = load_all_configs(args.config_dir)
    paths_config = configs['paths']
    model_config = configs['model']
    eval_config = configs['eval']
    
    # Ensure output directories exist
    ensure_directories(paths_config)
    
    # Load model
    print("Loading SIIM model...")
    model_path = paths_config['external']['siim']['model_path']
    model_wrapper = SIIMModelWrapper(
        model_path=model_path,
        device=model_config['inference']['device']
    )
    
    # Get target layer
    target_layer_name = model_config['cam'].get('target_layer', 'layer4')
    target_layer = model_wrapper.get_target_layer(target_layer_name)
    
    # Initialize metrics
    metrics = {}
    if 'confidence_drop' in args.metrics:
        perturbation_steps = eval_config['confidence_drop']['perturbation_steps']
        metrics['confidence_drop'] = ConfidenceDropMetric(
            model_wrapper.model,
            perturbation_steps=perturbation_steps
        )
    if 'deletion' in args.metrics:
        num_steps = eval_config['deletion']['num_steps']
        metrics['deletion'] = DeletionMetric(
            model_wrapper.model,
            num_steps=num_steps
        )
    if 'insertion' in args.metrics:
        num_steps = eval_config['deletion']['num_steps']  # Use same steps
        metrics['insertion'] = InsertionMetric(
            model_wrapper.model,
            num_steps=num_steps
        )
    
    # Load data
    print(f"Loading {args.data_split} data...")
    csv_file = paths_config['data']['isic2018'][f'{args.data_split}_split']
    img_dir = paths_config['data']['isic2018']['images']
    
    dataloader = create_dataloader(
        csv_file=csv_file,
        img_dir=img_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        input_size=model_config['siim_model']['input_size'],
        augment=False,
        class_names=paths_config['classes']
    )
    
    # Evaluate
    print(f"Evaluating {args.cam_method} on {args.data_split} split...")
    all_results = {metric_name: [] for metric_name in args.metrics}
    num_processed = 0
    
    for batch in tqdm(dataloader):
        if args.num_samples and num_processed >= args.num_samples:
            break
        
        images = batch['image'].to(model_wrapper.device)
        image_names = batch['image_name']
        
        # Get predictions
        preds, probs = model_wrapper.predict(images)
        
        # Generate or load CAMs
        if args.use_saved_cams:
            # Load pre-saved CAMs
            cams = []
            for img_name in image_names:
                img_name_base = os.path.splitext(img_name)[0]
                cam_path = os.path.join(
                    paths_config['outputs']['heatmaps'],
                    f"{img_name_base}_*_heatmap.png"
                )
                # Find the heatmap file (simplified - assumes one file per image)
                import glob
                cam_files = glob.glob(cam_path)
                if cam_files:
                    cam = load_cam_heatmap(cam_files[0])
                    cams.append(cam)
                else:
                    # Generate on the fly if not found
                    print(f"Warning: CAM not found for {img_name}, generating...")
                    cam = generate_gradcam_batch(
                        model_wrapper, 
                        images[len(cams):len(cams)+1],
                        preds[len(cams):len(cams)+1],
                        target_layer
                    )[0].cpu().numpy()
                    cams.append(cam)
            cams = torch.tensor(np.stack(cams), device=images.device)
        else:
            # Generate CAMs on the fly
            cams = generate_gradcam_batch(model_wrapper, images, preds, target_layer)
        
        # Compute metrics
        for metric_name, metric in metrics.items():
            result = metric.compute(images, cams, preds)
            all_results[metric_name].append(result)
        
        num_processed += len(images)
        
        if args.num_samples and num_processed >= args.num_samples:
            break
    
    # Aggregate results
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS ({args.cam_method} on {args.data_split} split)")
    print("="*80)
    
    aggregated_results = {}
    
    for metric_name in args.metrics:
        results_list = all_results[metric_name]
        
        if metric_name == 'confidence_drop':
            # Aggregate confidence drop results
            avg_original_conf = np.mean([r['original_confidence'] for r in results_list])
            avg_drop = np.mean([r['average_drop'] for r in results_list])
            avg_relative_drop = np.mean([r['average_relative_drop'] for r in results_list])
            
            print(f"\nConfidence Drop Metric:")
            print(f"  Original Confidence: {avg_original_conf:.4f}")
            print(f"  Average Drop: {avg_drop:.4f}")
            print(f"  Average Relative Drop: {avg_relative_drop:.4f}")
            
            aggregated_results['confidence_drop'] = {
                'original_confidence': float(avg_original_conf),
                'average_drop': float(avg_drop),
                'average_relative_drop': float(avg_relative_drop)
            }
        
        elif metric_name == 'deletion':
            # Aggregate deletion results
            avg_auc = np.mean([r['auc'] for r in results_list])
            
            print(f"\nDeletion Metric:")
            print(f"  Area Under Curve: {avg_auc:.4f}")
            
            aggregated_results['deletion'] = {
                'auc': float(avg_auc)
            }
        
        elif metric_name == 'insertion':
            # Aggregate insertion results
            avg_auc = np.mean([r['auc'] for r in results_list])
            
            print(f"\nInsertion Metric:")
            print(f"  Area Under Curve: {avg_auc:.4f}")
            
            aggregated_results['insertion'] = {
                'auc': float(avg_auc)
            }
    
    # Save results
    output_dir = paths_config['outputs']['eval_results']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{args.cam_method}_{args.data_split}_evaluation.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
