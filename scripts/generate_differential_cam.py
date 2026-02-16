"""
Generate Differential CAMs

Script to generate differential (top-2 / class-vs-class) CAMs that highlight
discriminative features between classes.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.isic_dataset import create_dataloader
from src.models.siim_inference import SIIMModelWrapper
from src.utils.config import load_all_configs, ensure_directories
from src.utils.visualization import save_cam_visualization, plot_cam_comparison, tensor_to_numpy_image


def generate_gradcam_for_class(model_wrapper, image, target_class, target_layer):
    """
    Generate Grad-CAM for a specific class.
    
    Args:
        model_wrapper: SIIM model wrapper
        image: Input image tensor [1, C, H, W]
        target_class: Target class index
        target_layer: Layer to compute CAM from
        
    Returns:
        CAM heatmap as numpy array
    """
    # Register hooks
    model_wrapper.register_hooks(target_layer)
    
    # Forward pass
    image.requires_grad = True
    logits = model_wrapper.model(image)
    
    # Backward pass for target class
    model_wrapper.model.zero_grad()
    target = logits[0, target_class]
    target.backward(retain_graph=True)
    
    # Compute CAM
    gradients = model_wrapper.gradients[0].cpu().numpy()  # [C, H, W]
    features = model_wrapper.features[0].cpu().numpy()    # [C, H, W]
    
    # Global average pooling of gradients
    weights = np.mean(gradients, axis=(1, 2))  # [C]
    
    # Weighted combination of feature maps
    cam = np.sum(weights[:, None, None] * features, axis=0)  # [H, W]
    
    # ReLU
    cam = np.maximum(cam, 0)
    
    return cam


def generate_differential_cam_top2(model_wrapper, image, target_layer):
    """
    Generate differential CAM for top-2 predicted classes.
    
    Args:
        model_wrapper: SIIM model wrapper
        image: Input image tensor [1, C, H, W]
        target_layer: Layer to compute CAM from
        
    Returns:
        dict with CAMs and class information
    """
    # Get predictions
    with torch.no_grad():
        logits = model_wrapper.model(image)
        probs = torch.softmax(logits, dim=1)[0]
    
    # Get top-2 classes
    top2_probs, top2_classes = torch.topk(probs, 2)
    class1, class2 = top2_classes[0].item(), top2_classes[1].item()
    prob1, prob2 = top2_probs[0].item(), top2_probs[1].item()
    
    # Generate CAMs for both classes
    cam1 = generate_gradcam_for_class(model_wrapper, image, class1, target_layer)
    cam2 = generate_gradcam_for_class(model_wrapper, image, class2, target_layer)
    
    # Normalize CAMs
    cam1_norm = (cam1 - cam1.min()) / (cam1.max() - cam1.min() + 1e-8)
    cam2_norm = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)
    
    # Compute differential CAM
    differential_cam = cam1_norm - cam2_norm
    
    return {
        'class1': class1,
        'class2': class2,
        'prob1': prob1,
        'prob2': prob2,
        'cam1': cam1,
        'cam2': cam2,
        'differential_cam': differential_cam
    }


def generate_differential_cam_pair(model_wrapper, image, class1, class2, target_layer):
    """
    Generate differential CAM for a specific pair of classes.
    
    Args:
        model_wrapper: SIIM model wrapper
        image: Input image tensor [1, C, H, W]
        class1: First class index
        class2: Second class index
        target_layer: Layer to compute CAM from
        
    Returns:
        dict with CAMs and class information
    """
    # Generate CAMs for both classes
    cam1 = generate_gradcam_for_class(model_wrapper, image, class1, target_layer)
    cam2 = generate_gradcam_for_class(model_wrapper, image, class2, target_layer)
    
    # Normalize CAMs
    cam1_norm = (cam1 - cam1.min()) / (cam1.max() - cam1.min() + 1e-8)
    cam2_norm = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)
    
    # Compute differential CAM
    differential_cam = cam1_norm - cam2_norm
    
    return {
        'class1': class1,
        'class2': class2,
        'cam1': cam1,
        'cam2': cam2,
        'differential_cam': differential_cam
    }


def main():
    parser = argparse.ArgumentParser(description='Generate differential CAMs')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Path to configs directory')
    parser.add_argument('--data-split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to process')
    parser.add_argument('--mode', type=str, default='top2',
                       choices=['top2', 'class_vs_class'],
                       help='Differential CAM mode')
    parser.add_argument('--class-pairs', type=str, nargs='+', default=None,
                       help='Class pairs for class_vs_class mode (e.g., MEL-NV BCC-AKIEC)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to process (None for all)')
    parser.add_argument('--colormap', type=str, default='RdBu_r',
                       help='Colormap for differential visualization (use diverging colormap)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Overlay transparency')
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    configs = load_all_configs(args.config_dir)
    paths_config = configs['paths']
    model_config = configs['model']
    eval_config = configs['eval']
    
    # Ensure output directories exist
    ensure_directories(paths_config)
    os.makedirs(paths_config['outputs']['differential_cams'], exist_ok=True)
    
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
    
    # Load data
    print(f"Loading {args.data_split} data...")
    csv_file = paths_config['data']['isic2018'][f'{args.data_split}_split']
    img_dir = paths_config['data']['isic2018']['images']
    
    dataloader = create_dataloader(
        csv_file=csv_file,
        img_dir=img_dir,
        batch_size=1,  # Process one at a time for differential CAMs
        shuffle=False,
        num_workers=4,
        input_size=model_config['siim_model']['input_size'],
        augment=False,
        class_names=paths_config['classes']
    )
    
    # Parse class pairs if in class_vs_class mode
    class_pairs = []
    if args.mode == 'class_vs_class':
        if args.class_pairs:
            for pair_str in args.class_pairs:
                class1_name, class2_name = pair_str.split('-')
                class1_idx = paths_config['classes'].index(class1_name)
                class2_idx = paths_config['classes'].index(class2_name)
                class_pairs.append((class1_idx, class2_idx, class1_name, class2_name))
        else:
            # Use default pairs from config
            for pair in eval_config['differential']['class_pairs']:
                class1_name, class2_name = pair
                class1_idx = paths_config['classes'].index(class1_name)
                class2_idx = paths_config['classes'].index(class2_name)
                class_pairs.append((class1_idx, class2_idx, class1_name, class2_name))
    
    # Process images
    print(f"Generating differential CAMs for {args.data_split} split...")
    num_processed = 0
    
    for batch in tqdm(dataloader):
        if args.num_samples and num_processed >= args.num_samples:
            break
        
        images = batch['image']
        image_names = batch['image_name']
        image_paths = batch['image_path']
        
        image_name = os.path.splitext(image_names[0])[0]
        
        if args.mode == 'top2':
            # Generate top-2 differential CAM
            result = generate_differential_cam_top2(model_wrapper, images, target_layer)
            
            class1_name = paths_config['classes'][result['class1']]
            class2_name = paths_config['classes'][result['class2']]
            
            # Save differential CAM
            output_base = os.path.join(
                paths_config['outputs']['differential_cams'],
                f"{image_name}_diff_{class1_name}_vs_{class2_name}"
            )
            
            # Convert differential CAM to absolute values for visualization
            diff_cam_abs = np.abs(result['differential_cam'])
            save_cam_visualization(
                image_path=image_paths[0],
                cam=diff_cam_abs,
                output_path=output_base,
                alpha=args.alpha,
                colormap=args.colormap,
                save_separate=True
            )
            
            # Also save individual CAMs
            save_cam_visualization(
                image_path=image_paths[0],
                cam=result['cam1'],
                output_path=f"{output_base}_{class1_name}",
                alpha=args.alpha,
                colormap='jet',
                save_separate=False
            )
            
            save_cam_visualization(
                image_path=image_paths[0],
                cam=result['cam2'],
                output_path=f"{output_base}_{class2_name}",
                alpha=args.alpha,
                colormap='jet',
                save_separate=False
            )
            
            num_processed += 1
            
        else:  # class_vs_class mode
            for class1_idx, class2_idx, class1_name, class2_name in class_pairs:
                result = generate_differential_cam_pair(
                    model_wrapper, images, class1_idx, class2_idx, target_layer
                )
                
                # Save differential CAM
                output_base = os.path.join(
                    paths_config['outputs']['differential_cams'],
                    f"{image_name}_diff_{class1_name}_vs_{class2_name}"
                )
                
                diff_cam_abs = np.abs(result['differential_cam'])
                save_cam_visualization(
                    image_path=image_paths[0],
                    cam=diff_cam_abs,
                    output_path=output_base,
                    alpha=args.alpha,
                    colormap=args.colormap,
                    save_separate=True
                )
            
            num_processed += 1
    
    print(f"\nProcessed {num_processed} images")
    print(f"Differential CAMs saved to: {paths_config['outputs']['differential_cams']}")


if __name__ == '__main__':
    main()
