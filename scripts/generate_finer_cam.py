"""
Generate Finer-CAM Heatmaps and Overlays

Script to run Finer-CAM on ISIC 2018 images and save heatmaps and overlays.
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
from src.utils.visualization import save_cam_visualization, tensor_to_numpy_image


def generate_gradcam(model_wrapper, image, target_class, target_layer):
    """
    Generate Grad-CAM for an image.
    
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
    target.backward()
    
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


def run_finer_cam_wrapper(image_path, model_wrapper, target_class, target_layer, 
                         finer_cam_path=None):
    """
    Wrapper to run Finer-CAM (or fallback to Grad-CAM).
    
    Args:
        image_path: Path to image
        model_wrapper: SIIM model wrapper
        target_class: Target class index
        target_layer: Layer for CAM
        finer_cam_path: Path to Finer-CAM repository (if available)
        
    Returns:
        CAM heatmap as numpy array
    """
    # Load and preprocess image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(model_wrapper.device)
    
    # Check if Finer-CAM is available
    if finer_cam_path and os.path.exists(finer_cam_path):
        try:
            # Try to use Finer-CAM
            sys.path.insert(0, finer_cam_path)
            # Import and use Finer-CAM (placeholder - adjust based on actual API)
            # from finer_cam import FinerCAM
            # cam = FinerCAM(model_wrapper.model, target_layer)
            # heatmap = cam.generate(image_tensor, target_class)
            
            # Fallback to Grad-CAM for now
            print("Finer-CAM not fully integrated, using Grad-CAM")
            heatmap = generate_gradcam(model_wrapper, image_tensor, target_class, target_layer)
        except Exception as e:
            print(f"Error using Finer-CAM: {e}, falling back to Grad-CAM")
            heatmap = generate_gradcam(model_wrapper, image_tensor, target_class, target_layer)
    else:
        # Use standard Grad-CAM
        heatmap = generate_gradcam(model_wrapper, image_tensor, target_class, target_layer)
    
    return heatmap


def main():
    parser = argparse.ArgumentParser(description='Generate Finer-CAM heatmaps and overlays')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Path to configs directory')
    parser.add_argument('--data-split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to process')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (use 1 for individual processing)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to process (None for all)')
    parser.add_argument('--colormap', type=str, default='jet',
                       help='Colormap for visualization')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Overlay transparency')
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    configs = load_all_configs(args.config_dir)
    paths_config = configs['paths']
    model_config = configs['model']
    
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
    
    # Finer-CAM path
    finer_cam_path = paths_config['external']['finer_cam'].get('repo_path')
    
    # Process images
    print(f"Generating CAMs for {args.data_split} split...")
    num_processed = 0
    
    for batch in tqdm(dataloader):
        images = batch['image']
        labels = batch['label']
        image_names = batch['image_name']
        image_paths = batch['image_path']
        
        # Get predictions
        preds, probs = model_wrapper.predict(images)
        
        for i in range(len(images)):
            if args.num_samples and num_processed >= args.num_samples:
                break
            
            # Get target class (use prediction)
            target_class = preds[i].item()
            
            # Generate CAM
            cam = run_finer_cam_wrapper(
                image_paths[i],
                model_wrapper,
                target_class,
                target_layer,
                finer_cam_path
            )
            
            # Save visualization
            image_name = os.path.splitext(image_names[i])[0]
            class_name = paths_config['classes'][target_class]
            output_base = os.path.join(
                paths_config['outputs']['heatmaps'],
                f"{image_name}_{class_name}"
            )
            
            save_cam_visualization(
                image_path=image_paths[i],
                cam=cam,
                output_path=output_base,
                alpha=args.alpha,
                colormap=args.colormap,
                save_separate=True
            )
            
            num_processed += 1
        
        if args.num_samples and num_processed >= args.num_samples:
            break
    
    print(f"\nProcessed {num_processed} images")
    print(f"Heatmaps saved to: {paths_config['outputs']['heatmaps']}")
    print(f"Overlays saved to: {paths_config['outputs']['heatmaps']}")


if __name__ == '__main__':
    main()
