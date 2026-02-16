"""
Visualization Utilities

Helper functions for visualizing CAMs and creating overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import cv2
from typing import Optional, Tuple


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """
    Normalize CAM to [0, 1] range.
    
    Args:
        cam: CAM heatmap
        
    Returns:
        Normalized CAM
    """
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam


def apply_colormap(cam: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """
    Apply colormap to CAM heatmap.
    
    Args:
        cam: Normalized CAM heatmap [H, W] in range [0, 1]
        colormap: Name of matplotlib colormap
        
    Returns:
        Colored CAM [H, W, 3] in range [0, 255]
    """
    cam_uint8 = np.uint8(255 * cam)
    cmap = cm.get_cmap(colormap)
    colored_cam = cmap(cam_uint8)[:, :, :3]  # Remove alpha channel
    colored_cam = np.uint8(255 * colored_cam)
    return colored_cam


def create_overlay(image: np.ndarray, cam: np.ndarray, 
                  alpha: float = 0.5, colormap: str = 'jet') -> np.ndarray:
    """
    Create overlay of CAM on original image.
    
    Args:
        image: Original image [H, W, 3] in range [0, 255]
        cam: CAM heatmap [H, W] (will be normalized)
        alpha: Overlay transparency (0=image only, 1=cam only)
        colormap: Colormap name
        
    Returns:
        Overlay image [H, W, 3] in range [0, 255]
    """
    # Normalize CAM
    cam_normalized = normalize_cam(cam)
    
    # Resize CAM to match image size if needed
    if cam_normalized.shape != image.shape[:2]:
        cam_normalized = cv2.resize(cam_normalized, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    colored_cam = apply_colormap(cam_normalized, colormap)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - alpha, colored_cam, alpha, 0)
    
    return overlay


def save_cam_visualization(image_path: str, cam: np.ndarray, 
                          output_path: str, alpha: float = 0.5,
                          colormap: str = 'jet', save_separate: bool = True):
    """
    Save CAM visualization with overlay.
    
    Args:
        image_path: Path to original image
        cam: CAM heatmap [H, W]
        output_path: Path to save output (without extension)
        alpha: Overlay transparency
        colormap: Colormap name
        save_separate: Whether to save heatmap and overlay separately
    """
    # Load original image
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Create overlay
    overlay = create_overlay(image, cam, alpha, colormap)
    
    # Save overlay
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(f"{output_path}_overlay.png")
    
    if save_separate:
        # Save heatmap
        cam_normalized = normalize_cam(cam)
        if cam_normalized.shape != image.shape[:2]:
            cam_normalized = cv2.resize(cam_normalized, (image.shape[1], image.shape[0]))
        
        colored_cam = apply_colormap(cam_normalized, colormap)
        heatmap_img = Image.fromarray(colored_cam)
        heatmap_img.save(f"{output_path}_heatmap.png")


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy image.
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        Numpy image [H, W, 3] in range [0, 255]
    """
    # Handle batch dimension
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # Move to CPU and convert to numpy
    img = tensor.cpu().numpy()
    
    # Denormalize (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * std + mean
    
    # Clip and convert to uint8
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    img = np.uint8(255 * img)
    
    return img


def plot_cam_comparison(images: list, cams: list, titles: list,
                       output_path: str, colormap: str = 'jet'):
    """
    Plot comparison of multiple CAMs side by side.
    
    Args:
        images: List of images (numpy arrays)
        cams: List of CAM heatmaps
        titles: List of titles for each subplot
        output_path: Path to save the comparison plot
        colormap: Colormap name
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n):
        # Original image
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')
        
        # CAM overlay
        overlay = create_overlay(images[i], cams[i], alpha=0.5, colormap=colormap)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'{titles[i]} CAM')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
