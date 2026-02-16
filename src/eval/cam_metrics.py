"""
CAM Evaluation Metrics

Implements confidence drop and deletion metrics for evaluating CAM quality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Dict
from tqdm import tqdm


class ConfidenceDropMetric:
    """
    Measures the drop in confidence when masking regions highlighted by CAM.
    
    A good CAM should show high confidence drop when important regions are masked.
    """
    
    def __init__(self, model: nn.Module, perturbation_steps: List[float] = None):
        """
        Args:
            model: The model to evaluate
            perturbation_steps: List of perturbation ratios (e.g., [0.1, 0.2, ...])
        """
        self.model = model
        self.model.eval()
        if perturbation_steps is None:
            self.perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            self.perturbation_steps = perturbation_steps
    
    def compute(self, images: torch.Tensor, cams: torch.Tensor, 
                target_classes: torch.Tensor, mask_value: float = 0.0) -> Dict[str, float]:
        """
        Compute confidence drop metric.
        
        Args:
            images: Input images [B, C, H, W]
            cams: CAM heatmaps [B, H, W]
            target_classes: Target class indices [B]
            mask_value: Value to use for masked regions (0 for black)
            
        Returns:
            Dictionary with confidence drop results
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Get original predictions
        with torch.no_grad():
            original_logits = self.model(images)
            original_probs = torch.softmax(original_logits, dim=1)
            original_conf = original_probs[range(batch_size), target_classes]
        
        results = {
            'original_confidence': original_conf.mean().item(),
            'drops': []
        }
        
        # Compute confidence for each perturbation level
        for ratio in self.perturbation_steps:
            perturbed_images = self._perturb_images(images, cams, ratio, mask_value)
            
            with torch.no_grad():
                perturbed_logits = self.model(perturbed_images)
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)
                perturbed_conf = perturbed_probs[range(batch_size), target_classes]
            
            drop = (original_conf - perturbed_conf).mean().item()
            results['drops'].append({
                'ratio': ratio,
                'confidence': perturbed_conf.mean().item(),
                'drop': drop,
                'relative_drop': drop / (original_conf.mean().item() + 1e-8)
            })
        
        # Compute average drop
        results['average_drop'] = np.mean([d['drop'] for d in results['drops']])
        results['average_relative_drop'] = np.mean([d['relative_drop'] for d in results['drops']])
        
        return results
    
    def _perturb_images(self, images: torch.Tensor, cams: torch.Tensor, 
                       ratio: float, mask_value: float) -> torch.Tensor:
        """
        Perturb images by masking top ratio% of CAM-highlighted regions.
        
        Args:
            images: Input images [B, C, H, W]
            cams: CAM heatmaps [B, H, W]
            ratio: Ratio of pixels to mask (0-1)
            mask_value: Value to use for masked regions
            
        Returns:
            Perturbed images [B, C, H, W]
        """
        batch_size = images.shape[0]
        perturbed = images.clone()
        
        # Resize CAMs to match image size if needed
        if cams.shape[-2:] != images.shape[-2:]:
            cams = nn.functional.interpolate(
                cams.unsqueeze(1), 
                size=images.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        for i in range(batch_size):
            cam = cams[i]
            
            # Get threshold for top ratio% pixels
            flat_cam = cam.flatten()
            threshold = torch.quantile(flat_cam, 1 - ratio)
            
            # Create mask
            mask = cam >= threshold
            
            # Apply mask to all channels
            for c in range(images.shape[1]):
                perturbed[i, c][mask] = mask_value
        
        return perturbed


class DeletionMetric:
    """
    Measures the area under the deletion curve.
    
    Progressively deletes pixels in order of CAM importance and tracks 
    the decrease in prediction confidence.
    """
    
    def __init__(self, model: nn.Module, num_steps: int = 100):
        """
        Args:
            model: The model to evaluate
            num_steps: Number of deletion steps
        """
        self.model = model
        self.model.eval()
        self.num_steps = num_steps
    
    def compute(self, images: torch.Tensor, cams: torch.Tensor, 
                target_classes: torch.Tensor, mask_value: float = 0.0) -> Dict[str, float]:
        """
        Compute deletion metric (area under deletion curve).
        
        Args:
            images: Input images [B, C, H, W]
            cams: CAM heatmaps [B, H, W]
            target_classes: Target class indices [B]
            mask_value: Value to use for masked regions
            
        Returns:
            Dictionary with deletion metric results
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Get original predictions
        with torch.no_grad():
            original_logits = self.model(images)
            original_probs = torch.softmax(original_logits, dim=1)
            original_conf = original_probs[range(batch_size), target_classes]
        
        # Resize CAMs to match image size
        if cams.shape[-2:] != images.shape[-2:]:
            cams = nn.functional.interpolate(
                cams.unsqueeze(1), 
                size=images.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Track confidence over deletion steps
        confidence_curve = []
        
        for step in range(self.num_steps + 1):
            ratio = step / self.num_steps
            
            # Delete top ratio% of pixels
            perturbed_images = self._delete_pixels(images, cams, ratio, mask_value)
            
            with torch.no_grad():
                perturbed_logits = self.model(perturbed_images)
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)
                perturbed_conf = perturbed_probs[range(batch_size), target_classes]
            
            confidence_curve.append(perturbed_conf.mean().item())
        
        # Compute area under curve (normalized)
        auc = np.trapz(confidence_curve, dx=1.0/self.num_steps)
        
        results = {
            'original_confidence': original_conf.mean().item(),
            'auc': auc,
            'confidence_curve': confidence_curve,
            'deletion_steps': list(range(self.num_steps + 1))
        }
        
        return results
    
    def _delete_pixels(self, images: torch.Tensor, cams: torch.Tensor,
                      ratio: float, mask_value: float) -> torch.Tensor:
        """
        Delete top ratio% of pixels based on CAM importance.
        """
        batch_size = images.shape[0]
        perturbed = images.clone()
        
        for i in range(batch_size):
            cam = cams[i]
            
            # Get threshold for top ratio% pixels
            flat_cam = cam.flatten()
            if ratio > 0:
                threshold = torch.quantile(flat_cam, 1 - ratio)
                mask = cam >= threshold
            else:
                mask = torch.zeros_like(cam, dtype=torch.bool)
            
            # Apply mask to all channels
            for c in range(images.shape[1]):
                perturbed[i, c][mask] = mask_value
        
        return perturbed


class InsertionMetric:
    """
    Measures the area under the insertion curve.
    
    Progressively inserts pixels in order of CAM importance (starting from 
    a baseline image) and tracks the increase in prediction confidence.
    """
    
    def __init__(self, model: nn.Module, num_steps: int = 100):
        """
        Args:
            model: The model to evaluate
            num_steps: Number of insertion steps
        """
        self.model = model
        self.model.eval()
        self.num_steps = num_steps
    
    def compute(self, images: torch.Tensor, cams: torch.Tensor,
                target_classes: torch.Tensor) -> Dict[str, float]:
        """
        Compute insertion metric (area under insertion curve).
        
        Args:
            images: Input images [B, C, H, W]
            cams: CAM heatmaps [B, H, W]
            target_classes: Target class indices [B]
            
        Returns:
            Dictionary with insertion metric results
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Get original predictions
        with torch.no_grad():
            original_logits = self.model(images)
            original_probs = torch.softmax(original_logits, dim=1)
            original_conf = original_probs[range(batch_size), target_classes]
        
        # Resize CAMs to match image size
        if cams.shape[-2:] != images.shape[-2:]:
            cams = nn.functional.interpolate(
                cams.unsqueeze(1),
                size=images.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Track confidence over insertion steps
        confidence_curve = []
        
        # Start with baseline (black image)
        baseline = torch.zeros_like(images)
        
        for step in range(self.num_steps + 1):
            ratio = step / self.num_steps
            
            # Insert top ratio% of pixels
            inserted_images = self._insert_pixels(images, baseline, cams, ratio)
            
            with torch.no_grad():
                inserted_logits = self.model(inserted_images)
                inserted_probs = torch.softmax(inserted_logits, dim=1)
                inserted_conf = inserted_probs[range(batch_size), target_classes]
            
            confidence_curve.append(inserted_conf.mean().item())
        
        # Compute area under curve (normalized)
        auc = np.trapz(confidence_curve, dx=1.0/self.num_steps)
        
        results = {
            'original_confidence': original_conf.mean().item(),
            'auc': auc,
            'confidence_curve': confidence_curve,
            'insertion_steps': list(range(self.num_steps + 1))
        }
        
        return results
    
    def _insert_pixels(self, images: torch.Tensor, baseline: torch.Tensor,
                      cams: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Insert top ratio% of pixels from original image into baseline.
        """
        batch_size = images.shape[0]
        inserted = baseline.clone()
        
        for i in range(batch_size):
            cam = cams[i]
            
            # Get threshold for top ratio% pixels
            flat_cam = cam.flatten()
            if ratio > 0:
                threshold = torch.quantile(flat_cam, 1 - ratio)
                mask = cam >= threshold
            else:
                mask = torch.zeros_like(cam, dtype=torch.bool)
            
            # Insert pixels for all channels
            for c in range(images.shape[1]):
                inserted[i, c][mask] = images[i, c][mask]
        
        return inserted
