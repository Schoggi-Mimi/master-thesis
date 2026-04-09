"""
CAM Evaluation Metrics

Implements confidence drop and deletion metrics for evaluating CAM quality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Dict, Optional
from tqdm import tqdm


def _safe_divide(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    """Safely divide two floats."""
    return float(numerator) / float(denominator + eps)


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
        batch_size = images.shape[0]

        # Get original predictions
        with torch.no_grad():
            original_logits = self.model(images)
            original_probs = torch.softmax(original_logits, dim=1)
            original_conf = original_probs[range(batch_size), target_classes]

        original_conf_mean = float(original_conf.mean().item())
        per_sample_original = original_conf.detach().cpu().numpy().astype(np.float64)

        results = {
            'original_confidence': original_conf_mean,
            'drops': [],
            'per_sample_original_confidence': per_sample_original.tolist(),
        }

        # Compute confidence for each perturbation level
        for ratio in self.perturbation_steps:
            perturbed_images = self._perturb_images(images, cams, ratio, mask_value)

            with torch.no_grad():
                perturbed_logits = self.model(perturbed_images)
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)
                perturbed_conf = perturbed_probs[range(batch_size), target_classes]

            per_sample_perturbed = perturbed_conf.detach().cpu().numpy().astype(np.float64)
            per_sample_drop = per_sample_original - per_sample_perturbed
            per_sample_relative_drop = [
                _safe_divide(drop_i, orig_i)
                for drop_i, orig_i in zip(per_sample_drop.tolist(), per_sample_original.tolist())
            ]

            mean_conf = float(np.mean(per_sample_perturbed))
            mean_drop = float(np.mean(per_sample_drop))
            mean_relative_drop = float(np.mean(per_sample_relative_drop))

            results['drops'].append({
                'ratio': ratio,
                'confidence': mean_conf,
                'drop': mean_drop,
                'relative_drop': mean_relative_drop,
                'per_sample_confidence': per_sample_perturbed.tolist(),
                'per_sample_drop': per_sample_drop.tolist(),
                'per_sample_relative_drop': per_sample_relative_drop,
            })

        # Compute average drop across all perturbation ratios
        results['average_drop'] = float(np.mean([d['drop'] for d in results['drops']]))
        results['average_relative_drop'] = float(np.mean([d['relative_drop'] for d in results['drops']]))

        # Common alias used in papers / tables
        results['relative_confidence_drop'] = results['average_relative_drop']

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
        per_sample_curve = []

        for step in range(self.num_steps + 1):
            ratio = step / self.num_steps

            # Delete top ratio% of pixels
            perturbed_images = self._delete_pixels(images, cams, ratio, mask_value)

            with torch.no_grad():
                perturbed_logits = self.model(perturbed_images)
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)
                perturbed_conf = perturbed_probs[range(batch_size), target_classes]

            per_sample_conf = perturbed_conf.detach().cpu().numpy().astype(np.float64)
            per_sample_curve.append(per_sample_conf.tolist())
            confidence_curve.append(float(np.mean(per_sample_conf)))

        # Compute area under curve (normalized)
        auc = float(np.trapz(confidence_curve, dx=1.0 / self.num_steps))

        results = {
            'original_confidence': float(original_conf.mean().item()),
            'auc': auc,
            'confidence_curve': confidence_curve,
            'per_sample_confidence_curve': per_sample_curve,
            'deletion_steps': list(range(self.num_steps + 1)),
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
        per_sample_curve = []

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

            per_sample_conf = inserted_conf.detach().cpu().numpy().astype(np.float64)
            per_sample_curve.append(per_sample_conf.tolist())
            confidence_curve.append(float(np.mean(per_sample_conf)))

        # Compute area under curve (normalized)
        auc = float(np.trapz(confidence_curve, dx=1.0 / self.num_steps))

        results = {
            'original_confidence': float(original_conf.mean().item()),
            'auc': auc,
            'confidence_curve': confidence_curve,
            'per_sample_confidence_curve': per_sample_curve,
            'insertion_steps': list(range(self.num_steps + 1)),
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


def summarize_metric_dict(metric_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Keep only compact scalar fields that are easy to serialize into CSV/JSON tables.
    """
    summary = {}
    for key in [
        'original_confidence',
        'average_drop',
        'average_relative_drop',
        'relative_confidence_drop',
        'auc',
    ]:
        if key in metric_dict:
            summary[key] = float(metric_dict[key])
    return summary


@torch.no_grad()
def evaluate_multiple_cams(
    model: nn.Module,
    images: torch.Tensor,
    cams_by_method: Dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    confidence_metric: Optional[ConfidenceDropMetric] = None,
    deletion_metric: Optional[DeletionMetric] = None,
    insertion_metric: Optional[InsertionMetric] = None,
    mask_value: float = 0.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate several CAM tensors on the same image batch.

    Returns a nested dictionary of the form:
    {
        method_name: {
            'confidence_drop': {...},
            'deletion': {...},
            'insertion': {...},
        }
    }
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method_name, cams in cams_by_method.items():
        method_results: Dict[str, Dict[str, float]] = {}

        if confidence_metric is not None:
            method_results['confidence_drop'] = confidence_metric.compute(
                images=images,
                cams=cams,
                target_classes=target_classes,
                mask_value=mask_value,
            )

        if deletion_metric is not None:
            method_results['deletion'] = deletion_metric.compute(
                images=images,
                cams=cams,
                target_classes=target_classes,
                mask_value=mask_value,
            )

        if insertion_metric is not None:
            method_results['insertion'] = insertion_metric.compute(
                images=images,
                cams=cams,
                target_classes=target_classes,
            )

        results[method_name] = method_results

    return results
