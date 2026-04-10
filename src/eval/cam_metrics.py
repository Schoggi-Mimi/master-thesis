"""
CAM Evaluation Metrics

Implements relative confidence drop, deletion, and insertion metrics for
CAM quality evaluation.

"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Dict, Optional
from tqdm import tqdm


class RelativeConfidenceDropMetric:
    """
    Measures the FinerCAM-style relative confidence drop:
        RD = (p_c - p*_c) - (p_d - p*_d)
    where c is the target class and d is the reference/similar class.

    If multiple reference classes are provided, their probabilities are averaged.
    """

    def __init__(self, model: nn.Module, perturbation_steps: List[float] = None):
        self.model = model
        self.model.eval()
        if perturbation_steps is None:
            self.perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            self.perturbation_steps = perturbation_steps

    def compute(
        self,
        images: torch.Tensor,
        cams: torch.Tensor,
        target_classes: torch.Tensor,
        reference_classes: torch.Tensor,
        mask_value: float = 0.0,
    ):
        batch_size = images.shape[0]

        if target_classes.shape[0] != batch_size:
            raise ValueError("target_classes must have the same batch size as images")

        if reference_classes.ndim == 1:
            reference_classes = reference_classes.unsqueeze(0).repeat(batch_size, 1)

        if reference_classes.shape[0] != batch_size:
            raise ValueError("reference_classes must have batch dimension equal to images")

        with torch.no_grad():
            original_logits = self.model(images)
            original_probs = torch.softmax(original_logits, dim=1)

            batch_idx = torch.arange(batch_size, device=images.device)
            original_target_conf = original_probs[batch_idx, target_classes]

            ref_probs = torch.gather(original_probs, 1, reference_classes)
            original_reference_conf = ref_probs.mean(dim=1)

        per_sample_original_target = original_target_conf.detach().cpu().numpy().astype(np.float64)
        per_sample_original_reference = original_reference_conf.detach().cpu().numpy().astype(np.float64)

        results = {
            "original_target_confidence": float(original_target_conf.mean().item()),
            "original_reference_confidence": float(original_reference_conf.mean().item()),
            "steps": [],
            "per_sample_original_target_confidence": per_sample_original_target.tolist(),
            "per_sample_original_reference_confidence": per_sample_original_reference.tolist(),
        }

        for ratio in self.perturbation_steps:
            perturbed_images = self._perturb_images(images, cams, ratio, mask_value)

            with torch.no_grad():
                perturbed_logits = self.model(perturbed_images)
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)

                perturbed_target_conf = perturbed_probs[batch_idx, target_classes]
                perturbed_ref_probs = torch.gather(perturbed_probs, 1, reference_classes)
                perturbed_reference_conf = perturbed_ref_probs.mean(dim=1)

            per_sample_target_masked = perturbed_target_conf.detach().cpu().numpy().astype(np.float64)
            per_sample_reference_masked = perturbed_reference_conf.detach().cpu().numpy().astype(np.float64)

            per_sample_target_drop = per_sample_original_target - per_sample_target_masked
            per_sample_reference_drop = per_sample_original_reference - per_sample_reference_masked
            per_sample_rd = per_sample_target_drop - per_sample_reference_drop

            results["steps"].append(
                {
                    "ratio": float(ratio),
                    "target_confidence": float(np.mean(per_sample_original_target)),
                    "reference_confidence": float(np.mean(per_sample_original_reference)),
                    "target_confidence_masked": float(np.mean(per_sample_target_masked)),
                    "reference_confidence_masked": float(np.mean(per_sample_reference_masked)),
                    "target_drop": float(np.mean(per_sample_target_drop)),
                    "reference_drop": float(np.mean(per_sample_reference_drop)),
                    "relative_confidence_drop": float(np.mean(per_sample_rd)),
                    "per_sample_target_confidence_masked": per_sample_target_masked.tolist(),
                    "per_sample_reference_confidence_masked": per_sample_reference_masked.tolist(),
                    "per_sample_target_drop": per_sample_target_drop.tolist(),
                    "per_sample_reference_drop": per_sample_reference_drop.tolist(),
                    "per_sample_relative_confidence_drop": per_sample_rd.tolist(),
                }
            )

        results["average_target_drop"] = float(np.mean([d["target_drop"] for d in results["steps"]]))
        results["average_reference_drop"] = float(np.mean([d["reference_drop"] for d in results["steps"]]))
        results["average_relative_confidence_drop"] = float(
            np.mean([d["relative_confidence_drop"] for d in results["steps"]])
        )
        results["relative_confidence_drop"] = results["average_relative_confidence_drop"]
        results["max_relative_confidence_drop"] = float(
            np.max([d["relative_confidence_drop"] for d in results["steps"]])
        )

        return results

    def _perturb_images(
        self,
        images: torch.Tensor,
        cams: torch.Tensor,
        ratio: float,
        mask_value: float,
    ) -> torch.Tensor:
        batch_size = images.shape[0]
        perturbed = images.clone()

        if cams.shape[-2:] != images.shape[-2:]:
            cams = nn.functional.interpolate(
                cams.unsqueeze(1),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        for i in range(batch_size):
            cam = cams[i]
            flat_cam = cam.flatten()
            threshold = torch.quantile(flat_cam, 1 - ratio)
            mask = cam >= threshold
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
        "original_confidence",
        "original_target_confidence",
        "original_reference_confidence",
        "average_drop",
        "average_relative_drop",
        "average_target_drop",
        "average_reference_drop",
        "average_relative_confidence_drop",
        "relative_confidence_drop",
        "max_relative_confidence_drop",
        "auc",
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
    reference_classes: Optional[torch.Tensor] = None,
    relative_confidence_metric: Optional[RelativeConfidenceDropMetric] = None,
    deletion_metric: Optional[DeletionMetric] = None,
    insertion_metric: Optional[InsertionMetric] = None,
    mask_value: float = 0.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate several CAM tensors on the same image batch.

    Returns a nested dictionary of the form:
    {
        method_name: {
            'relative_confidence_drop': {...},
            'deletion': {...},
            'insertion': {...},
        }
    }
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method_name, cams in cams_by_method.items():
        method_results: Dict[str, Dict[str, float]] = {}

        if relative_confidence_metric is not None:
            if reference_classes is None:
                raise ValueError("reference_classes are required for relative confidence drop evaluation")
            method_results["relative_confidence_drop"] = relative_confidence_metric.compute(
                images=images,
                cams=cams,
                target_classes=target_classes,
                reference_classes=reference_classes,
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
