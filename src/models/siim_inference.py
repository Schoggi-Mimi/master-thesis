"""
SIIM-ISIC Model Inference Wrapper

Wrapper for loading and running inference with SIIM-ISIC pretrained models.
"""

import os
import torch
import torch.nn as nn
import yaml
from typing import Dict, List, Tuple, Optional
import numpy as np


class SIIMModelWrapper:
    """
    Wrapper for SIIM-ISIC pretrained model inference.
    
    Args:
        model_path (str): Path to the pretrained model checkpoint
        config_path (str, optional): Path to model configuration file
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'cuda'):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Set up for feature extraction
        self.features = None
        self.gradients = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_model(self) -> nn.Module:
        """
        Load the pretrained model.
        Assumes the model is saved as a state dict or full model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            # Try loading as state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                # Create model architecture (placeholder - adjust based on actual model)
                model = self._create_model_architecture()
                model.load_state_dict(state_dict, strict=False)
            else:
                # Assume it's a full model
                model = checkpoint
                
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def _create_model_architecture(self) -> nn.Module:
        """
        Create the model architecture.
        This is a placeholder - should be replaced with actual architecture.
        """
        # Import model architectures (adjust based on actual model)
        try:
            import timm
            architecture = self.config.get('architecture', 'efficientnet_b0')
            num_classes = self.config.get('num_classes', 7)
            model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
            return model
        except ImportError:
            # Fallback to torchvision models
            import torchvision.models as models
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 7)
            return model
    
    def register_hooks(self, target_layer: nn.Module):
        """
        Register forward and backward hooks for gradient-based CAM methods.
        
        Args:
            target_layer: The layer to extract features and gradients from
        """
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def predict(self, images: torch.Tensor, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
            return_logits: Whether to return logits instead of probabilities
            
        Returns:
            predictions: Class predictions or probabilities
            logits: Raw model outputs
        """
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.model(images)
            
            if return_logits:
                return torch.argmax(logits, dim=1), logits
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                return preds, probs
    
    def forward_with_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also extracts intermediate features.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            logits: Model outputs
            features: Extracted features from target layer
        """
        images = images.to(self.device)
        logits = self.model(images)
        return logits, self.features
    
    def get_target_layer(self, layer_name: Optional[str] = None) -> nn.Module:
        """
        Get the target layer for CAM generation.
        
        Args:
            layer_name: Name of the target layer (if None, uses last conv layer)
            
        Returns:
            Target layer module
        """
        if layer_name:
            # Try to get named layer
            for name, module in self.model.named_modules():
                if name == layer_name:
                    return module
        
        # Default: find last convolutional layer
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            raise ValueError("Could not find suitable target layer")
            
        return target_layer
    
    def get_class_names(self) -> List[str]:
        """Get the class names for ISIC 2018."""
        return ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable."""
        _, probs = self.predict(images)
        return probs
