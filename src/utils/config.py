"""
Configuration Utilities

Helper functions for loading and managing configurations.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_all_configs(config_dir: str = 'configs') -> Dict[str, Dict]:
    """
    Load all configuration files from the configs directory.
    
    Args:
        config_dir: Path to configs directory
        
    Returns:
        Dictionary with all configs
    """
    configs = {}
    
    config_files = {
        'paths': 'paths.yaml',
        'model': 'model.yaml',
        'eval': 'eval.yaml'
    }
    
    for name, filename in config_files.items():
        config_path = os.path.join(config_dir, filename)
        if os.path.exists(config_path):
            configs[name] = load_config(config_path)
    
    return configs


def ensure_directories(config: Dict[str, Any]):
    """
    Ensure all output directories exist.
    
    Args:
        config: Configuration dictionary with path information
    """
    if 'outputs' in config:
        outputs = config['outputs']
        for key, path in outputs.items():
            os.makedirs(path, exist_ok=True)
