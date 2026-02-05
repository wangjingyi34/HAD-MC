"""Configuration loading utilities for HAD-MC 2.0"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = None, config_name: str = 'default') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Full path to config file (optional)
        config_name: Name of config in hadmc2/configs/ (optional)

    Returns:
        dict: Configuration dictionary
    """
    # Determine config path
    if config_path is None:
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to hadmc2/configs/
        config_dir = os.path.join(os.path.dirname(current_dir), 'configs')
        config_path = os.path.join(config_dir, f'{config_name}.yaml')

    # Load YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {config_path}")
        return config

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple config dictionaries (later configs override earlier ones).

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        dict: Merged configuration
    """
    merged = {}

    for config in configs:
        merged.update(config)

    return merged


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        dict: Default configuration
    """
    return load_config(config_name='default')
