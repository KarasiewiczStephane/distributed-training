"""Configuration loading and merging utilities using OmegaConf."""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a DictConfig.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    logger.info("Loading configuration from %s", config_path)
    return OmegaConf.load(config_path)


def merge_configs(*config_paths: str) -> DictConfig:
    """Load and merge multiple YAML configuration files.

    Later files override earlier ones for overlapping keys.

    Args:
        *config_paths: Paths to YAML configuration files.

    Returns:
        Merged configuration as a DictConfig.
    """
    configs = [load_config(p) for p in config_paths]
    merged = OmegaConf.merge(*configs)
    logger.info("Merged %d configuration files", len(configs))
    return merged


def config_to_dict(config: DictConfig) -> dict:
    """Convert a DictConfig to a plain Python dictionary.

    Args:
        config: OmegaConf DictConfig to convert.

    Returns:
        Plain dictionary representation.
    """
    return OmegaConf.to_container(config, resolve=True)
