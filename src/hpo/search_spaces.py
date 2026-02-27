"""Hyperparameter search space definitions for Ray Tune."""

import logging

logger = logging.getLogger(__name__)

try:
    from ray import tune

    RAY_AVAILABLE = True
except ImportError:
    tune = None
    RAY_AVAILABLE = False


def get_default_search_space() -> dict:
    """Get the default hyperparameter search space.

    Returns:
        Dictionary mapping hyperparameter names to Ray Tune distributions.

    Raises:
        ImportError: If Ray Tune is not installed.
    """
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray Tune is not installed. Install with: pip install ray[tune]"
        )
    return {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.0, 0.5),
    }


def get_search_space_from_config(config: dict) -> dict:
    """Build a search space from a configuration dictionary.

    Args:
        config: Dictionary with param names and range specs.

    Returns:
        Dictionary of Ray Tune distributions.

    Raises:
        ImportError: If Ray Tune is not installed.
    """
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray Tune is not installed. Install with: pip install ray[tune]"
        )
    space = {}
    for name, spec in config.items():
        dist_type = spec.get("type", "uniform")
        low = spec.get("low", 0.0)
        high = spec.get("high", 1.0)
        if dist_type == "loguniform":
            space[name] = tune.loguniform(low, high)
        elif dist_type == "choice":
            space[name] = tune.choice(spec.get("values", []))
        else:
            space[name] = tune.uniform(low, high)
    return space
