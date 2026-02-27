"""Ray Tune hyperparameter optimization with ASHA and Bayesian search."""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    RAY_AVAILABLE = True
except ImportError:
    tune = None
    ASHAScheduler = None
    RAY_AVAILABLE = False

try:
    from ray.tune.search.optuna import OptunaSearch

    OPTUNA_AVAILABLE = True
except ImportError:
    OptunaSearch = None
    OPTUNA_AVAILABLE = False


class HPORunner:
    """Hyperparameter optimization runner using Ray Tune.

    Supports ASHA scheduling for early stopping of poor trials,
    random search, and Bayesian optimization via Optuna.

    Args:
        train_fn: Training function that Ray Tune will call per trial.
        config: Base training configuration.
        num_gpus_per_trial: GPUs to allocate per trial.
    """

    def __init__(
        self,
        train_fn: Callable,
        config: dict[str, Any],
        num_gpus_per_trial: float = 0,
    ) -> None:
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is not installed. Install with: pip install ray[tune]"
            )
        self.train_fn = train_fn
        self.config = config
        self.num_gpus_per_trial = num_gpus_per_trial

    def create_asha_scheduler(
        self,
        max_epochs: int = 100,
        grace_period: int = 10,
        metric: str = "val_accuracy",
        mode: str = "max",
    ) -> Any:
        """Create an ASHA scheduler for early stopping.

        Args:
            max_epochs: Maximum training epochs.
            grace_period: Minimum epochs before early stopping.
            metric: Metric to optimize.
            mode: Optimization direction ('max' or 'min').

        Returns:
            ASHAScheduler instance.
        """
        return ASHAScheduler(
            max_t=max_epochs,
            grace_period=grace_period,
            reduction_factor=2,
            metric=metric,
            mode=mode,
        )

    def run_random_search(
        self,
        search_space: dict,
        num_samples: int = 20,
        max_epochs: int = 100,
    ) -> Any:
        """Run HPO with random search and ASHA scheduling.

        Args:
            search_space: Dictionary of hyperparameter distributions.
            num_samples: Number of trials to run.
            max_epochs: Maximum epochs per trial.

        Returns:
            Ray Tune analysis results.
        """
        scheduler = self.create_asha_scheduler(max_epochs=max_epochs)
        result = tune.run(
            self.train_fn,
            config=search_space,
            num_samples=num_samples,
            scheduler=scheduler,
            resources_per_trial={"gpu": self.num_gpus_per_trial},
        )
        logger.info("Random search complete: %d trials", num_samples)
        return result

    def run_bayesian_search(
        self,
        search_space: dict,
        num_samples: int = 50,
        max_epochs: int = 100,
    ) -> Any:
        """Run HPO with Bayesian optimization (Optuna) and ASHA.

        Args:
            search_space: Dictionary of hyperparameter distributions.
            num_samples: Number of trials to run.
            max_epochs: Maximum epochs per trial.

        Returns:
            Ray Tune analysis results.

        Raises:
            ImportError: If Optuna is not installed.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Install with: pip install optuna"
            )
        optuna_search = OptunaSearch(metric="val_accuracy", mode="max")
        scheduler = self.create_asha_scheduler(max_epochs=max_epochs)
        result = tune.run(
            self.train_fn,
            config=search_space,
            num_samples=num_samples,
            search_alg=optuna_search,
            scheduler=scheduler,
            resources_per_trial={"gpu": self.num_gpus_per_trial},
        )
        logger.info("Bayesian search complete: %d trials", num_samples)
        return result

    @staticmethod
    def extract_best_config(
        analysis: Any,
        metric: str = "val_accuracy",
        mode: str = "max",
    ) -> Optional[dict]:
        """Extract the best configuration from analysis results.

        Args:
            analysis: Ray Tune analysis object.
            metric: Metric to select the best by.
            mode: 'max' or 'min'.

        Returns:
            Best hyperparameter configuration dictionary.
        """
        best = analysis.get_best_config(metric=metric, mode=mode)
        logger.info("Best config: %s", best)
        return best
