"""Tests for Ray Tune HPO module (with mocked ray)."""

import importlib
import sys
from unittest.mock import MagicMock


def _setup_ray_mock():
    """Set up mocked ray and ray.tune for testing."""
    mock_tune = MagicMock()
    mock_tune.loguniform = MagicMock(side_effect=lambda lo, hi: (lo, hi))
    mock_tune.uniform = MagicMock(side_effect=lambda lo, hi: (lo, hi))
    mock_tune.choice = MagicMock(side_effect=lambda vals: vals)

    mock_ray = MagicMock()
    mock_ray.tune = mock_tune

    mock_scheduler = MagicMock()
    mock_tune_schedulers = MagicMock()
    mock_tune_schedulers.ASHAScheduler = MagicMock(return_value=mock_scheduler)

    mock_optuna = MagicMock()
    mock_optuna.OptunaSearch = MagicMock()

    sys.modules["ray"] = mock_ray
    sys.modules["ray.tune"] = mock_tune
    sys.modules["ray.tune.schedulers"] = mock_tune_schedulers
    sys.modules["ray.tune.search"] = MagicMock()
    sys.modules["ray.tune.search.optuna"] = mock_optuna

    import src.hpo.ray_tune_search as rts_mod
    import src.hpo.search_spaces as ss_mod

    importlib.reload(rts_mod)
    importlib.reload(ss_mod)

    return mock_tune, rts_mod, ss_mod


def _cleanup_ray_mock():
    """Remove mocked ray from sys.modules."""
    for key in list(sys.modules.keys()):
        if key.startswith("ray"):
            sys.modules.pop(key)
    import src.hpo.ray_tune_search as rts_mod
    import src.hpo.search_spaces as ss_mod

    importlib.reload(rts_mod)
    importlib.reload(ss_mod)


class TestSearchSpaces:
    """Tests for search space generation."""

    def test_default_search_space(self):
        """Default search space contains expected parameters."""
        _, _, ss_mod = _setup_ray_mock()
        try:
            space = ss_mod.get_default_search_space()
            assert "lr" in space
            assert "batch_size" in space
            assert "weight_decay" in space
            assert "dropout" in space
        finally:
            _cleanup_ray_mock()

    def test_search_space_from_config(self):
        """Custom search space is built from config dict."""
        _, _, ss_mod = _setup_ray_mock()
        try:
            config = {
                "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                "batch_size": {"type": "choice", "values": [32, 64]},
            }
            space = ss_mod.get_search_space_from_config(config)
            assert "lr" in space
            assert "batch_size" in space
        finally:
            _cleanup_ray_mock()


class TestHPORunner:
    """Tests for the HPORunner class."""

    def test_init(self):
        """HPORunner initializes with a training function."""
        mock_tune, rts_mod, _ = _setup_ray_mock()
        try:
            runner = rts_mod.HPORunner(
                train_fn=lambda config: None,
                config={},
                num_gpus_per_trial=0,
            )
            assert runner.num_gpus_per_trial == 0
        finally:
            _cleanup_ray_mock()

    def test_create_asha_scheduler(self):
        """ASHA scheduler is created with correct parameters."""
        _, rts_mod, _ = _setup_ray_mock()
        try:
            runner = rts_mod.HPORunner(train_fn=lambda config: None, config={})
            scheduler = runner.create_asha_scheduler(max_epochs=50, grace_period=5)
            assert scheduler is not None
        finally:
            _cleanup_ray_mock()

    def test_run_random_search(self):
        """Random search calls tune.run."""
        mock_tune, rts_mod, _ = _setup_ray_mock()
        try:
            runner = rts_mod.HPORunner(train_fn=lambda config: None, config={})
            runner.run_random_search(search_space={"lr": 0.01}, num_samples=5)
            mock_tune.run.assert_called_once()
        finally:
            _cleanup_ray_mock()

    def test_extract_best_config(self):
        """extract_best_config calls analysis.get_best_config."""
        _, rts_mod, _ = _setup_ray_mock()
        try:
            mock_analysis = MagicMock()
            mock_analysis.get_best_config.return_value = {"lr": 0.001}
            best = rts_mod.HPORunner.extract_best_config(mock_analysis)
            assert best == {"lr": 0.001}
            mock_analysis.get_best_config.assert_called_once()
        finally:
            _cleanup_ray_mock()
