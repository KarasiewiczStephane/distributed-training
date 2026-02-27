"""Tests for configuration loading and merging."""

import pytest
from omegaconf import DictConfig

from src.utils.config import config_to_dict, load_config, merge_configs


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Config loads successfully from a valid YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("model:\n  name: resnet50\n  num_classes: 100\n")
        cfg = load_config(str(config_file))
        assert isinstance(cfg, DictConfig)
        assert cfg.model.name == "resnet50"
        assert cfg.model.num_classes == 100

    def test_load_nonexistent_config(self):
        """FileNotFoundError raised for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_training_config(self):
        """Training config from the project loads correctly."""
        cfg = load_config("configs/training.yaml")
        assert cfg.model.name == "resnet50"
        assert cfg.training.epochs == 100
        assert cfg.training.batch_size == 128


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_two_configs(self, tmp_path):
        """Merging two configs overrides values from the first."""
        base = tmp_path / "base.yaml"
        base.write_text("training:\n  lr: 0.1\n  epochs: 100\n")
        override = tmp_path / "override.yaml"
        override.write_text("training:\n  lr: 0.01\n")
        merged = merge_configs(str(base), str(override))
        assert merged.training.lr == 0.01
        assert merged.training.epochs == 100


class TestConfigToDict:
    """Tests for config_to_dict function."""

    def test_converts_to_plain_dict(self, tmp_path):
        """DictConfig is converted to a regular Python dict."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("a: 1\nb: 2\n")
        cfg = load_config(str(config_file))
        result = config_to_dict(cfg)
        assert isinstance(result, dict)
        assert result == {"a": 1, "b": 2}
