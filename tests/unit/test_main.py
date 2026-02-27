"""Tests for the main entry point."""

from unittest.mock import patch

from src.main import parse_args


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_args(self):
        """Default arguments use single trainer and default config."""
        with patch("sys.argv", ["main"]):
            args = parse_args()
        assert args.config == "configs/training.yaml"
        assert args.trainer == "single"
        assert args.distributed is False

    def test_distributed_flag(self):
        """--distributed flag sets distributed to True."""
        with patch("sys.argv", ["main", "--distributed"]):
            args = parse_args()
        assert args.distributed is True

    def test_trainer_choice(self):
        """--trainer accepts valid choices."""
        with patch("sys.argv", ["main", "--trainer", "ddp"]):
            args = parse_args()
        assert args.trainer == "ddp"

    def test_custom_config(self):
        """--config overrides the default config path."""
        with patch("sys.argv", ["main", "--config", "custom.yaml"]):
            args = parse_args()
        assert args.config == "custom.yaml"
