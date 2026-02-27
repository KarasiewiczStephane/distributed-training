"""Tests for the main entry point."""

from unittest.mock import MagicMock, patch

import pytest

from src.main import main, parse_args


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


class TestMain:
    """Tests for the main() function."""

    @patch("src.main.SingleGPUTrainer")
    @patch("src.main.create_dataloader")
    @patch("src.main.get_cifar100")
    @patch("src.main.create_resnet50")
    @patch("src.main.load_config")
    def test_main_single_trainer(
        self, mock_config, mock_model, mock_dataset, mock_loader, mock_trainer
    ):
        """main() creates SingleGPUTrainer for default args."""
        cfg = MagicMock()
        cfg.model.num_classes = 10
        cfg.model.pretrained = False
        cfg.data.root = "./data"
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.training.batch_size = 32
        mock_config.return_value = cfg
        mock_model.return_value = MagicMock()
        mock_dataset.return_value = MagicMock()
        mock_loader.return_value = MagicMock()

        trainer_inst = MagicMock()
        trainer_inst.train.return_value = {"history": [{"train_loss": 0.5}]}
        mock_trainer.return_value = trainer_inst

        with patch("sys.argv", ["main", "--config", "configs/training.yaml"]):
            main()

        mock_trainer.assert_called_once()
        trainer_inst.train.assert_called_once()

    @patch("src.main.SingleGPUTrainer")
    @patch("src.main.create_dataloader")
    @patch("src.main.get_cifar100")
    @patch("src.main.create_resnet50")
    @patch("src.main.load_config")
    def test_main_unsupported_trainer(
        self, mock_config, mock_model, mock_dataset, mock_loader, mock_trainer
    ):
        """main() raises NotImplementedError for horovod trainer."""
        cfg = MagicMock()
        cfg.model.num_classes = 10
        cfg.model.pretrained = False
        cfg.data.root = "./data"
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.training.batch_size = 32
        mock_config.return_value = cfg
        mock_model.return_value = MagicMock()
        mock_dataset.return_value = MagicMock()
        mock_loader.return_value = MagicMock()

        with (
            patch("sys.argv", ["main", "--trainer", "horovod"]),
            pytest.raises(NotImplementedError, match="horovod"),
        ):
            main()
