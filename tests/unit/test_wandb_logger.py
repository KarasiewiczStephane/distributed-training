"""Tests for W&B logger (with mocked wandb)."""

from unittest.mock import MagicMock, patch

from src.tracking.wandb_logger import WandBLogger


class TestWandBLogger:
    """Tests for WandBLogger class."""

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_init_rank_zero(self, mock_wandb):
        """Rank 0 initializes a W&B run."""
        mock_wandb.Settings.return_value = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        wl = WandBLogger(project="test", config={"lr": 0.01}, rank=0, enabled=True)
        assert wl.enabled is True
        mock_wandb.init.assert_called_once()

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_init_rank_nonzero_disabled(self, mock_wandb):
        """Non-rank-0 processes do not log."""
        wl = WandBLogger(project="test", config={"lr": 0.01}, rank=1, enabled=True)
        assert wl.enabled is False

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_log_metrics(self, mock_wandb):
        """log_metrics calls wandb.log on rank 0."""
        mock_wandb.Settings.return_value = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        wl = WandBLogger(project="test", config={}, rank=0, enabled=True)
        wl.log_metrics({"loss": 0.5}, step=1)
        mock_wandb.log.assert_called_with({"loss": 0.5}, step=1)

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_log_metrics_disabled(self, mock_wandb):
        """log_metrics is a no-op when disabled."""
        wl = WandBLogger(project="test", config={}, rank=1, enabled=True)
        wl.log_metrics({"loss": 0.5}, step=1)
        mock_wandb.log.assert_not_called()

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_log_artifact(self, mock_wandb):
        """log_artifact creates and uploads an artifact."""
        mock_wandb.Settings.return_value = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Artifact.return_value = MagicMock()
        wl = WandBLogger(project="test", config={}, rank=0, enabled=True)
        wl.log_artifact("/path/to/model.pt", "model_v1", "model")
        mock_wandb.Artifact.assert_called_once_with("model_v1", type="model")

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_finish(self, mock_wandb):
        """finish calls wandb.finish."""
        mock_wandb.Settings.return_value = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        wl = WandBLogger(project="test", config={}, rank=0, enabled=True)
        wl.finish()
        mock_wandb.finish.assert_called_once()

    def test_disabled_when_wandb_not_available(self):
        """Logger is disabled when wandb is not installed."""
        with patch("src.tracking.wandb_logger.WANDB_AVAILABLE", False):
            wl = WandBLogger(project="test", config={}, rank=0, enabled=True)
            assert wl.enabled is False

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_log_system_metrics_disabled(self, mock_wandb):
        """log_system_metrics is a no-op when disabled."""
        wl = WandBLogger(project="test", config={}, rank=1, enabled=True)
        wl.log_system_metrics()
        mock_wandb.log.assert_not_called()

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_log_artifact_disabled(self, mock_wandb):
        """log_artifact is a no-op when disabled."""
        wl = WandBLogger(project="test", config={}, rank=1, enabled=True)
        wl.log_artifact("/path", "name", "model")
        mock_wandb.Artifact.assert_not_called()

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_create_scaling_chart_disabled(self, mock_wandb):
        """create_scaling_chart is a no-op when disabled."""
        wl = WandBLogger(project="test", config={}, rank=1, enabled=True)
        wl.create_scaling_chart([{"num_gpus": 1, "speedup": 1.0, "efficiency": 1.0}])
        mock_wandb.Table.assert_not_called()

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_create_scaling_chart_enabled(self, mock_wandb):
        """create_scaling_chart creates table and logs chart."""
        mock_wandb.Settings.return_value = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table
        mock_wandb.plot.line.return_value = MagicMock()

        wl = WandBLogger(project="test", config={}, rank=0, enabled=True)
        wl.create_scaling_chart(
            [
                {"num_gpus": 1, "speedup": 1.0, "efficiency": 1.0},
                {"num_gpus": 2, "speedup": 1.8, "efficiency": 0.9},
            ]
        )
        assert mock_table.add_data.call_count == 2
        mock_wandb.log.assert_called_once()

    @patch("src.tracking.wandb_logger.WANDB_AVAILABLE", True)
    @patch("src.tracking.wandb_logger.wandb")
    def test_finish_disabled(self, mock_wandb):
        """finish is a no-op when disabled."""
        wl = WandBLogger(project="test", config={}, rank=1, enabled=True)
        wl.finish()
        mock_wandb.finish.assert_not_called()
