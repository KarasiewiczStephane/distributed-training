"""Tests for checkpoint management."""

import torch

from src.utils.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Tests for CheckpointManager save/load functionality."""

    def test_save_and_load(self, tmp_path):
        """Checkpoint round-trip preserves state dict contents."""
        mgr = CheckpointManager(str(tmp_path), rank=0)
        state = {"epoch": 5, "model_state": {"weight": torch.tensor([1.0, 2.0])}}
        mgr.save(state, "test.pt")
        loaded = mgr.load("test.pt")
        assert loaded is not None
        assert loaded["epoch"] == 5
        assert torch.equal(loaded["model_state"]["weight"], torch.tensor([1.0, 2.0]))

    def test_save_best(self, tmp_path):
        """is_best=True creates an additional best.pt file."""
        mgr = CheckpointManager(str(tmp_path), rank=0)
        state = {"epoch": 3}
        mgr.save(state, "epoch_3.pt", is_best=True)
        assert (tmp_path / "epoch_3.pt").exists()
        assert (tmp_path / "best.pt").exists()

    def test_rank_nonzero_skips_save(self, tmp_path):
        """Non-rank-0 processes do not write checkpoint files."""
        mgr = CheckpointManager(str(tmp_path), rank=1)
        state = {"epoch": 1}
        mgr.save(state, "should_not_exist.pt")
        assert not (tmp_path / "should_not_exist.pt").exists()

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading a missing checkpoint returns None."""
        mgr = CheckpointManager(str(tmp_path), rank=0)
        result = mgr.load("nonexistent.pt")
        assert result is None

    def test_creates_directory(self, tmp_path):
        """Save dir is created automatically on rank 0."""
        save_dir = tmp_path / "nested" / "checkpoints"
        mgr = CheckpointManager(str(save_dir), rank=0)
        assert save_dir.exists()
        state = {"epoch": 1}
        mgr.save(state, "test.pt")
        assert (save_dir / "test.pt").exists()
