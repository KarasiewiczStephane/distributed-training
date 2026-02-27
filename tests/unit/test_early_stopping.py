"""Tests for distributed early stopping."""

from src.utils.checkpoint import DistributedEarlyStopping


class TestDistributedEarlyStopping:
    """Tests for the DistributedEarlyStopping class."""

    def test_no_stop_when_loss_improves(self):
        """Counter stays at 0 when loss keeps improving."""
        es = DistributedEarlyStopping(patience=3, rank=0, world_size=1)
        assert es.should_stop(1.0) is False
        assert es.should_stop(0.9) is False
        assert es.should_stop(0.8) is False
        assert es.counter == 0

    def test_stop_after_patience_exceeded(self):
        """Training stops after patience epochs without improvement."""
        es = DistributedEarlyStopping(patience=3, rank=0, world_size=1)
        es.should_stop(1.0)  # best = 1.0
        es.should_stop(1.1)  # counter = 1
        es.should_stop(1.2)  # counter = 2
        result = es.should_stop(1.3)  # counter = 3 >= patience
        assert result is True

    def test_counter_resets_on_improvement(self):
        """Counter resets to 0 when loss improves."""
        es = DistributedEarlyStopping(patience=3, rank=0, world_size=1)
        es.should_stop(1.0)
        es.should_stop(1.1)  # counter = 1
        es.should_stop(0.5)  # improves, counter = 0
        assert es.counter == 0

    def test_min_delta(self):
        """Loss must decrease by at least min_delta to count as improvement."""
        es = DistributedEarlyStopping(patience=2, rank=0, world_size=1, min_delta=0.1)
        es.should_stop(1.0)  # best = 1.0
        es.should_stop(0.95)  # not enough improvement (< 0.1)
        assert es.counter == 1
        es.should_stop(0.85)  # improvement of 0.15 > 0.1
        assert es.counter == 0

    def test_should_stop_flag_set(self):
        """should_stop_flag is updated correctly."""
        es = DistributedEarlyStopping(patience=1, rank=0, world_size=1)
        es.should_stop(1.0)
        assert es.should_stop_flag is False
        es.should_stop(1.1)  # counter hits patience
        assert es.should_stop_flag is True
