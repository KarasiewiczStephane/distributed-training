"""Tests for mixed precision training (AMP)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.mixed_precision import AMPMixin, benchmark_amp


class ConcreteAMP(AMPMixin):
    """Concrete class using AMPMixin for testing."""

    pass


class TestAMPMixin:
    """Tests for AMPMixin class."""

    def test_setup_amp_disabled_on_cpu(self):
        """AMP is disabled when no CUDA is available."""
        amp = ConcreteAMP()
        amp.setup_amp(enabled=True)
        # On CPU machines, amp_enabled should be False
        if not torch.cuda.is_available():
            assert amp.amp_enabled is False

    def test_setup_amp_disabled_explicitly(self):
        """AMP is disabled when enabled=False."""
        amp = ConcreteAMP()
        amp.setup_amp(enabled=False)
        assert amp.amp_enabled is False

    def test_autocast_context_cpu(self):
        """On CPU, get_autocast_context returns nullcontext."""
        amp = ConcreteAMP()
        amp.setup_amp(enabled=False)
        ctx = amp.get_autocast_context()
        # Should be a nullcontext (usable in with-statement)
        with ctx:
            pass

    def test_train_step_amp_fp32(self):
        """train_step_amp works in FP32 mode on CPU."""
        amp = ConcreteAMP()
        amp.setup_amp(enabled=False)

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        inputs = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))

        loss, outputs = amp.train_step_amp(
            inputs, targets, model, criterion, optimizer, gradient_clip=1.0
        )
        assert loss > 0
        assert outputs.shape == (8, 10)

    def test_scaler_initialized(self):
        """GradScaler is initialized after setup_amp."""
        amp = ConcreteAMP()
        amp.setup_amp(enabled=True)
        assert hasattr(amp, "scaler")


class TestBenchmarkAMP:
    """Tests for the benchmark_amp function."""

    def test_benchmark_returns_metrics(self):
        """benchmark_amp returns fp32_time, amp_time, speedup."""
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
        loader = DataLoader(dataset, batch_size=8)
        device = torch.device("cpu")

        result = benchmark_amp(model, loader, device, num_iterations=2)
        assert "fp32_time" in result
        assert "amp_time" in result
        assert "speedup" in result
        assert result["fp32_time"] > 0
        assert result["amp_time"] > 0
