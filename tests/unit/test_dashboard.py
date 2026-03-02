"""Tests for the distributed training dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    generate_experiment_table,
    generate_scaling_data,
    generate_training_curves,
)


class TestGenerateTrainingCurves:
    """Tests for generate_training_curves."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_training_curves()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "epoch",
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "experiment",
        }
        assert expected_cols == set(df.columns)

    def test_row_count(self):
        """Generates correct number of rows for epochs x experiments."""
        df = generate_training_curves(num_epochs=10, num_experiments=2)
        assert len(df) == 20

    def test_default_experiments(self):
        """Default call produces 3 experiments with 50 epochs each."""
        df = generate_training_curves()
        assert df["experiment"].nunique() == 3
        assert len(df) == 150

    def test_loss_is_positive(self):
        """All loss values are positive."""
        df = generate_training_curves()
        assert (df["loss"] > 0).all()
        assert (df["val_loss"] > 0).all()

    def test_accuracy_range(self):
        """Accuracy values are between 0 and 1."""
        df = generate_training_curves()
        assert (df["accuracy"] >= 0).all()
        assert (df["accuracy"] <= 1).all()
        assert (df["val_accuracy"] >= 0).all()
        assert (df["val_accuracy"] <= 1).all()

    def test_learning_rate_non_negative(self):
        """Learning rates are non-negative."""
        df = generate_training_curves()
        assert (df["learning_rate"] >= 0).all()

    def test_reproducible_with_seed(self):
        """Same seed produces identical output."""
        df1 = generate_training_curves(seed=123)
        df2 = generate_training_curves(seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        df1 = generate_training_curves(seed=1)
        df2 = generate_training_curves(seed=2)
        assert not df1["loss"].equals(df2["loss"])


class TestGenerateScalingData:
    """Tests for generate_scaling_data."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_scaling_data()
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "gpu_count",
            "framework",
            "throughput",
            "speedup",
            "efficiency",
            "time_seconds",
            "memory_gb",
        }
        assert expected_cols == set(df.columns)

    def test_frameworks(self):
        """Contains both DDP and Horovod frameworks."""
        df = generate_scaling_data()
        assert set(df["framework"]) == {"DDP", "Horovod"}

    def test_gpu_counts(self):
        """Contains expected GPU count values."""
        df = generate_scaling_data()
        assert set(df["gpu_count"]) == {1, 2, 4, 8}

    def test_row_count(self):
        """Expected number of rows: 4 gpu counts x 2 frameworks."""
        df = generate_scaling_data()
        assert len(df) == 8

    def test_throughput_positive(self):
        """All throughput values are positive."""
        df = generate_scaling_data()
        assert (df["throughput"] > 0).all()

    def test_speedup_increases_with_gpus(self):
        """Speedup generally increases with more GPUs for each framework."""
        df = generate_scaling_data()
        for framework in ["DDP", "Horovod"]:
            fw_df = df[df["framework"] == framework].sort_values("gpu_count")
            speedups = fw_df["speedup"].tolist()
            assert speedups[-1] > speedups[0]


class TestGenerateExperimentTable:
    """Tests for generate_experiment_table."""

    def test_returns_dataframe(self):
        """Returns a DataFrame with expected columns."""
        df = generate_experiment_table()
        assert isinstance(df, pd.DataFrame)
        assert "Experiment" in df.columns
        assert "Best Accuracy" in df.columns
        assert "Training Time (min)" in df.columns

    def test_experiment_count(self):
        """Contains 3 experiments."""
        df = generate_experiment_table()
        assert len(df) == 3

    def test_accuracy_range(self):
        """Accuracy values are between 0 and 1."""
        df = generate_experiment_table()
        assert (df["Best Accuracy"] > 0).all()
        assert (df["Best Accuracy"] <= 1).all()

    def test_training_time_positive(self):
        """Training times are positive."""
        df = generate_experiment_table()
        assert (df["Training Time (min)"] > 0).all()
