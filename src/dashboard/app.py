"""Streamlit dashboard for distributed training experiment visualization.

Displays training loss curves, GPU scaling benchmarks, experiment
comparisons, and resource utilization metrics using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def generate_training_curves(
    num_epochs: int = 50,
    num_experiments: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic training loss and accuracy curves.

    Args:
        num_epochs: Number of training epochs to simulate.
        num_experiments: Number of experiment runs.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with epoch, loss, accuracy, val_loss, val_accuracy,
        learning_rate, and experiment columns.
    """
    rng = np.random.default_rng(seed)
    records = []
    experiment_names = [
        "ResNet50-DDP-4GPU",
        "ResNet50-Horovod-4GPU",
        "ResNet50-SingleGPU",
    ]

    for exp_idx in range(num_experiments):
        base_loss = 2.5 - exp_idx * 0.1
        convergence_rate = 0.06 + exp_idx * 0.005
        noise_scale = 0.05

        for epoch in range(1, num_epochs + 1):
            t = epoch / num_epochs
            loss = base_loss * np.exp(-convergence_rate * epoch) + rng.normal(
                0, noise_scale
            )
            loss = max(loss, 0.05)
            val_loss = loss + 0.1 * (1 - t) + rng.normal(0, noise_scale * 1.2)
            val_loss = max(val_loss, 0.08)
            accuracy = 1.0 - loss / base_loss + rng.normal(0, 0.01)
            accuracy = np.clip(accuracy, 0.0, 0.99)
            val_accuracy = accuracy - 0.02 * (1 - t) + rng.normal(0, 0.01)
            val_accuracy = np.clip(val_accuracy, 0.0, 0.98)

            warmup_epochs = 5
            if epoch <= warmup_epochs:
                lr = 0.001 + (0.1 - 0.001) * (epoch / warmup_epochs)
            else:
                lr = (
                    0.1
                    * 0.5
                    * (
                        1
                        + np.cos(
                            np.pi
                            * (epoch - warmup_epochs)
                            / (num_epochs - warmup_epochs)
                        )
                    )
                )

            records.append(
                {
                    "epoch": epoch,
                    "loss": round(loss, 4),
                    "accuracy": round(accuracy, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_accuracy, 4),
                    "learning_rate": round(lr, 6),
                    "experiment": experiment_names[exp_idx],
                }
            )

    return pd.DataFrame(records)


def generate_scaling_data() -> pd.DataFrame:
    """Generate synthetic GPU scaling benchmark data.

    Returns:
        DataFrame with gpu_count, framework, throughput, speedup,
        efficiency, time_seconds, and memory_gb columns.
    """
    records = []
    gpu_counts = [1, 2, 4, 8]

    for framework in ["DDP", "Horovod"]:
        base_throughput = 320 if framework == "DDP" else 310
        for n_gpus in gpu_counts:
            overhead = 0.05 if framework == "DDP" else 0.08
            efficiency = 1.0 - overhead * np.log2(max(n_gpus, 1))
            efficiency = max(efficiency, 0.6)
            throughput = base_throughput * n_gpus * efficiency
            speedup = throughput / base_throughput
            time_s = 10000 / throughput

            records.append(
                {
                    "gpu_count": n_gpus,
                    "framework": framework,
                    "throughput": round(throughput, 1),
                    "speedup": round(speedup, 2),
                    "efficiency": round(efficiency * 100, 1),
                    "time_seconds": round(time_s, 1),
                    "memory_gb": round(4.2 + 0.3 * n_gpus, 1),
                }
            )

    return pd.DataFrame(records)


def generate_experiment_table() -> pd.DataFrame:
    """Generate experiment comparison table with hyperparameters and results.

    Returns:
        DataFrame with experiment configuration and final metrics.
    """
    return pd.DataFrame(
        [
            {
                "Experiment": "ResNet50-DDP-4GPU",
                "Framework": "DDP",
                "GPUs": 4,
                "Batch Size": 256,
                "Learning Rate": 0.1,
                "Epochs": 50,
                "Final Loss": 0.18,
                "Best Accuracy": 0.945,
                "Training Time (min)": 42.3,
            },
            {
                "Experiment": "ResNet50-Horovod-4GPU",
                "Framework": "Horovod",
                "GPUs": 4,
                "Batch Size": 256,
                "Learning Rate": 0.1,
                "Epochs": 50,
                "Final Loss": 0.21,
                "Best Accuracy": 0.938,
                "Training Time (min)": 47.1,
            },
            {
                "Experiment": "ResNet50-SingleGPU",
                "Framework": "Single",
                "GPUs": 1,
                "Batch Size": 64,
                "Learning Rate": 0.01,
                "Epochs": 50,
                "Final Loss": 0.25,
                "Best Accuracy": 0.921,
                "Training Time (min)": 156.8,
            },
        ]
    )


def render_training_curves(df: pd.DataFrame) -> None:
    """Render training loss and accuracy curves section."""
    st.header("Training Loss Curves")

    col1, col2 = st.columns(2)

    with col1:
        fig_loss = px.line(
            df,
            x="epoch",
            y="loss",
            color="experiment",
            title="Training Loss Over Epochs",
            labels={"loss": "Loss", "epoch": "Epoch"},
        )
        fig_loss.update_layout(height=400)
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        fig_acc = px.line(
            df,
            x="epoch",
            y="accuracy",
            color="experiment",
            title="Training Accuracy Over Epochs",
            labels={"accuracy": "Accuracy", "epoch": "Epoch"},
        )
        fig_acc.update_layout(height=400)
        st.plotly_chart(fig_acc, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig_val = px.line(
            df,
            x="epoch",
            y="val_loss",
            color="experiment",
            title="Validation Loss Over Epochs",
            labels={"val_loss": "Validation Loss", "epoch": "Epoch"},
        )
        fig_val.update_layout(height=400)
        st.plotly_chart(fig_val, use_container_width=True)

    with col4:
        fig_lr = px.line(
            df[df["experiment"] == df["experiment"].iloc[0]],
            x="epoch",
            y="learning_rate",
            title="Learning Rate Schedule (Warmup + Cosine)",
            labels={"learning_rate": "Learning Rate", "epoch": "Epoch"},
        )
        fig_lr.update_layout(height=400)
        st.plotly_chart(fig_lr, use_container_width=True)


def render_scaling_benchmark(df: pd.DataFrame) -> None:
    """Render GPU scaling benchmark section."""
    st.header("GPU Scaling Benchmark")

    col1, col2 = st.columns(2)

    with col1:
        fig_speedup = px.bar(
            df,
            x="gpu_count",
            y="speedup",
            color="framework",
            barmode="group",
            title="Speedup vs Number of GPUs",
            labels={"speedup": "Speedup (x)", "gpu_count": "GPU Count"},
        )
        ideal = pd.DataFrame({"gpu_count": [1, 2, 4, 8], "speedup": [1, 2, 4, 8]})
        fig_speedup.add_trace(
            go.Scatter(
                x=ideal["gpu_count"],
                y=ideal["speedup"],
                mode="lines+markers",
                name="Ideal Linear",
                line={"dash": "dash", "color": "gray"},
            )
        )
        fig_speedup.update_layout(height=400)
        st.plotly_chart(fig_speedup, use_container_width=True)

    with col2:
        fig_throughput = px.bar(
            df,
            x="gpu_count",
            y="throughput",
            color="framework",
            barmode="group",
            title="Training Throughput (samples/sec)",
            labels={
                "throughput": "Throughput (samples/sec)",
                "gpu_count": "GPU Count",
            },
        )
        fig_throughput.update_layout(height=400)
        st.plotly_chart(fig_throughput, use_container_width=True)

    fig_eff = px.line(
        df,
        x="gpu_count",
        y="efficiency",
        color="framework",
        markers=True,
        title="Scaling Efficiency (%)",
        labels={"efficiency": "Efficiency (%)", "gpu_count": "GPU Count"},
    )
    fig_eff.update_layout(height=350)
    st.plotly_chart(fig_eff, use_container_width=True)


def render_resource_utilization() -> None:
    """Render simulated resource utilization gauges."""
    st.header("Resource Utilization (Simulated)")

    col1, col2, col3, col4 = st.columns(4)

    gauges = [
        ("GPU Compute", 87, col1),
        ("GPU Memory", 72, col2),
        ("CPU Usage", 45, col3),
        ("Network I/O", 63, col4),
    ]

    for label, value, col in gauges:
        with col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={"text": label},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1f77b4"},
                        "steps": [
                            {"range": [0, 50], "color": "#e8f4f8"},
                            {"range": [50, 80], "color": "#b3d9e8"},
                            {"range": [80, 100], "color": "#ff9999"},
                        ],
                    },
                    number={"suffix": "%"},
                )
            )
            fig.update_layout(height=250, margin={"t": 50, "b": 0, "l": 20, "r": 20})
            st.plotly_chart(fig, use_container_width=True)


def render_experiment_comparison(df: pd.DataFrame) -> None:
    """Render experiment comparison table section."""
    st.header("Experiment Comparison")

    st.dataframe(
        df.style.format(
            {
                "Learning Rate": "{:.3f}",
                "Final Loss": "{:.3f}",
                "Best Accuracy": "{:.3f}",
                "Training Time (min)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)
    best = df.loc[df["Best Accuracy"].idxmax()]
    fastest = df.loc[df["Training Time (min)"].idxmin()]

    col1.metric("Best Accuracy", f"{best['Best Accuracy']:.1%}", best["Experiment"])
    col2.metric(
        "Fastest Training",
        f"{fastest['Training Time (min)']:.1f} min",
        fastest["Experiment"],
    )
    col3.metric(
        "Speedup (4GPU vs 1GPU)",
        f"{156.8 / 42.3:.1f}x",
        "DDP",
    )


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all synthetic demo data.

    Returns:
        Tuple of (training_curves, scaling_data, experiment_table).
    """
    return (
        generate_training_curves(),
        generate_scaling_data(),
        generate_experiment_table(),
    )


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Distributed Training Dashboard",
        page_icon="🔧",
        layout="wide",
    )

    st.title("Distributed Training Dashboard")
    st.markdown(
        "Training metrics, GPU scaling benchmarks, and experiment comparisons "
        "for PyTorch DDP and Horovod distributed training."
    )

    training_df, scaling_df, experiments_df = load_demo_data()

    render_training_curves(training_df)
    st.divider()
    render_scaling_benchmark(scaling_df)
    st.divider()
    render_experiment_comparison(experiments_df)
    st.divider()
    render_resource_utilization()


if __name__ == "__main__":
    main()
