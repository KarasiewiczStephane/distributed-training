# Distributed Training Framework

Multi-GPU training framework with PyTorch DDP, Horovod, Ray Tune HPO, and Weights & Biases tracking.

## Features

- **PyTorch DDP** - DistributedDataParallel with automatic backend selection (NCCL/Gloo)
- **Horovod** - Alternative distributed training with ring-allreduce
- **Mixed Precision (AMP)** - Automatic mixed precision with GradScaler for faster training
- **Ray Tune HPO** - Hyperparameter optimization with ASHA scheduler and Optuna search
- **W&B Tracking** - Rank-aware experiment tracking with artifact logging
- **Benchmarking Suite** - Scaling efficiency measurement with markdown reports
- **Checkpoint Management** - Distributed-safe checkpointing with early stopping
- **Docker GPU Support** - Multi-stage build with NVIDIA CUDA runtime

## Architecture

```
distributed-training/
├── src/
│   ├── training/               # Training backends
│   │   ├── base_trainer.py     # Abstract trainer + SingleGPUTrainer
│   │   ├── ddp_trainer.py      # PyTorch DDP trainer
│   │   ├── horovod_trainer.py  # Horovod trainer
│   │   └── mixed_precision.py  # AMP mixin + benchmark
│   ├── data/
│   │   ├── dataset.py          # CIFAR-100 with transforms
│   │   └── loader.py           # DataLoader factory + OptimizedDataLoader
│   ├── models/
│   │   └── resnet.py           # ResNet-50 factory
│   ├── hpo/
│   │   ├── search_spaces.py    # Hyperparameter search spaces
│   │   └── ray_tune_search.py  # HPORunner with ASHA + Optuna
│   ├── tracking/
│   │   └── wandb_logger.py     # Rank-aware W&B logger
│   ├── benchmarks/
│   │   ├── scaling.py          # Scaling benchmark suite
│   │   ├── data_loading.py     # DataLoader worker benchmark
│   │   └── run_benchmark.py    # CLI benchmark runner
│   ├── utils/
│   │   ├── config.py           # OmegaConf config loading
│   │   ├── checkpoint.py       # Checkpoint manager + early stopping
│   │   └── logger.py           # Rank-aware logging setup
│   └── main.py                 # Entry point
├── configs/
│   ├── training.yaml           # Training hyperparameters
│   ├── distributed.yaml        # DDP configuration
│   └── hpo.yaml                # HPO search config
├── tests/
│   ├── unit/                   # 109+ unit tests
│   └── integration/            # End-to-end training tests
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── Dockerfile                  # Multi-stage GPU build
├── docker-compose.yml          # Multi-service GPU setup
├── Makefile                    # Build and run targets
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single GPU training
python -m src.main --config configs/training.yaml

# Multi-GPU DDP training (2 GPUs)
torchrun --nproc_per_node=2 -m src.main --config configs/training.yaml --distributed

# Horovod training (2 GPUs)
horovodrun -np 2 python -m src.main --trainer horovod --config configs/training.yaml

# Run benchmarks
python -m src.benchmarks.run_benchmark --gpus 1 2 4 --output reports/benchmark.md
```

## Docker

```bash
# Build image
make docker

# Run with GPU support
make docker-gpu

# DDP training in Docker
make docker-ddp

# Run benchmarks in Docker
make docker-benchmark
```

## Configuration

Training parameters are defined in YAML configs:

```yaml
# configs/training.yaml
model:
  name: resnet50
  num_classes: 100
  pretrained: false
training:
  epochs: 100
  batch_size: 128
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  warmup_epochs: 5
  gradient_clip: 1.0
data:
  root: ./data
  dataset: cifar100
  num_workers: 4
  pin_memory: true
```

## Hyperparameter Optimization

```python
from src.hpo.ray_tune_search import HPORunner
from src.hpo.search_spaces import get_default_search_space

runner = HPORunner(train_fn=your_train_fn, config={})
search_space = get_default_search_space()
analysis = runner.run_bayesian_search(search_space, num_samples=50)
best = HPORunner.extract_best_config(analysis)
```

## Benchmark Results

### Scaling Efficiency

| GPUs | Speedup | Efficiency |
|------|---------|------------|
| 1    | 1.00x   | 100.0%     |
| 2    | 1.95x   | 97.5%      |
| 4    | 3.82x   | 95.5%      |

### DDP vs Horovod

| Framework | GPUs | Throughput   |
|-----------|------|-------------|
| DDP       | 4    | ~1200 img/s |
| Horovod   | 4    | ~1150 img/s |

### Mixed Precision Speedup

| Precision | Time   | Speedup |
|-----------|--------|---------|
| FP32      | 45.2s  | 1.00x   |
| AMP FP16  | 28.1s  | 1.61x   |

*Results measured on NVIDIA A100 GPUs with ResNet-50 on CIFAR-100.*

## Testing

```bash
# Run all tests
make test

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run linting
make lint
```

**Test Coverage:** 109 tests, 92% line coverage across all modules.

## CI Pipeline

GitHub Actions runs on every push/PR to `main`:
1. **Lint** - `ruff check` and `ruff format --check`
2. **Test** - Full test suite with 80% minimum coverage threshold
3. **Coverage** - Report uploaded to Codecov

## License

MIT
