"""Entry point for distributed training framework."""

import argparse
import logging

from src.data.dataset import get_cifar100
from src.data.loader import create_dataloader
from src.models.resnet import create_resnet50
from src.training.base_trainer import SingleGPUTrainer
from src.training.ddp_trainer import DDPTrainer
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Distributed Training Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration YAML.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training (DDP).",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="single",
        choices=["single", "ddp", "horovod"],
        help="Training backend to use.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training based on CLI arguments and configuration."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(level="INFO")

    logger.info("Starting training with config: %s", args.config)

    model = create_resnet50(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
    )

    train_dataset = get_cifar100(root=config.data.root, train=True)
    val_dataset = get_cifar100(root=config.data.root, train=False)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    trainer_type = args.trainer
    if args.distributed:
        trainer_type = "ddp"

    if trainer_type == "single":
        trainer = SingleGPUTrainer(model, train_loader, val_loader, config)
    elif trainer_type == "ddp":
        trainer = DDPTrainer(model, train_dataset, val_dataset, config)
    else:
        raise NotImplementedError(f"Trainer '{trainer_type}' not yet implemented")

    result = trainer.train()
    logger.info("Training complete. Final metrics: %s", result["history"][-1])

    if trainer_type == "ddp":
        trainer.cleanup()


if __name__ == "__main__":
    main()
