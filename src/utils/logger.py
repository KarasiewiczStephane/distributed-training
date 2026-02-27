"""Structured logging setup for the distributed training framework."""

import logging
import sys


def setup_logging(level: str = "INFO", rank: int = 0) -> None:
    """Configure structured logging for the application.

    Only rank 0 logs at the requested level; other ranks log WARNING and above
    to avoid duplicate output in distributed settings.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG").
        rank: Process rank in distributed training. Only rank 0 gets full logging.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    if rank == 0:
        root_logger.setLevel(log_level)
    else:
        root_logger.setLevel(logging.WARNING)
