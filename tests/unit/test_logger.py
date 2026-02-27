"""Tests for the structured logging setup."""

import logging

from src.utils.logger import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_rank_zero_gets_info_level(self):
        """Rank 0 process gets the requested log level."""
        setup_logging(level="INFO", rank=0)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_non_rank_zero_gets_warning(self):
        """Non-rank-0 processes are set to WARNING level."""
        setup_logging(level="INFO", rank=1)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_debug_level(self):
        """DEBUG level is applied correctly for rank 0."""
        setup_logging(level="DEBUG", rank=0)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_handler_is_stream_handler(self):
        """Root logger has exactly one StreamHandler after setup."""
        setup_logging(level="INFO", rank=0)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
