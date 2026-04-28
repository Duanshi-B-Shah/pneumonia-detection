"""Structured logging setup for the project."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", name: str = "pneumonia") -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_logger(name: str = "pneumonia") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
