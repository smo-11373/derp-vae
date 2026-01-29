"""Logging utilities for Derp-VAE training and inference."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
        log_file: Optional path to log file. If None, logs only to console.
        fmt: Log message format string.
        datefmt: Date format string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_training_logger(
    experiment_name: Optional[str] = None,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger specifically for training runs.

    Args:
        experiment_name: Name of the experiment. If None, uses timestamp.
        log_dir: Directory to store log files.
        level: Logging level.

    Returns:
        Configured logger with both console and file output.
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{experiment_name}.log"

    return get_logger(
        name=f"derp_vae.{experiment_name}",
        level=level,
        log_file=str(log_file),
    )


class MetricLogger:
    """Logger for tracking and displaying training metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize metric logger.

        Args:
            logger: Optional logger instance. Creates default if None.
        """
        self.logger = logger or get_logger("derp_vae.metrics")
        self.history: dict[str, list[float]] = {}

    def log_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names to values.
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Epoch {epoch:04d} | {metric_str}")

    def log_health(self, msi: float, beta: float, separation: float) -> None:
        """Log health metrics with status indicators.

        Args:
            msi: Manifold Stability Index.
            beta: Calibration beta.
            separation: px separation.
        """
        msi_status = "OK" if msi > 0.80 else ("WARN" if msi > 0.60 else "CRIT")
        beta_status = "OK" if 0.10 <= beta <= 0.50 else "WARN"
        sep_status = "OK" if separation > 0.02 else "WARN"

        self.logger.info(
            f"Health | MSI: {msi:.4f} [{msi_status}] | "
            f"Beta: {beta:.4f} [{beta_status}] | "
            f"Separation: {separation:.4f} [{sep_status}]"
        )

    def get_history(self, metric_name: str) -> list[float]:
        """Get history of a specific metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            List of metric values across epochs.
        """
        return self.history.get(metric_name, [])
