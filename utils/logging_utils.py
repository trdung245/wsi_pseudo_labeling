"""
Logging and TensorBoard utilities.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


def setup_logging(log_dir: str | Path, name: str = "train") -> logging.Logger:
    """Configure root logger to write to stdout and a file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger(name)


class MetricLogger:
    """Thin wrapper around TensorBoard SummaryWriter with dict logging."""

    def __init__(self, log_dir: str | Path) -> None:
        self._writer: Any = None
        if _TB_AVAILABLE:
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        if self._writer is None:
            return
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self._writer.add_scalar(tag, v, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
