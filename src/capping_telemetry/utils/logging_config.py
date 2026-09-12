"""Centralized logging setup, called once from the notebook if desired."""
from __future__ import annotations

import logging
import sys

from capping_telemetry.config import Settings


def configure_logging(settings: Settings) -> None:
    log_dir = settings.resolve(settings.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "capping_telemetry.log"

    level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
