"""Logging configuration for OpenPI UR5e inference."""

import logging
import os
from pathlib import Path

from .config import Config


def setup_logging(config: Config, path: Path) -> None:
    """Configure logging for the inference pipeline.

    Args:
        config: Configuration object containing logging level and output directory
    """

    level = getattr(logging, str(config.logging).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(path, "output.log"),
        force=True,
    )
