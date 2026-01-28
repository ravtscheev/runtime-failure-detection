"""Logging configuration for OpenPI UR5e inference."""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


def setup_logging(config: "Config") -> None:
    """Configure logging for the inference pipeline.

    Args:
        config: Configuration object containing logging level and output directory
    """
    level = getattr(logging, str(config.logging).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(config.task.video_out_dir, "ur5e_robosuite_openpi.log"),
        force=True,
    )
