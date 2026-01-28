"""Policy loading and inference utilities."""

import logging
import os
from typing import TYPE_CHECKING

import openpi.shared.download as download
from openpi.policies import policy_config
from openpi.training import config as _config

if TYPE_CHECKING:
    from .config import Config


def load_policy_from_config(config: "Config"):
    """Load an OpenPI policy from configuration.

    Args:
        config: Configuration object with policy and checkpoint settings

    Returns:
        Loaded policy instance ready for inference
    """
    logging.info(f"Loading policy configuration: {config.policy_config}")
    policy_cfg = _config.get_config(config.policy_config)

    if config.checkpoint_dir is None:
        logging.info("No checkpoint directory specified, using default checkpoint path")
        print(f"Current working directory: {os.getcwd()}")
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    else:
        checkpoint_dir = config.checkpoint_dir

    logging.info(f"Loading policy from checkpoint: {checkpoint_dir}")
    policy = policy_config.create_trained_policy(policy_cfg, checkpoint_dir)
    logging.info("Policy loaded successfully")
    return policy
