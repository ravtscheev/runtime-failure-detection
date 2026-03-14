import logging
from typing import Any, Dict

import numpy as np

from .config import Config
from .policy import load_policy_from_config


class OpenPIAgent:
    def __init__(self, cfg: Config, record_dir: str | None = None):
        self.cfg = cfg
        # Load policy (using your existing logic)
        self.policy = load_policy_from_config(cfg, enable_recording=True, record_dir=record_dir)
        self.action_queue = []

    def step(self, obs_data: Dict[str, Any], trial_idx: int, timestep: int) -> np.ndarray:
        """Returns a single action, handling replanning internally."""

        # If queue is empty, query the policy
        if not self.action_queue:
            logging.debug("Action queue empty, querying policy...")

            # Construct input element expected by OpenPI
            element = {
                "observation/joints": obs_data["joints"],
                "observation/gripper_position": obs_data["gripper"],
                "observation/base_rgb": obs_data["base_img"],
                "observation/wrist_rgb": obs_data["wrist_img"],
                "prompt": self.cfg.task.prompt,
                "run/run_note": self.cfg.save_name,
                "run/task_id": self.cfg.task.env_name,
                "run/episode_idx": trial_idx,
                "run/timestep": timestep,
            }

            result = self.policy.infer(element)
            actions = result["actions"]

            # Validate plan length
            if len(actions) < self.cfg.replan_steps:
                logging.warning(f"Policy predicted {len(actions)} steps, expected {self.cfg.replan_steps}")

            self.action_queue.extend(actions[: self.cfg.replan_steps])

        return np.asarray(self.action_queue.pop(0))

    def reset(self):
        self.action_queue = []
