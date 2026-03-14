import datetime
import logging
import pathlib
import random
from pathlib import Path

import hydra
import numpy as np
from robosuite.controllers import load_controller_config  # ty:ignore[unresolved-import]

from runtime_failure_detection.inference.openpi_ur5e.agent import OpenPIAgent
from runtime_failure_detection.inference.openpi_ur5e.config import Config
from runtime_failure_detection.inference.openpi_ur5e.env_utils import make_env
from runtime_failure_detection.inference.openpi_ur5e.logger import setup_logging
from runtime_failure_detection.inference.openpi_ur5e.recorder import RolloutRecorder
from runtime_failure_detection.inference.openpi_ur5e.runner import run_episode


@hydra.main(version_base="1.1", config_path="../../../configs/inference", config_name="config")
def main(cfg: Config) -> None:
    base_folder: Path = "rollouts" / pathlib.Path(cfg.save_name)
    if cfg.hierarchical_structure:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rollout_folder: Path = base_folder / pathlib.Path(cfg.task.video_out_dir) / ts
    else:
        rollout_folder: Path = base_folder

    rollout_folder.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg, rollout_folder)

    # Set random seeds for reproducibility
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        logging.info(f"Random seed set to {cfg.seed}")

    # 2. Initialize Agent (Policy loads once)
    agent = OpenPIAgent(cfg, record_dir=str(rollout_folder / "policy_records"))

    # 3. Initialize Controller & Env
    controller_config = load_controller_config(default_controller="OSC_POSE")  # TODO: Hammer & Kitchen use OCS_POSITION

    results = []

    # 4. Trial Loop
    for trial_idx in range(cfg.num_trials):
        env = make_env(cfg, controller_config)
        recorder = RolloutRecorder(cfg, str(rollout_folder / "env_records"), trial_idx)

        try:
            # Run the episode
            stats = run_episode(env, agent, recorder, trial_idx)
            success = stats["success"]
            results.append(success)

            # Save data
            recorder.close(success=success, metadata=stats)

        finally:
            env.close()

    # 5. Summary
    logging.info(f"Success Rate: {sum(results)}/{len(results)} ({sum(results) / len(results) * 100:.1f}%)")


if __name__ == "__main__":
    main()
