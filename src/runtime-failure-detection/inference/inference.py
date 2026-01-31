"""OpenPI UR5e Robosuite Inference - Main Entry Point.

Runs OpenPI policy inference on UR5e robots in RoboSuite environments with video recording.

Usage:
    python interference_openpi_robosuite_ur5e.py                           # Default task
    python interference_openpi_robosuite_ur5e.py task=hammer_cleanup_D1   # Specific task
    python interference_openpi_robosuite_ur5e.py seed=42 logging=DEBUG    # With overrides
"""

import datetime
import logging
import pathlib
import pickle

import hydra
import mimicgen  # noqa: F401  # ty:ignore[unresolved-import]
import numpy as np
import robosuite_task_zoo  # ty:ignore[unresolved-import]  # noqa: F401
from openpi_ur5e.config import Config
from openpi_ur5e.environment import make_env, setup_writers
from openpi_ur5e.logger import setup_logging
from openpi_ur5e.policy import load_policy_from_config
from openpi_ur5e.rollout import run_rollout
from robosuite.controllers import load_controller_config  # ty:ignore[unresolved-import]
from tqdm import tqdm


@hydra.main(version_base="1.1", config_path="../../../configs/inference", config_name="config")
def main(cfg: Config) -> None:
    """Main entry point for OpenPI UR5e inference.

    Args:
        cfg: Hydra configuration dictionary
    """

    # Setup
    pathlib.Path(cfg.task.video_out_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(cfg)

    logging.info("Starting UR5e robosuite + OpenPI integration")
    logging.info(f"Running {cfg.num_trials_per_task} trial(s) for task: {cfg.task.env_name}")

    np.random.seed(cfg.seed)

    logging.info("Loading BASIC composite controller configuration")
    controller_config = load_controller_config(default_controller="OSC_POSITION")  # HAMMER NEEDS OSC_POSITION

    # Create timestamped rollout folder
    ts: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rollout_folder = pathlib.Path(cfg.task.video_out_dir) / ts
    rollout_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created rollout folder: {rollout_folder}")

    # Load policy once (reuse for all trials)
    # Enable PolicyRecorder to save policy records to disk
    policy = load_policy_from_config(cfg, enable_recording=True, record_dir=str(rollout_folder / "policy_records"))

    # Track overall results
    results = []

    # Run multiple trials
    for trial_idx in tqdm(range(cfg.num_trials_per_task), desc="Running trials", unit="trial"):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Starting Trial {trial_idx + 1}/{cfg.num_trials_per_task}")
        logging.info(f"{'=' * 60}")

        # Create environment for this trial
        env = make_env(cfg, controller_config)
        writers, temp_paths = setup_writers(cfg, str(rollout_folder))

        success = False
        rollout_info = {}
        try:
            rollout_info = run_rollout(env, policy, cfg, writers, trial_idx=trial_idx)
            success = rollout_info.get("success", False)
            results.append(success)
        finally:
            if writers:
                import os

                logging.info("Closing video writers")
                for cam, w in writers.items():
                    w.close()
                    logging.info(f"Video saved for {cam}")

                # Rename files to include trial number and success status
                if temp_paths:
                    for cam, temp_path in temp_paths.items():
                        if os.path.exists(temp_path):
                            # Extract base info and add trial + success status
                            base_path = temp_path.rsplit(".", 1)[0]  # Remove .mp4
                            new_path = f"{base_path}--trial{trial_idx}--succ{int(success)}.mp4"
                            os.rename(temp_path, new_path)
                            logging.info(f"Renamed {temp_path} -> {new_path}")

                    # Save metadata pickle file
                    first_cam_path = list(temp_paths.values())[0]
                    base_meta_path = first_cam_path.rsplit(".", 1)[0]  # Remove .mp4
                    meta_save_path = f"{base_meta_path}--trial{trial_idx}--succ{int(success)}.pkl"

                    meta_dict = {
                        "task_env_name": cfg.task.env_name,
                        "task_prompt": cfg.task.prompt,
                        "trial_idx": trial_idx,
                        "episode_success": success,
                        "steps": rollout_info.get("steps", 0),
                        "horizon": cfg.horizon,
                        "replan_steps": cfg.replan_steps,
                        "resize_size": cfg.resize_size,
                        "policy_config": cfg.policy_config,
                        "checkpoint_dir": cfg.checkpoint_dir,
                        "seed": cfg.seed,
                        "camera_names": list(cfg.camera_names),
                        "video_paths": {cam: temp_paths[cam] for cam in temp_paths},
                    }
                    pickle.dump(meta_dict, open(meta_save_path, "wb"))
                    logging.info(f"Saved metadata at {meta_save_path}")

            env.close()
            logging.info(f"Trial {trial_idx + 1} completed - Success: {success}")

    # Summary statistics
    logging.info(f"\n{'=' * 60}")
    logging.info("All trials completed!")
    logging.info(f"Total trials: {len(results)}")
    logging.info(f"Successful: {sum(results)}")
    logging.info(f"Failed: {len(results) - sum(results)}")
    logging.info(f"Success rate: {sum(results) / len(results) * 100:.1f}%")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
