"""Rollout execution for OpenPI policy inference."""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config


def run_rollout(env, policy, config: "Config", writers: dict) -> bool:
    """Execute a single rollout episode with the policy.

    Args:
        env: RoboSuite environment instance
        policy: OpenPI policy instance
        config: Configuration object with horizon and replan settings
        writers: Dictionary of video writers for recording

    Returns:
        bool: True if the task was successful (done=True), False otherwise
    """
    logging.info("Starting rollout")
    action_plan: list[np.ndarray] = []
    t = 0
    done = False
    obs = env.reset()
    logging.info("Environment reset complete")

    try:
        while t < config.horizon and not done:
            logging.debug(f"Step {t}/{config.horizon}")

            # Import here to avoid circular imports
            from .environment import build_state, get_images

            base_cam, wrist_cam, base_img, wrist_img = get_images(obs, config.camera_names, config.resize_size)

            if writers:
                writers.get(base_cam, None) and writers[base_cam].append_data(base_img)
                writers.get(wrist_cam, None) and writers[wrist_cam].append_data(wrist_img)

            joints, gripper = build_state(obs)

            if not action_plan:
                logging.debug("Action plan empty, querying policy for new actions")
                element = {
                    "observation/joints": joints,
                    "observation/gripper_position": gripper[:1],
                    "observation/base_rgb": base_img,
                    "observation/wrist_rgb": wrist_img,
                    "prompt": config.task.prompt,
                }

                # Infer actions from policy via OpenPI client
                result = policy.infer(element)
                actions = result["actions"]
                logging.debug(f"Policy returned {len(actions)} actions")
                assert (
                    len(actions) >= config.replan_steps
                ), f"Policy predicted {len(actions)} steps, expected at least {config.replan_steps}"

                action_plan.extend(actions[: config.replan_steps])
                logging.debug(f"Added {config.replan_steps} actions to plan")

            # Pop next action
            action = np.asarray(action_plan.pop(0))
            obs, reward, done, info = env.step(action.tolist())

            logging.debug(f"Reward: {reward:.4f}, Done: {done}")
            if done:
                logging.info(f"Task completed successfully at step {t}")
            t += 1

    finally:
        logging.info(f"Rollout finished after {t} steps")

    return done
