import logging

import numpy as np

from .agent import OpenPIAgent
from .env_utils import process_observation
from .recorder import RolloutRecorder


def _get_dummy_action(env) -> np.ndarray:
    if hasattr(env, "action_dim"):
        return np.zeros(env.action_dim, dtype=float)
    if hasattr(env, "action_spec"):
        low, _high = env.action_spec
        return np.zeros_like(low, dtype=float)
    raise AttributeError("Environment does not expose action_dim or action_spec for dummy action.")


def run_episode(env, agent: OpenPIAgent, recorder: RolloutRecorder, trial_idx: int) -> dict:
    """Executes a single episode."""

    agent.reset()
    obs = env.reset()
    t = 0
    done = False
    step_done = []  # Track done status at each step
    model_infer_times = 0  # Track number of model inferences
    dummy_action = _get_dummy_action(env)

    logging.info(f"Starting Trial {trial_idx}")

    try:
        # Run until horizon if same_length is True, otherwise stop early on success
        def should_continue():
            if agent.cfg.same_length:
                return t < agent.cfg.task.horizon
            return t < agent.cfg.task.horizon and not done

        while should_continue():
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < agent.cfg.num_steps_wait:
                obs, reward, done, info = env.step(dummy_action.tolist())
                step_done.append(done)
                t += 1
                continue
            # Process Observation (No circular import now)
            processed_obs = process_observation(obs, agent.cfg.camera_names, agent.cfg.resize_size)

            # Record Video
            recorder.record_frame(processed_obs["base_cam_name"], processed_obs["base_img"])
            recorder.record_frame(processed_obs["wrist_cam_name"], processed_obs["wrist_img"])

            # Get Action from Agent
            action = agent.step(processed_obs, trial_idx, t)
            model_infer_times += 1

            # Step Environment
            obs, reward, done, info = env.step(action.tolist())
            step_done.append(done)

            if done:
                logging.info(f"Success at step {t}")
            t += 1

    except Exception as e:
        logging.error(f"Error during rollout: {e}")
        raise e

    # Return stats for metadata
    return {
        "steps": t,
        "success": done or any(step_done),
        "step_done": step_done,
        "model_infer_times": model_infer_times,
        "horizon": agent.cfg.task.horizon,
        "seed": agent.cfg.seed,
    }
