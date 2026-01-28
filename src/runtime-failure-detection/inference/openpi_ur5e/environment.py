"""Environment setup and utilities for RoboSuite with UR5e."""

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
import robosuite as suite  # ty:ignore[unresolved-import]
from openpi_client import image_tools

if TYPE_CHECKING:
    from .config import Config


def make_env(config: "Config", controller_config):
    """Create and configure a RoboSuite environment.

    Args:
        config: Configuration object with environment and camera settings
        controller_config: Controller configuration for the robot

    Returns:
        Configured RoboSuite environment instance
    """
    logging.info(f"Creating robosuite environment: {config} with UR5e robot")
    logging.info(f"Cameras: {config.camera_names}")
    env = suite.make(
        config.task.env_name,
        robots=["UR5e"],
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(config.camera_names),
        camera_heights=config.cam_h,
        camera_widths=config.cam_w,
        control_freq=20,
        horizon=config.horizon,
        reward_shaping=False,
        render_collision_mesh=False,
        render_visual_mesh=False,
    )
    logging.info("Environment created successfully")
    return env


def setup_writers(config: "Config", ts: str) -> tuple[dict, dict]:
    """Create video writers for all configured cameras.

    Args:
        config: Configuration object with camera and output settings
        ts: Timestamp string for unique file naming

    Returns:
        Tuple of (writers dict, temp_paths dict) - temp paths will be renamed with success status later
    """
    import imageio

    writers: dict[str, imageio.core.Format.Writer] = {}
    temp_paths: dict[str, str] = {}
    if config.save_video:
        logging.info(f"Video recording enabled, saving to: {config.task.video_out_dir}")
        for cam in config.camera_names:
            out_path = f"{config.task.video_out_dir}/ur5e_{config.task.env_name}_{cam}_{ts}.mp4"
            temp_paths[cam] = out_path
            writers[cam] = imageio.get_writer(out_path, fps=20)
            logging.info(f"Video writer created for {cam}: {out_path}")
    return writers, temp_paths


def get_images(obs: dict, camera_names: tuple[str, ...], resize: int) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """Extract and preprocess camera images from observation.

    Args:
        obs: Observation dictionary from environment
        camera_names: Tuple of camera names to extract
        resize: Target size for image resizing

    Returns:
        Tuple of (base_cam_name, wrist_cam_name, base_img, wrist_img)
    """
    base_cam = camera_names[0] if camera_names else "agentview"
    base_img = obs.get(f"{base_cam}_image")
    if base_img is None:
        base_cam = "agentview"
        base_img = obs[f"{base_cam}_image"]

    wrist_cam = "robot0_eye_in_hand"
    wrist_img = obs.get(f"{wrist_cam}_image")
    if wrist_img is None:
        wrist_img = np.zeros_like(base_img)

    base_img = np.flipud(base_img)
    base_img = np.ascontiguousarray(base_img)

    wrist_img = np.ascontiguousarray(wrist_img[::-1])

    base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, resize, resize))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize, resize))

    return base_cam, wrist_cam, base_img, wrist_img


def build_state(obs: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract robot joint and gripper state from observation.

    Args:
        obs: Observation dictionary from environment

    Returns:
        Tuple of (joint_positions, gripper_positions)
    """
    sin_vals = obs["robot0_joint_pos_sin"]
    cos_vals = obs["robot0_joint_pos_cos"]
    joint_pos = np.arctan2(sin_vals, cos_vals)

    joint_pos = np.astype(joint_pos, float)
    gripper_pos = obs["robot0_gripper_qpos"]

    logging.debug(
        f"Robot Joint position: [{joint_pos[0]:.4f}, {joint_pos[1]:.4f}, {joint_pos[2]:.4f}, {joint_pos[3]:.4f}, {joint_pos[4]:.4f}, {joint_pos[5]:.4f}]"
    )
    logging.debug(f"Robot gripper position: [{gripper_pos[0]:.4f}, {gripper_pos[1]:.4f}]")

    return joint_pos, gripper_pos
