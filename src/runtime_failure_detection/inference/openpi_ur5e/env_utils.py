import logging
from typing import Dict, Tuple

import mimicgen  # ty:ignore[unresolved-import] # noqa: F401
import numpy as np
import robosuite as suite  # ty:ignore[unresolved-import]
import robosuite_task_zoo  # ty:ignore[unresolved-import] # noqa: F401
from beartype.typing import Any
from openpi_client import image_tools


def make_env(config, controller_config):
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
        gripper_types="Robotiq85Gripper",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(config.camera_names),
        camera_heights=config.cam_h,
        camera_widths=config.cam_w,
        control_freq=20,
        horizon=config.task.horizon,
        reward_shaping=False,
        render_collision_mesh=False,
        render_visual_mesh=False,
    )
    logging.info("Environment created successfully")
    return env


def process_observation(obs: Dict, camera_names: Tuple[str, ...], resize: int) -> Dict[str, Any]:
    """
    Consolidated function to extract all necessary data from Robosuite observation.
    Returns a clean dictionary used by Agent and Recorder.
    """
    # 1. Process Images
    base_cam = camera_names[0] if camera_names else "agentview"
    base_img = obs.get(f"{base_cam}_image", obs.get("agentview_image"))

    wrist_cam = "robot0_eye_in_hand"
    wrist_img = obs.get(f"{wrist_cam}_image", np.zeros_like(base_img))

    # Image Transforms
    base_img = np.ascontiguousarray(np.flipud(base_img))
    wrist_img = np.ascontiguousarray(wrist_img[::-1])

    base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, resize, resize))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize, resize))

    # 2. Process State
    joint_pos = np.arctan2(obs["robot0_joint_pos_sin"], obs["robot0_joint_pos_cos"]).astype(float)
    gripper_pos = obs["robot0_gripper_qpos"][:1]

    return {
        "base_cam_name": base_cam,
        "wrist_cam_name": wrist_cam,
        "base_img": base_img,
        "wrist_img": wrist_img,
        "joints": joint_pos,
        "gripper": gripper_pos,
    }
