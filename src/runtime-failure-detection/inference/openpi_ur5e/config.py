"""Configuration schemas for OpenPI UR5e inference."""

import dataclasses

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclasses.dataclass
class TaskConfig:
    """Schema for task-specific configuration.

    Attributes:
        env_name: Name of the robosuite environment (e.g., 'Kitchen_D1', 'HammerCleanup_D1')
        prompt: Natural language task description for the policy
        video_out_dir: Directory where task videos will be saved
        seed: Random seed for reproducibility (optional)
    """

    env_name: str = MISSING
    prompt: str = MISSING
    video_out_dir: str = MISSING


@dataclasses.dataclass
class Config:
    """Full configuration schema for OpenPI UR5e inference.

    Attributes:
        policy_config: OpenPI policy configuration name
        checkpoint_dir: Path to the policy checkpoint
        resize_size: Image resize dimension for policy input
        replan_steps: Number of steps to plan ahead
        env_name: Robosuite environment name
        horizon: Episode length
        prompt: Task description
        camera_names: List of camera names to use
        cam_w: Camera width
        cam_h: Camera height
        save_video: Whether to save rollout videos
        video_out_dir: Output directory for videos
        seed: Random seed
        logging: Logging level
    """

    # OpenPI policy configuration
    policy_config: str = "pi0_ur5_merged"
    checkpoint_dir: str | None = "../openpi-failure-detection/checkpoints/pi0_ur5_merged/merged-training/29999"

    # Image processing
    resize_size: int = 896
    replan_steps: int = 16

    # Robosuite task
    env_name: str = MISSING
    horizon: int = 700
    prompt: str = MISSING

    # Cameras
    camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    cam_w: int = 896
    cam_h: int = 896

    # Output video
    save_video: bool = True

    # Trials
    num_trials_per_task: int = 1

    # System
    seed: int | None = 3
    logging: str = "INFO"

    task: TaskConfig = MISSING


# Register the config schemas
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="task", name="base_task", node=TaskConfig)
