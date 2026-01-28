#!/usr/bin/env python3

"""
LeRobot Dataset Converter
Converts HDF5 episode files to LeRobot dataset format for Hugging Face Hub.

Batch processing is not supported with LeRobot 0.4.2 due to a bug. MR #2462 https://github.com/huggingface/lerobot/pull/2462 fixes this.

Usage:
    # 1. Fully manual (Tyro validates required args):
    python convert_lerobot.py --repo-id user/dataset --input-path ./data/raw

    # 2. Config file (Tyro loads defaults from YAML):
    python convert_lerobot.py --config config.yaml

    # 3. Config file + Overrides (CLI args take precedence):
    python convert_lerobot.py --config config.yaml --input-path ./data/new_batch --push-to-hub
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dacite
import h5py
import numpy as np
import torch
import tyro
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LeRobotConfig:
    """Configuration for LeRobot Dataset Conversion."""

    # Required Fields
    repo_id: str
    """Hugging Face repository ID (e.g., 'username/dataset-name')."""

    input_path: Path
    """Path to input HDF5 file or directory."""

    # Optional Fields with Defaults
    fps: int = 20
    """Frames per second for the dataset."""

    robot_type: Optional[str] = None
    """Type of robot (e.g., 'panda', 'ur5')."""

    push_to_hub: bool = False
    """Whether to push the dataset to the Hub after conversion."""

    private_repo: bool = False
    """Make the repository private when pushing."""

    tags: Optional[List[str]] = None
    """Optional tags for the Hugging Face Hub."""

    output_dir: Path = Path("./data/lerobot_formatted")
    """Directory to save the formatted dataset."""

    batch_size: int = 1
    """Batch size for processing frames."""

    task: str = "unknown"
    """Default task name if mapping is not found."""

    task_mapping: Optional[Union[Dict[str, str], List[str]]] = None
    """Mapping of files to tasks. Can be a dict (filename->task) or list (order-based)."""


class LeRobotDatasetConverter:
    """Converter for transforming HDF5 robot demonstration data to LeRobot format."""

    def __init__(self, config: LeRobotConfig) -> None:
        """Initialize the converter with a validated configuration object."""
        self.cfg = config

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path: Path = self.cfg.output_dir / f"conversion_{timestamp}"

        # Define features (Standard LeRobot structure)
        self.features: dict[str, dict[str, Any]] = {
            "observation.images.camera_base": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.images.camera_wrist_right": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "motors": [
                        "joint0",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "gripper",
                    ]
                },
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "motors": [
                        "joint0",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "gripper",
                    ]
                },
            },
        }

    def get_task_for_file(self, file_path: Path, file_index: int) -> str:
        """Get the task name for a given HDF5 file using the config mapping."""
        mapping = self.cfg.task_mapping

        if mapping is None:
            return self.cfg.task

        # List-based mapping (index matching)
        if isinstance(mapping, list):
            if file_index < len(mapping):
                return mapping[file_index]
            else:
                logger.warning(f"File index {file_index} exceeds task_mapping length. Using default task.")
                return self.cfg.task

        # Dict-based mapping (filename matching)
        if isinstance(mapping, dict):
            filename = file_path.name
            # Exact match
            if filename in mapping:
                return mapping[filename]
            # Pattern match
            for pattern, task_name in mapping.items():
                if pattern in filename:
                    return task_name

            logger.warning(f"No task mapping found for {filename}. Using default task.")
            return self.cfg.task

        return self.cfg.task

    def create_dataset(self) -> LeRobotDataset:
        """Create LeRobot dataset with configured parameters."""
        logger.info(f"Creating dataset in: {self.output_path}")

        return LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            fps=self.cfg.fps,
            features=self.features,
            root=self.output_path,
            robot_type=self.cfg.robot_type,
            use_videos=True,
            image_writer_processes=16,
            image_writer_threads=20,
            batch_encoding_size=self.cfg.batch_size,
        )

    def process_episode(self, hdf5_path: Path, dataset: LeRobotDataset, file_task: str) -> int:
        """Process HDF5 file containing demo_* folders."""
        processed_count = 0
        try:
            with h5py.File(hdf5_path, "r") as hdf5_file:
                if "data" not in hdf5_file:
                    logger.warning(f"No 'data' folder found in {hdf5_path.name}")
                    return 0

                data_group = hdf5_file["data"]
                demo_folders = sorted([key for key in data_group.keys() if key.startswith("demo_")])

                if not demo_folders:
                    logger.warning(f"No demo_* folders found in {hdf5_path.name}")
                    return 0

                for demo_name in demo_folders:
                    try:
                        demo_group: h5py.Group = data_group[demo_name]

                        # Task Priority: File Attr > Group Attr > Config Mapping
                        task: str
                        if "task" in hdf5_file.attrs:
                            task = hdf5_file.attrs["task"]
                        elif "task" in demo_group.attrs:
                            task = demo_group.attrs["task"]
                        else:
                            task = file_task

                        # Extract data
                        actions = demo_group["actions"][:].astype(np.float32)

                        obs_group = demo_group["obs"]
                        gripper_qpos = obs_group["robot0_gripper_qpos"][:].astype(np.float32)
                        joint_pos = obs_group["robot0_joint_pos"][:].astype(np.float32)

                        robot_observations = np.concatenate([joint_pos, gripper_qpos[:, :1]], axis=1).astype(np.float32)

                        # Image extraction
                        # Note: Assuming images are already numpy arrays in correct format
                        agentview_img = obs_group["agentview_image"][:]
                        eye_in_hand_img = obs_group["robot0_eye_in_hand_image"][:]

                        min_length = min(
                            len(actions),
                            len(robot_observations),
                            len(agentview_img),
                            len(eye_in_hand_img),
                        )

                        # Add frames
                        for i in range(min_length):
                            dataset.add_frame(
                                {
                                    "observation.images.camera_base": torch.from_numpy(agentview_img[i]),
                                    "observation.images.camera_wrist_right": torch.from_numpy(eye_in_hand_img[i]),
                                    "observation.state": torch.from_numpy(robot_observations[i]),
                                    "action": torch.from_numpy(actions[i]),
                                    "task": task,
                                }
                            )

                        dataset.save_episode()

                        processed_count += 1
                        logger.info(f"Processed {demo_name} from {hdf5_path.name} ({min_length} frames)")

                    except Exception as e:
                        logger.error(f"Error processing {demo_name} in {hdf5_path.name}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error reading file {hdf5_path.name}: {str(e)}")

        return processed_count

    def run_conversion(self) -> None:
        """Run the complete conversion process."""
        # Find files
        episode_files: List[Path]
        if self.cfg.input_path.is_file():
            if self.cfg.input_path.suffix != ".hdf5":
                raise ValueError(f"Input file must be .hdf5, got {self.cfg.input_path}")
            episode_files = [self.cfg.input_path]
            logger.info(f"Processing single file: {self.cfg.input_path}")
        elif self.cfg.input_path.is_dir():
            episode_files = sorted(self.cfg.input_path.rglob("*.hdf5"), key=lambda x: x.name)
            logger.info(f"Found {len(episode_files)} HDF5 files in {self.cfg.input_path}")
        else:
            raise FileNotFoundError(f"Input path does not exist: {self.cfg.input_path}")

        if not episode_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.cfg.input_path}")

        dataset = self.create_dataset()
        total_demos = 0

        for file_idx, hdf5_file in enumerate(tqdm(episode_files, desc="Processing HDF5 files")):
            file_task = self.get_task_for_file(hdf5_file, file_idx)
            logger.info(f"Processing {hdf5_file.name} with task: {file_task}")

            demos_processed = self.process_episode(hdf5_file, dataset, file_task)
            total_demos += demos_processed

        dataset.finalize()
        logger.info(f"Successfully processed {total_demos} demos from {len(episode_files)} files")

        if self.cfg.push_to_hub:
            logger.info(f"Pushing dataset to Hugging Face Hub at {self.cfg.repo_id}...")
            try:
                dataset.push_to_hub(
                    private=self.cfg.private_repo,
                    tags=self.cfg.tags,
                )
                logger.info(f"Successfully pushed dataset to {self.cfg.repo_id}")
            except Exception as e:
                logger.error(f"Failed to push dataset to hub: {str(e)}")
                raise


def load_yaml_defaults(yaml_path: str) -> dict[str, Any]:
    """Load YAML file to be used as default values for Tyro."""
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError("YAML content must be a dictionary.")
            # Convert string path in YAML to Path object for consistency
            if "input_path" in data:
                data["input_path"] = Path(data["input_path"])
            if "output_dir" in data:
                data["output_dir"] = Path(data["output_dir"])
            return data
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)


def main() -> None:
    # 1. Handle --config manually before Tyro sees it
    defaults: LeRobotConfig | None = None
    # Create a copy of args to modify for Tyro
    tyro_args = sys.argv[1:]

    if "--config" in sys.argv:
        try:
            # Find the index of --config
            config_idx = sys.argv.index("--config")

            # Ensure the value exists
            if config_idx + 1 >= len(sys.argv):
                raise ValueError("--config requires a file path argument")

            config_path = sys.argv[config_idx + 1]
            defaults = dacite.from_dict(LeRobotConfig, load_yaml_defaults(config_path))

            # Remove --config and its argument from the list passed to Tyro
            # We filter out exactly these two items
            tyro_args = [arg for i, arg in enumerate(sys.argv[1:]) if i != config_idx - 1 and i != config_idx]

        except ValueError as e:
            logger.error(f"Config Error: {e}")
            sys.exit(1)

    # 2. Parse arguments using Tyro with the cleaned argument list
    try:
        # We explicitly pass `args=tyro_args` so Tyro doesn't see '--config'
        config: LeRobotConfig = tyro.cli(LeRobotConfig, default=defaults, args=tyro_args)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Configuration Validation Error: {e}")
        # Helpful hint if validation failed
        logger.error("Tip: Check if your config.yaml contains 'repo_id' and 'input_path'")
        sys.exit(1)

    # 3. Execute
    try:
        converter = LeRobotDatasetConverter(config)
        converter.run_conversion()
    except Exception as e:
        logger.error(f"Fatal error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
