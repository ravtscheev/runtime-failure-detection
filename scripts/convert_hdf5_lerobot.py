#!/usr/bin/env python3
"""
LeRobot Dataset Converter
Converts HDF5 episode files to LeRobot dataset format for Hugging Face Hub.

Usage:
    python convert_lerobot.py --config config.yaml [--input FILE_OR_DIR]
    
    Examples:
        # Convert a single file
        python convert_lerobot.py --config config.yaml --input /path/to/episode_0.hdf5
        
        # Convert all HDF5 files in a directory (recursively)
        python convert_lerobot.py --config config.yaml --input /path/to/data
        
        # Default: Use all HDF5 files from ./mimicgen_generated (recursively)
        python convert_lerobot.py --config config.yaml
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import tyro
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeRobotDatasetConverter:
    """Converter for transforming HDF5 robot demonstration data to LeRobot format."""
    
    def __init__(self, config: dict[str, Any], input_path: str | Path | None = None) -> None:
        """Initialize the converter with configuration.
        
        Args:
            config: Configuration dictionary containing repo_id, fps, robot_type, etc.
            input_path: Path to input HDF5 file or directory. Defaults to ../data/mimicgen_generated.
        """
        self.config: dict[str, Any] = config
        self.repo_id: str = config["repo_id"]
        self.fps: int = config.get("fps", 20)
        self.robot_type: str | None = config.get("robot_type", None)
        
        # Determine input path
        self.input_path: Path = Path(input_path) if input_path else Path("../data/mimicgen_generated")
        
        # Create timestamped output directory to avoid overriding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(config.get("output_dir", "./data/lerobot_formatted"))
        self.output_dir: Path = base_output_dir / f"conversion_{timestamp}"
        self.batch_size: int = config.get("batch_size", 1)
        self.task: str = config.get("task", "unknown")
        
        # Task mapping: maps file patterns or indices to task names
        # Format can be:
        # - dict mapping filename patterns to tasks: {"file1.hdf5": "task1", "file2.hdf5": "task2"}
        # - list of tasks applied in order to sorted files: ["task1", "task2", "task3"]
        self.task_mapping: dict[str, str] | list[str] | None = config.get("task_mapping", None)
        
        self.features: dict[str, dict[str, Any]] = {
            "observation.images.camera_base": {
                "dtype": "video",
                "shape": (84, 84, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.images.camera_wrist_right": {
                "dtype": "video",
                "shape": (84, 84, 3),
                "names": ["height", "width", "rgb"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]},
            },
        }

    def get_task_for_file(self, file_path: Path, file_index: int) -> str:
        """Get the task name for a given HDF5 file.
        
        Args:
            file_path: Path to the HDF5 file.
            file_index: Index of the file in the sorted list of files.
            
        Returns:
            Task name for this file.
        """
        if self.task_mapping is None:
            return self.task
        
        # If task_mapping is a list, use index
        if isinstance(self.task_mapping, list):
            if file_index < len(self.task_mapping):
                return self.task_mapping[file_index]
            else:
                logger.warning(f"File index {file_index} exceeds task_mapping list length. Using default task.")
                return self.task
        
        # If task_mapping is a dict, try to match filename
        if isinstance(self.task_mapping, dict):
            filename = file_path.name
            # Try exact match first
            if filename in self.task_mapping:
                return self.task_mapping[filename]
            # Try pattern matching
            for pattern, task in self.task_mapping.items():
                if pattern in filename:
                    return task
            logger.warning(f"No task mapping found for {filename}. Using default task.")
            return self.task
        
        return self.task

    def _validate_batch_size(self, num_episodes: int) -> None:
        """Check if episode count is divisible by batch size.
        
        Args:
            num_episodes: Total number of episodes to validate.
            
        Raises:
            ValueError: If episodes cannot be evenly divided by batch size.
        """
        if num_episodes % self.batch_size != 0:
            raise ValueError(
                f"Cannot divide {num_episodes} episodes into batches of {self.batch_size}. "
                f"Please adjust batch_size in config or number of episodes."
            )
        logger.info(f"Using batch size: {self.batch_size}")

    def create_dataset(self) -> LeRobotDataset:
        """Create LeRobot dataset with configured parameters.
        
        Returns:
            Initialized LeRobotDataset ready for adding frames.
        """
        logger.info(f"Creating dataset in: {self.output_dir}")
            
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self.features,
            root=self.output_dir,
            robot_type=self.robot_type,
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=16,
            batch_encoding_size=self.batch_size,
        )

    def process_episode(self, hdf5_path: Path, dataset: LeRobotDataset, file_task: str) -> int:
        """Process HDF5 file containing demo_* folders.
        
        Args:
            hdf5_path: Path to the HDF5 file to process.
            dataset: LeRobotDataset instance to add frames to.
            file_task: Task name to use for this file.
            
        Returns:
            Number of successfully processed demos.
        """
        processed_count = 0
        try:
            with h5py.File(hdf5_path, "r") as hdf5_file:
                # Check if data folder exists
                if "data" not in hdf5_file:
                    logger.warning(f"No 'data' folder found in {hdf5_path.name}")
                    return 0
                
                data_group = hdf5_file["data"]
                
                # Find all demo_* folders
                demo_folders = sorted([key for key in data_group.keys() if key.startswith("demo_")])
                
                if not demo_folders:
                    logger.warning(f"No demo_* folders found in {hdf5_path.name}")
                    return 0
                
                # Process each demo
                for demo_name in demo_folders:
                    try:
                        demo_group: h5py.Group = data_group[demo_name]
                        
                        # Extract task from attributes (try different possible locations)
                        task: str
                        if "task" in hdf5_file.attrs:
                            task = hdf5_file.attrs["task"]
                        elif "task" in demo_group.attrs:
                            task = demo_group.attrs["task"]
                        else:
                            # Use the task assigned to this file
                            task = file_task
                        
                        # Extract actions and states
                        actions: np.ndarray = demo_group["actions"][:].astype(np.float32)
                        states: np.ndarray = demo_group["states"][:].astype(np.float32)
                        
                        # Extract observations from obs folder
                        obs_group: h5py.Group = demo_group["obs"]
                        
                        # Get specific robot observations: gripper_qpos and joint_pos
                        gripper_qpos: np.ndarray = obs_group["robot0_gripper_qpos"][:].astype(np.float32)
                        joint_pos: np.ndarray = obs_group["robot0_joint_pos"][:].astype(np.float32)
                        robot_observations: np.ndarray = np.concatenate(
                            [joint_pos, gripper_qpos[:, :1]], axis=1
                        ).astype(np.float32)
                        
                        # Extract images
                        image_data: dict[str, np.ndarray] = {
                            "agentview": obs_group["agentview_image"][:],
                            "eye_in_hand": obs_group["robot0_eye_in_hand_image"][:]
                        }
                        
                        # Decode images (currently bypassed - images are already in numpy format)
                        # TODO: Uncomment if images need BGR to RGB conversion
                        # decoded_images: dict[str, list[np.ndarray]] = {}
                        # for img_key, img_data in image_data.items():
                        #     decoded_images[img_key] = [
                        #         cv2.cvtColor(img_bytes, cv2.COLOR_BGR2RGB)
                        #         for img_bytes in img_data
                        #     ]
                        decoded_images: dict[str, np.ndarray] = image_data
                        
                        # Determine minimum length across all sequences
                        min_length: int = min(
                            len(actions),
                            len(robot_observations),
                            len(decoded_images["agentview"]),
                            len(decoded_images["eye_in_hand"])
                        )
                        
                        # Add frames to dataset
                        for i in range(min_length):
                            dataset.add_frame(
                                {
                                    "observation.images.camera_base": decoded_images["agentview"][i],
                                    "observation.images.camera_wrist_right": decoded_images["eye_in_hand"][i],
                                    "observation.state": robot_observations[i],
                                    "action": actions[i],
                                },
                                task=task
                            )
                            

                        dataset.save_episode()
                        processed_count += 1
                        logger.info(f"Processed {demo_name} from {hdf5_path.name} ({min_length} frames)")
                        
                    except Exception as e:
                        logger.error(f"Error processing {demo_name} in {hdf5_path.name}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error processing {hdf5_path.name}: {str(e)}")
            logger.error(f"Error processing {hdf5_path.name}: {str(e)}")
        
        return processed_count

    def run_conversion(self) -> None:
        """Run the complete conversion process.
        
        Finds all HDF5 files, creates the dataset, and processes each episode.
        """
        # Find HDF5 files based on input path type
        episode_files: list[Path]
        if self.input_path.is_file():
            # Single file case
            if not self.input_path.suffix == ".hdf5":
                raise ValueError(f"Input file must be an HDF5 file (.hdf5), got {self.input_path}")
            episode_files = [self.input_path]
            logger.info(f"Processing single file: {self.input_path}")
        elif self.input_path.is_dir():
            # Directory case: recursively find all HDF5 files
            episode_files = sorted(
                self.input_path.rglob("*.hdf5"),
                key=lambda x: x.name
            )
            logger.info(f"Found {len(episode_files)} HDF5 files in {self.input_path}")
        else:
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        if not episode_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.input_path}")

        dataset: LeRobotDataset = self.create_dataset()
        total_demos: int = 0

        for file_idx, hdf5_file in enumerate(tqdm(episode_files, desc="Processing HDF5 files")):
            file_task = self.get_task_for_file(hdf5_file, file_idx)
            logger.info(f"Processing {hdf5_file.name} with task: {file_task}")
            demos_processed = self.process_episode(hdf5_file, dataset, file_task)
            total_demos += demos_processed

        logger.info(f"Successfully processed {total_demos} demos from {len(episode_files)} HDF5 files")

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class Args:
    """Command-line arguments for HDF5 to LeRobot dataset conversion."""
    
    config: str = "./configs/lerobot_convert.yaml"
    """Path to configuration YAML file."""
    
    input: str = "./data/mimicgen_generated"
    """Input file or directory. If a file: convert single HDF5 file. If a directory: recursively convert all HDF5 files in it."""


def main() -> None:
    """Main entry point for the conversion script."""
    args = tyro.cli(Args)
    
    try:
        config = load_config(args.config) 
        converter = LeRobotDatasetConverter(config, input_path=args.input)
        converter.run_conversion()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()