#!/usr/bin/env python

"""
Check if all indices in a LeRobot dataset contain images and data.
"""

import argparse
import logging
import sys
from pathlib import Path

import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def check_dataset(
    repo_id: str,
    root: Path | None = None,
) -> None:
    """
    Iterate through a LeRobot dataset and check if all indices contain valid data.

    Args:
        repo_id: The ID of the dataset repository (e.g. "lerobot/pusht").
        root: Optional root directory where the dataset is stored.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"Loading dataset {repo_id}...")
    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    logging.info(f"Dataset loaded. Total frames: {len(dataset)}")
    logging.info(f"Features: {list(dataset.features.keys())}")

    # We want to check if we can access every frame and if it contains expected data
    # especially images/videos if present.

    camera_keys = dataset.meta.camera_keys
    logging.info(f"Camera keys to check: {camera_keys}")

    errors = 0

    for i in tqdm.tqdm(range(len(dataset)), desc="Checking frames"):
        try:
            item = dataset[i]

            if not isinstance(item, dict):
                logging.error(f"Index {i}: Returned item is not a dict")
                errors += 1
                continue

            # Check if all camera keys are present and not None
            for key in camera_keys:
                if key not in item:
                    logging.error(f"Index {i}: Missing key '{key}'")
                    errors += 1
                elif item[key] is None:
                    logging.error(f"Index {i}: Value for '{key}' is None")
                    errors += 1

                # For videos/images, LeRobotDataset returns torch tensors (after transform)
                # We can check if they have valid shape/content if needed, but presence is the main check.
                # If video decoding failed, it likely raised an exception caught below.

        except Exception as e:
            logging.error(f"Index {i}: Error accessing frame - {e}")
            errors += 1
            # Optional: stop after too many errors?
            # if errors > 10: break

    if errors == 0:
        logging.info("Success! All frames checked and contain valid data.")
    else:
        logging.error(f"Finished with {errors} errors.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if all indices in a LeRobot dataset contain images and data.")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The ID of the dataset repository (e.g. 'lerobot/pusht').",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root directory where the dataset is stored.",
    )

    args = parser.parse_args()
    check_dataset(args.repo_id, args.root)
