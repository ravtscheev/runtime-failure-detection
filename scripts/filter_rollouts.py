#!/usr/bin/env python

"""
Script to filter rollouts to a target number of episodes while keeping all successful runs.

Example usage:
    python scripts/filter_rollouts.py \
        --input-dir rollouts/pi0-ur5 \
        --output-dir rollouts_filtered/pi0-ur5 \
        --target-episodes 50 \
        --seed 42
"""

import pickle
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro

# Regex patterns for parsing filenames
ENV_RECORD_RE = re.compile(r"^(?P<prefix>.+)--trial(?P<trial>\d+)--succ(?P<succ>[01])(?P<suffix>\..+)$")
POLICY_RECORD_RE = re.compile(
    r"^step_(?P<step>\d+)--(?P<policy>.+)--task_(?P<task>.+)--ep_(?P<ep>\d+)--t_(?P<t>\d+)--meta\.pkl$"
)


@dataclass
class Config:
    """Filter rollouts to target number of episodes while preserving all successes."""

    input_dir: Path
    """Input rollouts directory containing env_records and policy_records subdirectories."""

    output_dir: Path
    """Output directory for filtered rollouts."""

    target_episodes: int = 50
    """Target number of episodes in the filtered dataset."""

    seed: int = 42
    """Random seed for reproducibility when selecting failed trials."""

    dry_run: bool = False
    """Print what would be done without actually copying files."""

    verbose: bool = False
    """Print detailed progress information."""


def parse_env_record_filename(filename: str) -> dict | None:
    """Parse env_record filename to extract trial number and success flag."""
    match = ENV_RECORD_RE.match(filename)
    if not match:
        return None
    return {
        "prefix": match.group("prefix"),
        "trial": int(match.group("trial")),
        "succ": int(match.group("succ")),
        "suffix": match.group("suffix"),
    }


def parse_policy_record_filename(filename: str) -> dict | None:
    """Parse policy_record filename to extract step, episode, and timestep."""
    match = POLICY_RECORD_RE.match(filename)
    if not match:
        return None
    return {
        "step": int(match.group("step")),
        "policy": match.group("policy"),
        "task": match.group("task"),
        "ep": int(match.group("ep")),
        "t": int(match.group("t")),
    }


def build_env_record_filename(prefix: str, trial: int, succ: int, suffix: str) -> str:
    """Build env_record filename from components."""
    return f"{prefix}--trial{trial}--succ{succ}{suffix}"


def build_policy_record_filename(step: int, policy: str, task: str, ep: int, t: int) -> str:
    """Build policy_record filename from components."""
    return f"step_{step}--{policy}--task_{task}--ep_{ep}--t_{t}--meta.pkl"


def main(cfg: Config):
    input_dir = cfg.input_dir
    output_dir = cfg.output_dir
    target_episodes = cfg.target_episodes
    seed = cfg.seed

    # Validate input directory
    env_records_dir = input_dir / "env_records"
    policy_records_dir = input_dir / "policy_records"

    if not env_records_dir.exists():
        raise FileNotFoundError(f"env_records directory not found: {env_records_dir}")
    if not policy_records_dir.exists():
        raise FileNotFoundError(f"policy_records directory not found: {policy_records_dir}")

    # Step 1: Collect all trials and their success status from env_records
    print("Step 1: Scanning env_records...")

    # Group files by trial number
    trials: dict[int, dict] = {}  # trial -> {succ, files: [path, ...]}

    for path in env_records_dir.iterdir():
        if not path.is_file():
            continue

        parsed = parse_env_record_filename(path.name)
        if not parsed:
            if cfg.verbose:
                print(f"  [SKIP] Unrecognized filename: {path.name}")
            continue

        trial = parsed["trial"]
        succ = parsed["succ"]

        if trial not in trials:
            trials[trial] = {"succ": succ, "files": []}

        # Verify consistency (all files for same trial should have same succ)
        if trials[trial]["succ"] != succ:
            print(f"  [WARN] Inconsistent succ for trial {trial}: {path.name}")

        trials[trial]["files"].append(path)

    # Separate successful and failed trials
    successful_trials = [t for t, data in trials.items() if data["succ"] == 1]
    failed_trials = [t for t, data in trials.items() if data["succ"] == 0]

    print(f"  Found {len(trials)} total trials")
    print(f"  Successful: {len(successful_trials)}")
    print(f"  Failed: {len(failed_trials)}")

    # Step 2: Select trials to keep
    print("\nStep 2: Selecting trials to keep...")

    if len(successful_trials) > target_episodes:
        print(f"  [WARN] More successful trials ({len(successful_trials)}) than target ({target_episodes})")
        print("  Keeping all successful trials anyway.")
        selected_trials = successful_trials.copy()
    else:
        # Keep all successful, fill rest with random failed
        num_failed_to_select = target_episodes - len(successful_trials)

        if num_failed_to_select > len(failed_trials):
            print(
                f"  [WARN] Not enough failed trials to reach target. "
                f"Need {num_failed_to_select}, have {len(failed_trials)}"
            )
            num_failed_to_select = len(failed_trials)

        random.seed(seed)
        selected_failed = random.sample(failed_trials, num_failed_to_select)
        selected_trials = sorted(successful_trials + selected_failed)

    print(f"  Selected {len(selected_trials)} trials")
    selected_successes = sum(1 for t in selected_trials if trials[t]["succ"] == 1)
    print(f"    - Successful: {selected_successes}")
    print(f"    - Failed: {len(selected_trials) - selected_successes}")

    # Create mapping from old trial number to new episode index
    trial_to_new_ep = {old_trial: new_ep for new_ep, old_trial in enumerate(selected_trials)}

    # Step 3: Collect policy records for selected trials
    print("\nStep 3: Scanning policy_records...")

    # Group policy records by episode
    policy_by_ep: dict[int, list[Path]] = defaultdict(list)

    for path in policy_records_dir.iterdir():
        if not path.is_file():
            continue

        parsed = parse_policy_record_filename(path.name)
        if not parsed:
            if cfg.verbose:
                print(f"  [SKIP] Unrecognized filename: {path.name}")
            continue

        policy_by_ep[parsed["ep"]].append(path)

    # Verify we have policy records for all selected trials
    missing_policy = []
    for trial in selected_trials:
        if trial not in policy_by_ep:
            missing_policy.append(trial)

    if missing_policy:
        print(f"  [WARN] Missing policy records for trials: {missing_policy[:10]}...")

    # Step 4: Copy files with renumbering
    print("\nStep 4: Copying and renumbering files...")

    out_env_dir = output_dir / "env_records"
    out_policy_dir = output_dir / "policy_records"

    if not cfg.dry_run:
        out_env_dir.mkdir(parents=True, exist_ok=True)
        out_policy_dir.mkdir(parents=True, exist_ok=True)

    # Track new global step counter for policy records
    new_step = 0

    env_copied = 0
    policy_copied = 0

    for old_trial in selected_trials:
        new_ep = trial_to_new_ep[old_trial]
        trial_data = trials[old_trial]

        # Copy env_records for this trial
        for old_path in trial_data["files"]:
            parsed = parse_env_record_filename(old_path.name)
            if parsed is None:
                continue
            new_name = build_env_record_filename(
                prefix=parsed["prefix"],
                trial=new_ep,
                succ=parsed["succ"],
                suffix=parsed["suffix"],
            )
            new_path = out_env_dir / new_name

            if cfg.verbose or cfg.dry_run:
                print(f"  ENV: {old_path.name} -> {new_name}")

            if not cfg.dry_run:
                shutil.copy2(old_path, new_path)
            env_copied += 1

        # Copy and renumber policy_records for this trial
        policy_files = policy_by_ep.get(old_trial, [])

        # Sort by timestep to maintain order
        policy_files_parsed = []
        for path in policy_files:
            parsed = parse_policy_record_filename(path.name)
            if parsed:
                policy_files_parsed.append((path, parsed))

        policy_files_parsed.sort(key=lambda x: x[1]["t"])

        for old_path, parsed in policy_files_parsed:
            new_name = build_policy_record_filename(
                step=new_step,
                policy=parsed["policy"],
                task=parsed["task"],
                ep=new_ep,
                t=parsed["t"],
            )
            new_path = out_policy_dir / new_name

            if cfg.verbose or cfg.dry_run:
                print(f"  POL: {old_path.name} -> {new_name}")

            if not cfg.dry_run:
                # Load, update episode index if stored in pickle, and save
                try:
                    with open(old_path, "rb") as f:
                        data = pickle.load(f)

                    # Update episode index if present in the data
                    if isinstance(data, dict):
                        if "episode_idx" in data:
                            data["episode_idx"] = new_ep
                        if "ep" in data:
                            data["ep"] = new_ep
                        if "step" in data:
                            data["step"] = new_step

                    with open(new_path, "wb") as f:
                        pickle.dump(data, f)
                except Exception as e:
                    print(f"  [ERROR] Failed to process {old_path.name}: {e}")
                    # Fall back to simple copy
                    shutil.copy2(old_path, new_path)

            policy_copied += 1
            new_step += 1

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Target episodes:  {target_episodes}")
    print(f"  Actual episodes:  {len(selected_trials)}")
    print(f"  - Successful:     {selected_successes}")
    print(f"  - Failed:         {len(selected_trials) - selected_successes}")
    print(f"  Env files copied: {env_copied}")
    print(f"  Policy files:     {policy_copied}")
    print(f"  Dry run:          {cfg.dry_run}")
    print("=" * 50)

    if cfg.dry_run:
        print("\n[DRY RUN] No files were actually copied.")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
