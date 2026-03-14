#!/usr/bin/env python
"""
Copy rollout bundles (env_records + policy_records) from one location to another.

Example:
    # Copy a single rollout bundle
    python scripts/copy_rollout_bundle.py \\
      --source rollouts/pi0-ur5/hammer_cleanup \\
      --dest rollouts_out/pi0-ur5/hammer_cleanup
"""

import argparse
import shutil
from pathlib import Path


def copy_bundle(source_dir: Path, dest_dir: Path, merge: bool = False, dry_run: bool = False) -> bool:
    """Copy env_records and policy_records from source to dest.

    If merge=True, copies files into dest's env_records/policy_records (flat structure).
    If merge=False, preserves source structure in dest.
    """
    env_src = source_dir / "env_records"
    policy_src = source_dir / "policy_records"

    # Check if both exist
    if not env_src.exists():
        print(f"  env_records not found in {source_dir}")
        return False
    if not policy_src.exists():
        print(f"  policy_records not found in {source_dir}")
        return False

    env_dest = dest_dir / "env_records"
    policy_dest = dest_dir / "policy_records"

    # Count files
    env_files = list(env_src.glob("*"))
    policy_files = list(policy_src.glob("*"))

    print(f" env_records:    {len(env_files)} file(s)")
    print(f" policy_records: {len(policy_files)} file(s)")

    if dry_run:
        print(f" [DRY RUN] Would copy to: {dest_dir}")
        return True

    try:
        if merge:
            # Merge mode: copy files into dest's env_records/policy_records
            env_dest = dest_dir / "env_records"
            policy_dest = dest_dir / "policy_records"

            # Create dirs if they don't exist
            env_dest.mkdir(parents=True, exist_ok=True)
            policy_dest.mkdir(parents=True, exist_ok=True)

            # Copy files (not entire directory)
            for file in env_files:
                if file.is_file():
                    shutil.copy2(file, env_dest / file.name)
            print(f" Merged env_records → {env_dest}")

            for file in policy_files:
                if file.is_file():
                    shutil.copy2(file, policy_dest / file.name)
            print(f" Merged policy_records → {policy_dest}")
        else:
            # Normal mode: preserve task structure
            dest_dir.mkdir(parents=True, exist_ok=True)
            env_dest = dest_dir / "env_records"
            policy_dest = dest_dir / "policy_records"

            if env_dest.exists():
                shutil.rmtree(env_dest)
            shutil.copytree(env_src, env_dest)
            print(f" Copied env_records → {env_dest}")

            if policy_dest.exists():
                shutil.rmtree(policy_dest)
            shutil.copytree(policy_src, policy_dest)
            print(f" Copied policy_records → {policy_dest}")

        return True
    except Exception as e:
        print(f" Error copying: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy rollout bundles (env_records + policy_records) between directories"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory (e.g., rollouts/pi0-ur5 or rollouts/pi0-ur5/hammer_cleanup)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Destination directory (e.g., rollouts_out/pi0-ur5)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific task subdirectories to copy (if source contains multiple tasks)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all tasks into single env_records/policy_records folders (flat structure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying",
    )

    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        print(f" Error: Source does not exist: {source}")
        return

    if args.dry_run:
        print(" DRY RUN MODE - No files will be copied\n")

    # Determine what to copy
    if args.tasks:
        # Copy specified tasks
        mode_str = "Merging" if args.merge else "Copying"
        print(f"{mode_str} {len(args.tasks)} task bundle(s) from {source}\n")
        copied = 0
        for task in args.tasks:
            print(f"Task: {task}")
            task_src = source / task
            task_dest = dest / task if not args.merge else dest

            if not task_src.exists():
                print(f" Task directory not found: {task_src}\n")
                continue

            if copy_bundle(task_src, task_dest, merge=args.merge, dry_run=args.dry_run):
                copied += 1
            print()

        print("=" * 60)
        if args.merge:
            print(f"Summary: Merged {copied}/{len(args.tasks)} task bundle(s) into {dest}")
        else:
            print(f"Summary: {copied}/{len(args.tasks)} task bundle(s) copied")

    else:
        # Check if source itself is a bundle
        if (source / "env_records").exists() and (source / "policy_records").exists():
            # Copy single bundle
            mode_str = "Merging" if args.merge else "Copying"
            print(f"{mode_str} bundle: {source.name}\n")
            copy_bundle(source, dest, merge=args.merge, dry_run=args.dry_run)
        else:
            # Copy all subdirectories that contain bundles
            tasks = [
                d
                for d in source.iterdir()
                if d.is_dir() and (d / "env_records").exists() and (d / "policy_records").exists()
            ]

            if not tasks:
                print(f" No rollout bundles found in {source}")
                print(" (Looking for subdirectories with both env_records and policy_records)")
                return

            mode_str = "Merging" if args.merge else "Found"
            print(f"{mode_str} {len(tasks)} task bundle(s) in {source}\n")
            copied = 0
            for task_dir in tasks:
                task_name = task_dir.name
                print(f"Task: {task_name}")
                task_dest = dest / task_name if not args.merge else dest

                if copy_bundle(task_dir, task_dest, merge=args.merge, dry_run=args.dry_run):
                    copied += 1
                print()

            print("=" * 60)
            if args.merge:
                print(f"Summary: Merged {copied}/{len(tasks)} task bundle(s) into {dest}")
            else:
                print(f"Summary: {copied}/{len(tasks)} task bundle(s) copied")

    if args.dry_run:
        print(" DRY RUN MODE - No files were actually copied")
        print("Run without --dry-run to perform the copy")


if __name__ == "__main__":
    main()
