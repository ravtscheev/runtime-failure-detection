#!/usr/bin/env python3
"""
Comprehensive fix for pickle files:
- episode_idx: extract from filename
- mp4_path & wrist_mp4_path: fix trial/succ numbers
- step_done: optionally sync to match episode_success (last element)

Example:
    python scripts/fix_meta_pkl_comprehensive.py --path rollouts_out/pi0-ur5/hammer_cleanup/env_records
    python scripts/fix_meta_pkl_comprehensive.py --path rollouts_out/pi0-ur5/hammer_cleanup/env_records --fix-step-done
"""

import argparse
import pickle
import re
from pathlib import Path


def extract_trial_succ(filename: str, pattern: str) -> tuple[int, int] | None:
    """Extract (trial, succ) from filename using a regex with two capture groups."""
    match = re.search(pattern, filename)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def update_path_suffix(path_str: str, trial: int, succ: int) -> str:
    """Replace the --trialX--succY portion of a path string."""
    return re.sub(r"--trial\d+--succ\d+", f"--trial{trial}--succ{succ}", path_str)


def fix_meta_pkl_file(
    pkl_path: Path,
    trial_regex: str,
    episode_key: str = "episode_idx",
    fix_step_done: bool = False,
    dry_run: bool = False,
) -> bool:
    """Fix all issues in a single pickle file."""
    print(f"Processing: {pkl_path.name}")

    # Extract trial/succ from filename
    parsed = extract_trial_succ(pkl_path.name, trial_regex)
    if parsed is None:
        print("    Skipping: Could not extract trial/succ from filename")
        return False

    trial, succ = parsed

    # Load pickle
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return False

    changes = []

    # --- 1. Fix episode_idx ---
    if episode_key in data:
        current_idx = data[episode_key]
        if current_idx != trial:
            changes.append(f"{episode_key}: {current_idx} → {trial}")
            if not dry_run:
                data[episode_key] = trial
    else:
        print(f"    '{episode_key}' key not found")

    # --- 2. Fix mp4_path & wrist_mp4_path ---
    for key in ("mp4_path", "wrist_mp4_path"):
        if key not in data:
            print(f"    '{key}' key not found")
            continue
        current = data[key]
        updated = update_path_suffix(current, trial, succ)
        if updated != current:
            changes.append(f"{key}: ...{current[-40:]} → ...{updated[-40:]}")
            if not dry_run:
                data[key] = updated

    # --- 3. Fix step_done (optional) ---
    if fix_step_done:
        # Extract success value from filename (succ should be 0 or 1)
        expected_success = bool(succ)

        # Fix episode_success if it doesn't match filename
        if "episode_success" in data:
            current_success = data["episode_success"]
            if current_success != expected_success:
                changes.append(f"episode_success: {current_success} → {expected_success}")
                if not dry_run:
                    data["episode_success"] = expected_success

        # Fix step_done[-1] if it doesn't match filename
        if "step_done" in data and data["step_done"]:
            step_done = data["step_done"]
            if step_done[-1] != expected_success:
                changes.append(f"step_done[-1]: {step_done[-1]} → {expected_success}")
                if not dry_run:
                    step_done[-1] = expected_success
                    data["step_done"] = step_done

    # --- Report & Save ---
    if changes:
        print(f"   Fixed {len(changes)} field(s):")
        for change in changes:
            print(f"    - {change}")

        if not dry_run:
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(data, f)
                print("  💾 Saved")
            except Exception as e:
                print(f"  ❌ Error saving file: {e}")
                return False
        else:
            print("   [DRY RUN - no changes written]")
        return True

    print("   No changes needed")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive fix for pickle files: episode_idx, mp4_path, wrist_mp4_path, step_done"
    )
    parser.add_argument("--path", type=str, required=True, help="Path to directory containing pickle files")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pkl",
        help="Glob pattern for pickle files (default: *.pkl)",
    )
    parser.add_argument(
        "--trial-regex",
        type=str,
        default=r"--trial(\d+)--succ(\d+)",
        help="Regex with two capture groups for trial and succ (default: --trial(\\d+)--succ(\\d+))",
    )
    parser.add_argument(
        "--episode-key",
        type=str,
        default="episode_idx",
        help="Key name for episode index in pickle (default: episode_idx)",
    )
    parser.add_argument(
        "--fix-step-done",
        action="store_true",
        help="Also sync step_done[-1] to match episode_success",
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively in subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")

    args = parser.parse_args()

    dir_path = Path(args.path)
    if not dir_path.exists():
        print(f"❌ Error: Path does not exist: {dir_path}")
        return
    if not dir_path.is_dir():
        print(f"❌ Error: Path is not a directory: {dir_path}")
        return

    pkl_files = sorted(dir_path.rglob(args.pattern) if args.recursive else dir_path.glob(args.pattern))
    if not pkl_files:
        print(f"  No pickle files found matching pattern '{args.pattern}' in {dir_path}")
        if not args.recursive:
            print("    (Try adding --recursive to search subdirectories)")
        return

    print(f"Found {len(pkl_files)} pickle file(s) in {dir_path}")
    if args.dry_run:
        print(" DRY RUN MODE - No files will be modified\n")
    print()

    fixed_count = 0
    for pkl_file in pkl_files:
        if fix_meta_pkl_file(
            pkl_file,
            args.trial_regex,
            episode_key=args.episode_key,
            fix_step_done=args.fix_step_done,
            dry_run=args.dry_run,
        ):
            fixed_count += 1
        print()

    print("=" * 60)
    print(f"Summary: {fixed_count}/{len(pkl_files)} file(s) had changes")
    if args.dry_run:
        print(" DRY RUN MODE - No files were actually modified")
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
