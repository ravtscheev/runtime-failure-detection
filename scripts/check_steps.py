#!/usr/bin/env python
"""
Script to check if all steps from 0 to max_step are present in a folder.
Files are expected to follow the naming pattern: step_XXXX--...--meta
"""

import re
import sys
from argparse import ArgumentParser
from pathlib import Path


def extract_step_number(filename):
    """Extract step number from filename like 'step_6899--pi0-ur5--task_Kitchen_D1--ep_49--t_695--meta'"""
    match = re.match(r"step_(\d+)--", filename)
    if match:
        return int(match.group(1))
    return None


def check_steps(folder_path, max_step=None):
    """
    Check if all steps from 0 to max_step are present in the folder.

    Args:
        folder_path: Path to the folder containing step files
        max_step: Maximum step number to check (if None, will find it automatically)

    Returns:
        Dictionary with results including found steps, missing steps, etc.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return None

    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return None

    # Extract step numbers from all files
    found_steps = set()
    step_to_files = {}  # Track files for each step to detect duplicates
    for filename in folder_path.iterdir():
        if filename.is_file():
            step_num = extract_step_number(filename.name)
            if step_num is not None:
                found_steps.add(step_num)
                if step_num not in step_to_files:
                    step_to_files[step_num] = []
                step_to_files[step_num].append(filename.name)

    if not found_steps:
        print(f"Error: No step files found in '{folder_path}'")
        print("Expected naming pattern: step_XXXX--...")
        return None

    # Determine max step
    if max_step is None:
        max_step = max(found_steps)

    # Find missing steps
    expected_steps = set(range(0, max_step + 1))
    missing_steps = sorted(expected_steps - found_steps)

    # Find duplicate steps
    duplicate_steps = {step: files for step, files in step_to_files.items() if len(files) > 1}

    # Prepare results
    results = {
        "folder": str(folder_path),
        "total_expected": max_step + 1,
        "total_found": len(found_steps),
        "missing_count": len(missing_steps),
        "missing_steps": missing_steps,
        "duplicate_count": len(duplicate_steps),
        "duplicate_steps": duplicate_steps,
        "all_present": len(missing_steps) == 0,
        "no_duplicates": len(duplicate_steps) == 0,
    }

    return results


def main():
    parser = ArgumentParser(description="Check if all steps from 0 to max_step are present in a folder")
    parser.add_argument("folder", help="Path to the folder containing step files")
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Maximum step number to check (default: auto-detect from found files)",
    )
    parser.add_argument("--show-missing", action="store_true", help="Show all missing step numbers")
    parser.add_argument(
        "--show-duplicates", action="store_true", help="Show all duplicate step numbers and their files"
    )

    args = parser.parse_args()

    results = check_steps(args.folder, args.max_step)

    if results is None:
        sys.exit(1)

    # Print results
    print(f"Folder: {results['folder']}")
    print(f"Expected steps: 0 to {results['total_expected'] - 1} (total: {results['total_expected']})")
    print(f"Found: {results['total_found']}")
    print(f"Missing: {results['missing_count']}")
    print(f"Duplicates: {results['duplicate_count']}")

    if results["all_present"]:
        print("\n All steps are present!")
    else:
        print(f"\n✗ {results['missing_count']} steps are missing")
        if args.show_missing and results["missing_steps"]:
            print("\nMissing steps:")
            # Print in ranges for better readability
            if len(results["missing_steps"]) > 20:
                print(f"  {results['missing_steps'][:20]} ... and {len(results['missing_steps']) - 20} more")
                print("\n  To save all missing steps to a file:")
                print(f"  python scripts/check_steps.py {args.folder} --show-missing > missing_steps.txt")
            else:
                for step in results["missing_steps"]:
                    print(f"  step_{step}")

    if not results["no_duplicates"]:
        print(f"\n✗ {results['duplicate_count']} steps have duplicates")
        if args.show_duplicates:
            print("\nDuplicate steps:")
            for step in sorted(results["duplicate_steps"].keys()):
                print(f"  step_{step}:")
                for filename in results["duplicate_steps"][step]:
                    print(f"    - {filename}")
    else:
        print("\n No duplicate steps found!")


if __name__ == "__main__":
    main()
