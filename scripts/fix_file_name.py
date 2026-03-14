#!/usr/bin/env python

"""
Script to fix succ flags in filenames based on agentview recordings.
Example usage:
    python scripts/fix_file_name.py --directory path/to/policy_records
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro

TRIAL_RE = re.compile(r"--trial(?P<trial>\d+)--succ(?P<succ>[01])")


@dataclass
class Config:
    """Synchronize succ flags across files based on agentview recordings."""

    directory: Path
    dry_run: bool = False
    verbose: bool = False


def log(msg: str, *, verbose: bool = False, cfg: Config | None = None):
    if not verbose or (cfg and cfg.verbose):
        print(msg)


def extract_trial_succ(name: str):
    """Extract (trial, succ) from filename or return None."""
    m = TRIAL_RE.search(name)
    if not m:
        return None
    return m.group("trial"), m.group("succ")


def main(cfg: Config):
    directory = cfg.directory

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = [p for p in directory.iterdir() if p.is_file()]
    if not files:
        raise RuntimeError("Directory contains no files.")

    # ------------------------------------------------------------------
    # Step 1: collect agentview success per trial
    # ------------------------------------------------------------------
    trial_to_succ: dict[str, str] = {}
    duplicates = defaultdict(list)

    for path in files:
        if "agentview" not in path.name:
            continue

        parsed = extract_trial_succ(path.name)
        if not parsed:
            continue

        trial, succ = parsed
        duplicates[trial].append((path.name, succ))

        # last one wins, but we warn below
        trial_to_succ[trial] = succ

    if not trial_to_succ:
        raise RuntimeError("No agentview files found — nothing to sync.")

    for trial, entries in duplicates.items():
        succs = {s for _, s in entries}
        if len(succs) > 1:
            print(f"[WARN] Conflicting agentview succ values for trial {trial}:")
            for name, succ in entries:
                print(f"  {name} → succ{succ}")

    # ------------------------------------------------------------------
    # Step 2: rename other files to match agentview
    # ------------------------------------------------------------------
    rename_count = 0
    skipped = 0

    for path in files:
        if "agentview" in path.name:
            continue  # source of truth

        parsed = extract_trial_succ(path.name)
        if not parsed:
            continue

        trial, current_succ = parsed

        target_succ = trial_to_succ.get(trial)
        if target_succ is None:
            print(f"[WARN] No agentview for trial {trial}: {path.name}")
            skipped += 1
            continue

        if current_succ == target_succ:
            log(f"[OK] {path.name}", verbose=True, cfg=cfg)
            continue

        new_name = TRIAL_RE.sub(
            f"--trial{trial}--succ{target_succ}",
            path.name,
        )
        new_path = path.with_name(new_name)

        if new_path.exists():
            print(f"[ERROR] Target already exists, skipping:\n  {new_name}")
            skipped += 1
            continue

        if cfg.dry_run:
            print(f"[DRY] {path.name} → {new_name}")
            continue

        print(f"Renaming:\n  {path.name}\n→ {new_name}\n")
        path.rename(new_path)
        rename_count += 1

    print("────────────────────────────────────")
    print(f"Renamed files : {rename_count}")
    print(f"Skipped files : {skipped}")
    print(f"Dry run       : {cfg.dry_run}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
