#!/usr/bin/env python3

"""Utility script to print available robosuite tasks (including mimicgen nut assembly variants)."""

import mimicgen  # noqa: F401  # ty:ignore[unresolved-import]
import robosuite as suite  # noqa: F401  # ty:ignore[unresolved-import]
import robosuite_task_zoo  # noqa: F401  # ty:ignore[unresolved-import]
from robosuite.environments import ALL_ENVIRONMENTS  # ty:ignore[unresolved-import]


def main() -> None:
    print("\n".join(sorted(ALL_ENVIRONMENTS)))


if __name__ == "__main__":
    main()
