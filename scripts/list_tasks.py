"""Utility script to print available robosuite tasks (including mimicgen nut assembly variants)."""

import mimicgen  # noqa: F401
import robosuite as suite  # noqa: F401
import robosuite_task_zoo  # noqa: F401
from robosuite.environments import ALL_ENVIRONMENTS


def main() -> None:
    print("\n".join(sorted(ALL_ENVIRONMENTS)))


if __name__ == "__main__":
    main()
