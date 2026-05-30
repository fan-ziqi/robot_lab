"""List robot_lab tasks registered for the mjlab backend."""

from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("ROBOT_LAB_FRAMEWORK", "mjlab")
    import robot_lab.tasks  # noqa: F401

    from mjlab.scripts.list_envs import main as mjlab_main

    mjlab_main()


if __name__ == "__main__":
    main()
