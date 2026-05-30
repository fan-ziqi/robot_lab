"""mjlab RSL-RL training entry point.

Thin wrapper that imports robot_lab.tasks (so all RobotLab-* tasks register
under the mjlab backend) and then delegates to mjlab's own train script.

Users get mjlab's native tyro CLI (e.g. ``--env.scene.num-envs 4096``) for
free; robot_lab adds task registration and the dispatcher routing.
"""

from __future__ import annotations

import os


def main() -> None:
    # Lock backend before any robot_lab imports.
    os.environ.setdefault("ROBOT_LAB_FRAMEWORK", "mjlab")

    # Trigger task registration into mjlab's task registry.
    import robot_lab.tasks  # noqa: F401

    # Delegate to mjlab's own train CLI (tyro-based; reads sys.argv).
    from mjlab.scripts.train import main as mjlab_train_main

    mjlab_train_main()


def cli_entry() -> None:
    main()


if __name__ == "__main__":
    main()
