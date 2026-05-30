"""mjlab RSL-RL play entry point. Wraps mjlab's own play.py."""

from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("ROBOT_LAB_FRAMEWORK", "mjlab")
    import robot_lab.tasks  # noqa: F401

    from mjlab.scripts.play import main as mjlab_play_main

    mjlab_play_main()


def cli_entry() -> None:
    main()


if __name__ == "__main__":
    main()
