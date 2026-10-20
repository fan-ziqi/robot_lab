# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Run the Isaac Lab 3.0 random-action agent with robot_lab tasks registered."""

import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import run_random_agent_cli  # noqa: E402

import robot_lab.tasks  # noqa: F401  # isort: skip


def main(argv: list[str] | None = None) -> int:
    """Run an environment with a random-action agent."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--headless" in args:
        args.remove("--headless")
        if not any(arg in ("--viz", "--visualizer") or arg.startswith(("--viz=", "--visualizer=")) for arg in args):
            args.extend(["--viz", "none"])
    return run_random_agent_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
