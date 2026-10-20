# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Run the Isaac Lab 3.0 random-action agent with robot_lab tasks registered."""


def main() -> int:
    import sys

    import robot_lab.tasks  # noqa: F401

    from isaaclab_rl.entrypoints import run_random_agent_cli

    args = list(sys.argv[1:])
    if "--headless" in args:
        args.remove("--headless")
        if not any(arg in ("--viz", "--visualizer") or arg.startswith(("--viz=", "--visualizer=")) for arg in args):
            args.extend(["--viz", "none"])
    return run_random_agent_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
