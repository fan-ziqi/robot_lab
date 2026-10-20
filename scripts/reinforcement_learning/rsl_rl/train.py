# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Train robot_lab tasks using the Isaac Lab 3.0 rsl_rl entrypoint.

Run with --help for current CLI flags. Use --checkpoint for resume/play and
--headless --viz none for simulation without a viewer.
"""

import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

import robot_lab.tasks  # noqa: F401  # isort: skip


def main(argv: list[str] | None = None) -> int:
    """Train an agent with RSL-RL."""
    args = list(sys.argv[1:] if argv is None else argv)
    # The installed 3.0 launcher uses --viz none; accept the task's headless spelling.
    if "--headless" in args:
        args.remove("--headless")
        if not any(arg in ("--viz", "--visualizer") or arg.startswith(("--viz=", "--visualizer=")) for arg in args):
            args.extend(["--viz", "none"])
    return run_train_cli(["--rl_library", "rsl_rl", *args])


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
