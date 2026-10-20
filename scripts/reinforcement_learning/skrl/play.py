# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Play robot_lab tasks using the Isaac Lab 3.0 skrl entrypoint.

Run with --help for current CLI flags. Use --checkpoint for resume/play and
--headless --viz none for simulation without a viewer.
"""

import argparse
import sys
from contextlib import contextmanager


def positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


@contextmanager
def bounded_playback(max_steps: int, common, backend):
    """Bound the upstream loop while preserving its inference/video/time handling."""
    original_budget = common.video_playback_steps
    original_playback = backend.run_playback

    def step_budget(args_cli, env_cfg):
        video_budget = original_budget(args_cli, env_cfg)
        return max_steps if video_budget is None else min(max_steps, video_budget)

    def counted_playback(step, **kwargs):
        completed = 0

        def counted_step():
            nonlocal completed
            step()
            completed += 1

        try:
            return original_playback(counted_step, **kwargs)
        finally:
            print(f"[robot_lab] playback_steps={completed} requested_max_steps={max_steps}")

    common.video_playback_steps = step_budget
    # The backend imports run_playback by value; patch its reference separately.
    backend.run_playback = counted_playback
    try:
        yield
    finally:
        common.video_playback_steps = original_budget
        backend.run_playback = original_playback


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--max_steps", type=positive_int, default=None)
    wrapper_args, args = parser.parse_known_args(args)
    if "--help" in args or "-h" in args:
        print("robot_lab wrapper option: --max_steps N (positive environment-step limit)")

    import warp as wp

    wp.config.enable_backward = False
    import robot_lab.tasks  # noqa: F401

    from isaaclab_rl.entrypoints import run_play_cli

    # The installed 3.0 launcher uses --viz none; accept the task's headless spelling.
    if "--headless" in args:
        args.remove("--headless")
        if not any(arg in ("--viz", "--visualizer") or arg.startswith(("--viz=", "--visualizer=")) for arg in args):
            args.extend(["--viz", "none"])
    if wrapper_args.max_steps is None:
        return run_play_cli(["--rl_library", "skrl", *args])

    from isaaclab_rl.entrypoints import common
    from isaaclab_rl.entrypoints.backends import play_skrl

    with bounded_playback(wrapper_args.max_steps, common, play_skrl):
        return run_play_cli(["--rl_library", "skrl", *args])


if __name__ == "__main__":
    raise SystemExit(main())
