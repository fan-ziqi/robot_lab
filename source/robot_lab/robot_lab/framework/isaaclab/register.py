"""Task registration — IsaacLab path."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym

from robot_lab.framework.spec import (
    FrameworkRequired,
    current_framework_supports,
)


def register_task(
    *,
    task_id: str,
    env_cfg: Callable | str,
    play_env_cfg: Callable | str | None = None,
    agent_cfg_entries: dict[str, str] | None = None,
    framework_required: FrameworkRequired = "any",
) -> None:
    """Register a robot_lab task with the IsaacLab gym registry.

    ``env_cfg`` and ``play_env_cfg`` are either zero-arg callables returning a
    fresh cfg, or ``module:Class`` entry-point strings (compatible with
    IsaacLab's ``isaaclab_tasks.utils`` machinery).
    """
    if not current_framework_supports(framework_required):
        print(
            f"INFO[robot_lab.framework] task {task_id!r} requires "
            f"framework={framework_required!r}; skipping under current framework."
        )
        return

    kwargs: dict[str, object] = {"env_cfg_entry_point": env_cfg}
    if play_env_cfg is not None:
        kwargs["play_env_cfg_entry_point"] = play_env_cfg
    if agent_cfg_entries:
        kwargs.update(agent_cfg_entries)

    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs=kwargs,
    )


__all__ = ["register_task"]
