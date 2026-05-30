"""Task registration — mjlab path."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_lab.framework.spec import (
    FrameworkRequired,
    current_framework_supports,
)


def _resolve_entry(entry: str | Callable) -> Any:
    """Resolve an entry-point string ``module:Class`` to the class itself."""
    if callable(entry):
        return entry
    module_name, _, attr = entry.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def register_task(
    *,
    task_id: str,
    env_cfg: Callable,
    play_env_cfg: Callable | None = None,
    agent_cfg_entries: dict[str, str] | None = None,
    framework_required: FrameworkRequired = "any",
) -> None:
    """Register a robot_lab task with mjlab's task registry."""
    if not current_framework_supports(framework_required):
        print(
            f"INFO[robot_lab.framework] task {task_id!r} requires "
            f"framework={framework_required!r}; skipping under current framework."
        )
        return

    rl_cfg = None
    if agent_cfg_entries and "rsl_rl_cfg_entry_point" in agent_cfg_entries:
        rl_cfg_obj = _resolve_entry(agent_cfg_entries["rsl_rl_cfg_entry_point"])
        rl_cfg = rl_cfg_obj() if isinstance(rl_cfg_obj, type) else rl_cfg_obj

    env_cfg_inst = env_cfg() if callable(env_cfg) else env_cfg
    play_env_cfg_inst = (
        play_env_cfg() if (play_env_cfg is not None and callable(play_env_cfg)) else play_env_cfg
    )

    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg_inst,
        play_env_cfg=play_env_cfg_inst if play_env_cfg_inst is not None else env_cfg_inst,
        rl_cfg=rl_cfg,
        runner_cls=VelocityOnPolicyRunner,
    )


__all__ = ["register_task"]
