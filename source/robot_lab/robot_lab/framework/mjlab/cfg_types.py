"""Cfg-type aliases for the mjlab backend.

Re-export every type the business layer references. mjlab puts term cfgs in
per-manager modules, so the imports look slightly different from IsaacLab's
flat ``isaaclab.managers`` namespace.

NOTE: ``configclass`` is intentionally NOT exposed (parity with the IsaacLab
side in cfg_types.py). User code never decorates cfg classes itself; managers
are constructed via the ``manager_container`` / ``make_observation_group``
factories.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg as EnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg


# mjlab does not expose a top-level ManagerTermBase; stateful reward terms use
# the ``__init__(self, cfg, env)`` + ``__call__(self, env, ...)`` pattern. We
# provide a minimal base class with the same surface so business code can
# target one name across both backends.
class ManagerTermBase:
    """Stateful manager-term base. Mirrors IsaacLab's ``ManagerTermBase`` API."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env

    def reset(self, env_ids):
        """Override in subclasses if state needs resetting on env reset."""
        del env_ids


__all__ = [
    "ActionTermCfg",
    "CommandTermCfg",
    "CurriculumTermCfg",
    "EnvCfg",
    "EventTermCfg",
    "ManagerTermBase",
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "RewardTermCfg",
    "SceneCfg",
    "SceneEntityCfg",
    "TerminationTermCfg",
]
