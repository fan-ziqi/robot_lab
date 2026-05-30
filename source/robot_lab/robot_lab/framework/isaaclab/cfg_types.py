"""Cfg-type aliases for the IsaacLab backend.

Re-export every type the business layer references so business code can write
``from robot_lab.framework import RewardTermCfg`` without ever touching
``isaaclab.*`` directly.

NOTE: ``configclass`` is intentionally NOT exported. The framework's public API
hides decorator-style cfg construction; user code uses dict-style cfg + the
``manager_container`` / ``make_observation_group`` factories instead. Exposing
``configclass`` would let users decorate their own classes and silently get
different semantics on mjlab (where it would have to be a no-op).
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg as EnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg as SceneCfg

# isaaclab.utils.configclass is used internally by container.py and obs_group.py
# but NOT re-exported here.

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
