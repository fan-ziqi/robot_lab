"""RL cfg shim — mjlab path.

mjlab.rl exposes RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg, RslRlModelCfg,
RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper. RslRlPpoActorCriticCfg is the
IsaacLab name for what mjlab calls RslRlModelCfg; alias accordingly so
business agent cfgs that import RslRlPpoActorCriticCfg keep working.
"""

from __future__ import annotations

from mjlab.rl import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg as RslRlPpoActorCriticCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

__all__ = [
    "RslRlBaseRunnerCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoActorCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlVecEnvWrapper",
]
