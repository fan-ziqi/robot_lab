"""Backend-specific sim cfg fields — mjlab."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def apply_sim_settings(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 50


__all__ = ["apply_sim_settings"]
