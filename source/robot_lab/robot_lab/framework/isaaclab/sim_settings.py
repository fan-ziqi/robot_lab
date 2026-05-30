"""Backend-specific sim cfg fields — IsaacLab.

Called from ``make_velocity_env_cfg`` after the cfg is otherwise complete.
Sets PhysX-specific knobs that don't belong on the neutral EnvCfg surface.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg


def apply_sim_settings(cfg: ManagerBasedRLEnvCfg) -> None:
    cfg.sim.dt = 0.005
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    if cfg.scene.terrain is not None and getattr(
        cfg.scene.terrain, "physics_material", None
    ) is not None:
        cfg.sim.physics_material = cfg.scene.terrain.physics_material


__all__ = ["apply_sim_settings"]
