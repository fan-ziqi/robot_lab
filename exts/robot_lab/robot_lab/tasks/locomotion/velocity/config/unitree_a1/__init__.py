import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeA1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeA1FlatPPORunnerCfg,
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeA1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeA1RoughPPORunnerCfg,
    },
)
