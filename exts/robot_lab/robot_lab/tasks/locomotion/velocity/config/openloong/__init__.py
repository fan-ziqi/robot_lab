import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-OpenLoong-OpenLoong-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.OpenLoong_RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenLoongOpenLoongRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-OpenLoong-OpenLoong-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OpenLoong_FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenLoongOpenLoongFlatPPORunnerCfg",
    },
)
