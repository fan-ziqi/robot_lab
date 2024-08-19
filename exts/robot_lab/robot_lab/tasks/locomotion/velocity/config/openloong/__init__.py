import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="rough-openloong-12-train",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.openloong_12_RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.openloong_12_RoughPPORunnerCfg,
    },
)

gym.register(
    id="rough-openloong-12-Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.openloong_12_RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.openloong_12_RoughPPORunnerCfg,
    },
)

gym.register(
    id="flat-openloong-12-trains",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.openloong_12_FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.openloong_12_FlatPPORunnerCfg,
    },
)

gym.register(
    id="flat-openloong-12-play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.openloong_12_FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.openloong_12_FlatPPORunnerCfg,
    },
)
