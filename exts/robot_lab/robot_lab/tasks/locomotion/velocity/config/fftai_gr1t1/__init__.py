import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.FFTAIGR1T1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FFTAIGR1T1RoughPPORunnerCfg,
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.FFTAIGR1T1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FFTAIGR1T1RoughPPORunnerCfg,
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.FFTAIGR1T1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FFTAIGR1T1FlatPPORunnerCfg,
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.FFTAIGR1T1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FFTAIGR1T1FlatPPORunnerCfg,
    },
)
