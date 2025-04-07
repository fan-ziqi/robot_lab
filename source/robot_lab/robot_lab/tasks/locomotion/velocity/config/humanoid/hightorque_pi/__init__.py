import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-HighTorque-Pi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HighTorquePiRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighTorquePiRoughPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-Velocity-Flat-HighTorque-Pi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HighTorquePiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighTorquePiFlatPPORunnerCfg",
    },
)