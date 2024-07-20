import glob

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from robot_lab.utils.wrappers.rsl_rl import (
    RslRlAmpPpoAlgorithmCfg
)

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
MOTION_FILES = glob.glob(f"{ISAACLAB_ASSETS_DATA_DIR}/motion_files/mocap_motions_a1/*")

@configclass
class UnitreeA1AmpRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_a1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlAmpPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size = 1000000
    )

    amp_reward_coef = 2.0
    amp_motion_files = MOTION_FILES
    amp_num_preload_transitions = 2000000
    amp_task_reward_lerp = 0.3
    amp_discr_hidden_dims = [1024, 512]
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    dt = 0.005


@configclass
class UnitreeA1AmpFlatPPORunnerCfg(UnitreeA1AmpRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.max_iterations = 300
        self.experiment_name = "unitree_a1_flat"
        # self.policy.actor_hidden_dims = [128, 128, 128]
        # self.policy.critic_hidden_dims = [128, 128, 128]
