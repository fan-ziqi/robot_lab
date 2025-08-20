# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import cusrl
from cusrl.environment.isaaclab import TrainerCfg

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry import anymal


@configclass
class AnymalDRoughTrainerCfg(TrainerCfg):
    max_iterations = 3000
    save_interval = 100
    experiment_name = "anymal_d_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@configclass
class AnymalDFlatTrainerCfg(AnymalDRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "anymal_d_flat"


def get_environment_mirrors(environment):
    def get_augmented_observation(obs):
        batch_size = obs.size(0)
        augmented_obs = anymal.compute_symmetric_states(environment, obs=obs)[0]
        # preserves only the augmented ones
        return augmented_obs[batch_size:]

    def get_augmented_action(action):
        batch_size = action.size(0)
        augmented_action = anymal.compute_symmetric_states(environment, actions=action)[1]
        return augmented_action[batch_size:]

    return {
        "mirror_observation": get_augmented_observation,
        "mirror_state": get_augmented_observation,
        "mirror_action": get_augmented_action,
    }


@configclass
class AnymalDRoughTrainerCfgWithSymmetryAugmentation(AnymalDRoughTrainerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.DynamicEnvironmentSpecOverride(get_environment_mirrors),
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.SymmetricDataAugmentation(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@configclass
class AnymalDFlatTrainerCfgWithSymmetryAugmentation(AnymalDRoughTrainerCfgWithSymmetryAugmentation):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "anymal_d_flat"
