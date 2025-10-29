# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class AnymalDFlatDistillationTrainerCfg(TrainerCfg):
    max_iterations = 300
    save_interval = 50
    experiment_name = "anymal_d_flat"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=120,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.StubModule.Factory(),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=2, num_mini_batches=8),
        hooks=[
            cusrl.hook.ModuleInitialization(init_actor=False, init_critic=False, distribution_std=0.1),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PolicyDistillationLoss(""),
            cusrl.hook.GradientClipping(1.0),
        ],
    )
