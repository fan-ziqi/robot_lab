# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 robot_lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Newton-only Go2 settings with a checksum-bound measured PhysX order profile."""

from robot_lab.tasks.manager_based.locomotion.velocity.newton_support import configure_newton, load_order_profile


def configure_go2_newton(cfg):
    """Preserve PhysX fields and apply independently measured exact Newton order."""
    joint_names, body_names = load_order_profile(cfg, "unitree_go2")
    return configure_newton(cfg, joint_names, body_names, njmax=256, nconmax=128)
