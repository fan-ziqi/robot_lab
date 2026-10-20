# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for ``isaaclab list_envs``."""

import sys

from isaaclab.cli.commands.list_envs import command_list_envs

if __name__ == "__main__":
    command_list_envs(sys.argv[1:])
