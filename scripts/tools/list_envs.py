# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `isaaclab_tasks` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import textwrap
from prettytable import PrettyTable

import robot_lab.tasks  # noqa: F401


def main():
    """Print all environments registered in `isaaclab_tasks` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"
    table.hrules = 1

    # set max width for text wrapping
    max_width = 50

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if "RobotLab" in task_spec.id:
            # wrap long text in each column before adding it to the table
            task_name = textwrap.fill(task_spec.id, max_width)
            entry_point = textwrap.fill(task_spec.entry_point, max_width)
            config = textwrap.fill(task_spec.kwargs["env_cfg_entry_point"], max_width)

            # add details to table
            table.add_row([index + 1, task_name, entry_point, config])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
