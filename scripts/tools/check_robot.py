# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to check robot model.")
parser.add_argument(
    "type",
    type=str,
    choices=["urdf", "mjcf", "xacro"],
    help="The type of the input file.",
)
parser.add_argument("path", type=str, help="The path to the input URDF file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.utils.assets import check_file_path

from robot_lab.assets.utils.usd_converter import mjcf_to_usd, urdf_to_usd, xacro_to_usd  # noqa: F401


def main():
    # check valid file path
    file_path = args_cli.path
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    if not check_file_path(file_path):
        raise ValueError(f"Invalid file path: {file_path}")

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    usd_path = None
    if args_cli.type == "urdf":
        usd_path = urdf_to_usd(
            file_path=file_path,
            merge_joints=True,
            fix_base=False,
        )
    elif args_cli.type == "mjcf":
        usd_path = mjcf_to_usd(
            file_path=file_path,
            import_sites=True,
            fix_base=False,
        )
    elif args_cli.type == "xacro":
        usd_path = xacro_to_usd(
            file_path=file_path,
            merge_joints=True,
            fix_base=False,
        )

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(str(usd_path))
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
