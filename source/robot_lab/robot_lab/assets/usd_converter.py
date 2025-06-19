# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import tempfile
from typing import Literal, Optional

from xacrodoc import XacroDoc

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg, UrdfConverter, UrdfConverterCfg


def urdf_to_usd(
    file_path: str,
    output_usd_path: Optional[str] = None,
    merge_joints: bool = False,
    fix_base: bool = False,
    joint_stiffness: float = 100.0,
    joint_damping: float = 1.0,
    joint_target_type: Literal["position", "velocity", "none"] = "position",
) -> str:
    """
    Convert URDF to USD format with full parameter support.

    Args:
        file_path: Input URDF file path
        output_usd_path: Output USD file path (generates temp file if None)
        merge_joints: Whether to merge fixed joints (default False)
        fix_base: Whether to fix the base (default False)
        joint_stiffness: Joint drive stiffness (default 100.0)
        joint_damping: Joint drive damping (default 1.0)
        joint_target_type: Joint control type ("position"/"velocity"/"none")

    Returns:
        Path to generated USD file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: For invalid joint_target_type
        RuntimeError: If conversion fails
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"URDF file not found: {file_path}")
    if joint_target_type not in ("position", "velocity", "none"):
        raise ValueError("joint_target_type must be 'position', 'velocity' or 'none'")

    is_temp_file = output_usd_path is None
    usd_path = output_usd_path if not is_temp_file else tempfile.mktemp(suffix=".usd")

    try:
        if not is_temp_file:
            os.makedirs(os.path.dirname(usd_path), exist_ok=True)

        converter_cfg = UrdfConverterCfg(
            asset_path=file_path,
            usd_dir=os.path.dirname(usd_path),
            usd_file_name=os.path.basename(usd_path),
            fix_base=fix_base,
            merge_fixed_joints=merge_joints,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=joint_stiffness, damping=joint_damping),
                target_type=joint_target_type,
            ),
        )

        UrdfConverter(converter_cfg)

        if is_temp_file:
            atexit.register(lambda: os.unlink(usd_path) if os.path.exists(usd_path) else None)

        return usd_path

    except Exception as e:
        if is_temp_file and os.path.exists(usd_path):
            os.unlink(usd_path)
        raise RuntimeError(f"URDF conversion failed: {str(e)}") from e


def xacro_to_usd(
    file_path: str,
    output_usd_path: Optional[str] = None,
    merge_joints: bool = False,
    fix_base: bool = False,
) -> str:
    """
    Convert XACRO to USD format using xacrodoc.

    Args:
        file_path: Input .xacro file path
        output_usd_path: Output USD path (optional)
        merge_joints: Whether to merge fixed joints
        fix_base: Whether to fix the base

    Returns:
        Path to generated USD file
    """
    doc = XacroDoc.from_file(file_path)

    with doc.temp_urdf_file_path() as file_path:
        return urdf_to_usd(
            file_path=file_path,
            output_usd_path=output_usd_path,
            merge_joints=merge_joints,
            fix_base=fix_base,
        )


def mjcf_to_usd(
    file_path: str,
    output_usd_path: Optional[str] = None,
    fix_base: bool = False,
    import_sites: bool = True,
    make_instanceable: bool = True,
) -> str:
    """
    Convert MJCF to USD format.

    Args:
        file_path: Input MJCF file path
        output_usd_path: Output USD path (generates temp file if None)
        fix_base: Whether to fix the base (default False)
        import_sites: Whether to import <site> tags (default True)
        make_instanceable: Whether to make instanceable (default True)

    Returns:
        Path to generated USD file

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If conversion fails
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"MJCF file not found: {file_path}")

    is_temp_file = output_usd_path is None
    usd_path = output_usd_path if not is_temp_file else tempfile.mktemp(suffix=".usd")

    try:
        if not is_temp_file:
            os.makedirs(os.path.dirname(usd_path), exist_ok=True)

        converter_cfg = MjcfConverterCfg(
            asset_path=file_path,
            usd_dir=os.path.dirname(usd_path),
            usd_file_name=os.path.basename(usd_path),
            fix_base=fix_base,
            import_sites=import_sites,
            make_instanceable=make_instanceable,
            force_usd_conversion=True,
        )

        MjcfConverter(converter_cfg)
        return usd_path

    except Exception as e:
        if is_temp_file and os.path.exists(usd_path):
            os.unlink(usd_path)
        raise RuntimeError(f"MJCF conversion failed: {str(e)}") from e
