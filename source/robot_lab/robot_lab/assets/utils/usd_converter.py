# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
This module provides utility functions for converting URDF, XACRO, and MJCF files to USD format.

It includes a lazy-loading mechanism to defer the conversion until the USD path is actually needed,
which is useful for improving performance when dealing with large numbers of assets.
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import numpy as np
import os
import tempfile
import time
from typing import Any, Literal

from pxr import Usd
from xacrodoc import XacroDoc

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners import from_files


def spawn_from_lazy_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | np.ndarray | None = None,
    orientation: tuple[float, float, float, float] | np.ndarray | None = None,
) -> Usd.Prim:
    """A wrapper for :func:`isaaclab.sim.spawners.from_files.spawn_from_usd` that resolves a :class:`LazyUsdPath`.

    This function allows deferring the USD conversion until the asset is actually spawned.
    It resolves the lazy-loaded path into a string before calling the original spawner.

    Args:
        prim_path: The prim path for the new articulation.
        cfg: The configuration instance. The ``cfg.usd_path`` is expected to be a :class:`LazyUsdPath` object.
        translation: The translation to apply to the prim. Defaults to None.
        orientation: The orientation to apply to the prim (as a quaternion). Defaults to None.

    Returns:
        The spawned prim.
    """
    # The usd_path is a LazyUsdPath object. We resolve it to a string here.
    if isinstance(cfg.usd_path, LazyUsdPath):
        cfg.usd_path = str(cfg.usd_path)
    # Call the original spawner from Isaac Lab.
    return from_files.spawn_from_usd(prim_path, cfg, translation, orientation)


class LazyUsdPath:
    """A wrapper for lazy-loading of USD paths.

    The conversion is only performed when the object is converted to a string.
    This object is designed to be serializable by Isaac Lab's config system.
    """

    def __init__(self, converter: UsdConverter, **kwargs: Any):
        """Initializes the lazy USD path loader.

        Args:
            converter: The USD converter instance to use.
            **kwargs: The keyword arguments for the conversion.
        """
        # Note: Attributes must not start with an underscore to be serialized by class_to_dict.
        self.converter = converter
        self.kwargs = kwargs
        # Private attribute for caching the result at runtime.
        self._usd_path: str | None = None

    def __str__(self) -> str:
        """Performs the conversion (if not already done) and returns the USD path as a string."""
        if self._usd_path is None:
            # Execute the conversion function and ensure the result is a string.
            self._usd_path = str(self.converter.convert(**self.kwargs))
        return self._usd_path

    def __repr__(self) -> str:
        """Returns a representation of the lazy loader."""
        return f"LazyUsdPath(converter={self.converter}, kwargs={self.kwargs})"

    def __eq__(self, other: object) -> bool:
        """Checks if two lazy paths are equal by comparing their string representation."""
        if not isinstance(other, LazyUsdPath):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self) -> int:
        """Computes the hash of the lazy path based on its string representation."""
        return hash(str(self))


class UsdConverter:
    """A unified converter for URDF, XACRO, and MJCF files to USD format."""

    def __init__(self, conversion_type: Literal["urdf", "xacro", "mjcf"]):
        """Initialize the converter with the specified conversion type.

        Args:
            conversion_type: The type of conversion to perform ("urdf", "xacro", or "mjcf").
        """
        self.conversion_type = conversion_type

    def convert(
        self,
        file_path: str,
        output_usd_path: str | None = None,
        merge_joints: bool = False,
        fix_base: bool = False,
        joint_stiffness: float = 100.0,
        joint_damping: float = 1.0,
        joint_target_type: Literal["position", "velocity", "none"] = "position",
        import_sites: bool = True,
        make_instanceable: bool = True,
    ) -> str:
        """Convert the input file to USD format.

        Args:
            file_path: The path to the input file.
            output_usd_path: The path for the output USD file. If None, a temporary file is created.
            merge_joints: Whether to merge fixed joints (for URDF/XACRO).
            fix_base: Whether to fix the base of the robot.
            joint_stiffness: Joint drive stiffness (for URDF/XACRO).
            joint_damping: Joint drive damping (for URDF/XACRO).
            joint_target_type: Joint control type (for URDF/XACRO).
            import_sites: Whether to import sites (for MJCF).
            make_instanceable: Whether to make instanceable (for MJCF).

        Returns:
            The path to the generated USD file.

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            ValueError: For invalid parameters.
            RuntimeError: If conversion fails.
        """
        if self.conversion_type == "urdf":
            return self._convert_urdf(
                file_path=file_path,
                output_usd_path=output_usd_path,
                merge_joints=merge_joints,
                fix_base=fix_base,
                joint_stiffness=joint_stiffness,
                joint_damping=joint_damping,
                joint_target_type=joint_target_type,
            )
        elif self.conversion_type == "xacro":
            return self._convert_xacro(
                file_path=file_path,
                output_usd_path=output_usd_path,
                merge_joints=merge_joints,
                fix_base=fix_base,
                joint_stiffness=joint_stiffness,
                joint_damping=joint_damping,
                joint_target_type=joint_target_type,
            )
        elif self.conversion_type == "mjcf":
            return self._convert_mjcf(
                file_path=file_path,
                output_usd_path=output_usd_path,
                fix_base=fix_base,
                import_sites=import_sites,
                make_instanceable=make_instanceable,
            )
        else:
            raise ValueError(f"Unsupported conversion type: {self.conversion_type}")

    @contextlib.contextmanager
    def file_lock(self, lock_path, timeout=60):
        start = time.time()
        # Ensure the parent directory of the lock file exists
        lock_dir = os.path.dirname(lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Timeout waiting for lock on {lock_path}")
                    time.sleep(0.1)
            try:
                yield lock_file
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _convert_urdf(
        self,
        file_path: str,
        output_usd_path: str | None = None,
        merge_joints: bool = False,
        fix_base: bool = False,
        joint_stiffness: float = 100.0,
        joint_damping: float = 1.0,
        joint_target_type: Literal["position", "velocity", "none"] = "position",
    ) -> str:
        """Convert URDF to USD."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"URDF file not found: {file_path}")
        if joint_target_type not in ("position", "velocity", "none"):
            raise ValueError("joint_target_type must be 'position', 'velocity' or 'none'")

        is_temp_file = output_usd_path is None
        usd_path = output_usd_path if not is_temp_file else tempfile.mktemp(suffix=".usd")

        # Ensure the output directory exists (if output path is specified)
        if not is_temp_file and output_usd_path is not None:
            os.makedirs(os.path.dirname(output_usd_path), exist_ok=True)

        lock_path = usd_path + ".lock"
        with self.file_lock(lock_path):
            try:
                converter_cfg = UrdfConverterCfg(
                    asset_path=file_path,
                    usd_dir=os.path.dirname(usd_path),
                    usd_file_name=os.path.basename(usd_path),
                    fix_base=fix_base,
                    merge_fixed_joints=merge_joints,
                    force_usd_conversion=True,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(
                        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                            stiffness=joint_stiffness, damping=joint_damping
                        ),
                        target_type=joint_target_type,
                    ),
                )

                UrdfConverter(converter_cfg)

                if is_temp_file:
                    atexit.register(lambda: os.unlink(usd_path) if os.path.exists(usd_path) else None)

                print(f"USD file generated at: {usd_path}")
                return usd_path

            except Exception as e:
                if is_temp_file and os.path.exists(usd_path):
                    os.unlink(usd_path)
                raise RuntimeError(f"URDF conversion failed: {str(e)}") from e

    def _convert_xacro(
        self,
        file_path: str,
        output_usd_path: str | None = None,
        merge_joints: bool = False,
        fix_base: bool = False,
        joint_stiffness: float = 100.0,
        joint_damping: float = 1.0,
        joint_target_type: Literal["position", "velocity", "none"] = "position",
    ) -> str:
        """Convert XACRO to USD."""
        doc = XacroDoc.from_file(file_path)

        with doc.temp_urdf_file_path() as urdf_file_path:
            return self._convert_urdf(
                file_path=urdf_file_path,
                output_usd_path=output_usd_path,
                merge_joints=merge_joints,
                fix_base=fix_base,
                joint_stiffness=joint_stiffness,
                joint_damping=joint_damping,
                joint_target_type=joint_target_type,
            )

    def _convert_mjcf(
        self,
        file_path: str,
        output_usd_path: str | None = None,
        fix_base: bool = False,
        import_sites: bool = True,
        make_instanceable: bool = True,
    ) -> str:
        """Convert MJCF to USD."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"MJCF file not found: {file_path}")

        is_temp_file = output_usd_path is None
        usd_path = output_usd_path if not is_temp_file else tempfile.mktemp(suffix=".usd")

        # Ensure the output directory exists (if output path is specified)
        if not is_temp_file and output_usd_path is not None:
            os.makedirs(os.path.dirname(output_usd_path), exist_ok=True)

        lock_path = usd_path + ".lock"
        with self.file_lock(lock_path):
            try:
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

                if is_temp_file:
                    atexit.register(lambda: os.unlink(usd_path) if os.path.exists(usd_path) else None)

                print(f"USD file generated at: {usd_path}")
                return usd_path

            except Exception as e:
                if is_temp_file and os.path.exists(usd_path):
                    os.unlink(usd_path)
                raise RuntimeError(f"MJCF conversion failed: {str(e)}") from e


# Convenience functions that maintain backward compatibility
def urdf_to_usd(
    file_path: str,
    output_usd_path: str | None = None,
    merge_joints: bool = False,
    fix_base: bool = False,
    joint_stiffness: float = 100.0,
    joint_damping: float = 1.0,
    joint_target_type: Literal["position", "velocity", "none"] = "position",
    lazy: bool = True,
) -> str | LazyUsdPath:
    """Converts a URDF file to USD, with an option for lazy conversion."""
    converter = UsdConverter("urdf")
    if lazy:
        return LazyUsdPath(
            converter,
            file_path=file_path,
            output_usd_path=output_usd_path,
            merge_joints=merge_joints,
            fix_base=fix_base,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            joint_target_type=joint_target_type,
        )
    else:
        return converter.convert(
            file_path=file_path,
            output_usd_path=output_usd_path,
            merge_joints=merge_joints,
            fix_base=fix_base,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            joint_target_type=joint_target_type,
        )


def xacro_to_usd(
    file_path: str,
    output_usd_path: str | None = None,
    merge_joints: bool = False,
    fix_base: bool = False,
    lazy: bool = True,
) -> str | LazyUsdPath:
    """Converts a XACRO file to USD, with an option for lazy conversion."""
    converter = UsdConverter("xacro")
    if lazy:
        return LazyUsdPath(
            converter,
            file_path=file_path,
            output_usd_path=output_usd_path,
            merge_joints=merge_joints,
            fix_base=fix_base,
        )
    else:
        return converter.convert(
            file_path=file_path,
            output_usd_path=output_usd_path,
            merge_joints=merge_joints,
            fix_base=fix_base,
        )


def mjcf_to_usd(
    file_path: str,
    output_usd_path: str | None = None,
    fix_base: bool = False,
    import_sites: bool = True,
    make_instanceable: bool = True,
    lazy: bool = True,
) -> str | LazyUsdPath:
    """Converts a MJCF file to USD, with an option for lazy conversion."""
    converter = UsdConverter("mjcf")
    if lazy:
        return LazyUsdPath(
            converter,
            file_path=file_path,
            output_usd_path=output_usd_path,
            fix_base=fix_base,
            import_sites=import_sites,
            make_instanceable=make_instanceable,
        )
    else:
        return converter.convert(
            file_path=file_path,
            output_usd_path=output_usd_path,
            fix_base=fix_base,
            import_sites=import_sites,
            make_instanceable=make_instanceable,
        )
