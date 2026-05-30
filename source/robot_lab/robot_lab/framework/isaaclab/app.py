"""Simulator launcher hook — IsaacLab."""

from __future__ import annotations

from typing import Any

from isaaclab.app import AppLauncher


def launch_app(args: Any) -> Any:
    """Launch the Omniverse app and return the simulation_app handle."""
    return AppLauncher(args)


__all__ = ["launch_app"]
