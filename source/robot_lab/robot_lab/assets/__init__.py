# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Paths and configurations for project-owned assets."""

from pathlib import Path

ROBOT_LAB_ASSETS_DIR = Path(__file__).resolve().parent / "data"
"""Path to project-owned asset data, in an editable checkout or an installed wheel."""

# Compatibility for asset configurations written against robot_lab 2.x.
ISAACLAB_ASSETS_DATA_DIR = str(ROBOT_LAB_ASSETS_DIR)
