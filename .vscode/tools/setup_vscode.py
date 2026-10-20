# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the upstream editor setup utility."""

import argparse
from pathlib import Path

from isaaclab.utils.editor import setup_editor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac_path", help="Optional Isaac Sim binaries installation.")
    args = parser.parse_args()
    setup_editor(Path(__file__).resolve().parents[2], args.isaac_path)
