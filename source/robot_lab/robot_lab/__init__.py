# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Robot Lab task registration without a Kit UI dependency."""

from isaaclab.utils.version import standalone_importers_available

# The standalone importer registers its USD schemas before the first stage is created.
if standalone_importers_available():
    import isaacsim.asset.importer.urdf  # noqa: F401

from . import tasks  # noqa: F401
