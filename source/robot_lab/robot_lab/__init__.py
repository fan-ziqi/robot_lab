# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Python module serving as a project/extension template.
"""

# Task and UI registration both depend on the active simulator (IsaacLab) being
# initialized. Under the mjlab backend, or in a generic Python interpreter
# before IsaacSim's SimulationApp has booted, those imports raise. We swallow
# the failure so that ``import robot_lab`` itself succeeds and so that the
# framework adapter layer (``robot_lab.framework``) is reachable. The actual
# task loader is rewritten in Phase 5 of the dual-framework migration.
import logging as _logging

_log = _logging.getLogger(__name__)

try:
    from .tasks import *  # noqa: F401,F403  (registers gym tasks)
except ImportError as _exc:  # pragma: no cover - exercised under mjlab/no-sim
    _log.debug("robot_lab.tasks not loaded yet: %s", _exc)

try:
    from .ui_extension_example import *  # noqa: F401,F403
except ImportError as _exc:  # pragma: no cover - same path
    _log.debug("robot_lab.ui_extension_example not loaded yet: %s", _exc)
