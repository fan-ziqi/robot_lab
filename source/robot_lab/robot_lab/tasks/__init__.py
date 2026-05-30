# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

Task registration goes through Isaac Lab's ``import_packages`` helper, which is
only available when the IsaacLab subpackages and Isaac Sim are importable. On a
mjlab-only venv, or in a plain Python interpreter before ``isaacsim`` boots,
the import below fails and we skip registration. The dual-framework migration
(Phase 5 / Task 5.1) rewrites this module to register tasks through the
``robot_lab.framework`` adapter layer instead.
"""

import logging as _logging

_log = _logging.getLogger(__name__)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]

try:
    import os  # noqa: F401  - kept for downstream code that may rely on it
    import toml  # noqa: F401

    from isaaclab_tasks.utils import import_packages

    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)
except ImportError as _exc:  # pragma: no cover - exercised under mjlab/no-sim
    _log.debug("robot_lab.tasks registration skipped: %s", _exc)
