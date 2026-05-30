# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the locomotion-velocity task.

This module exposes a stable namespace by combining:

- upstream MDP functions resolved via the active framework
  (``robot_lab.framework.mdp``);
- robot_lab-specific MDP terms defined under this package.

The local files (``commands.py``, ``curriculums.py``, ``events.py``,
``observations.py``, ``rewards.py``, ``utils.py``) currently still import
``isaaclab.*`` directly. They work as-is under the IsaacLab backend; under
mjlab, the import would fail. We wrap the local re-exports in try/ImportError
so ``mdp/__init__.py`` itself imports cleanly under either backend, but tasks
that reference an IsaacLab-only local term (e.g. ``mdp.GaitReward``) will get
``AttributeError`` at construction time on mjlab — which is the right
behavior for terms that have not been ported yet (Task 5.2/5.3 ports them
incrementally).
"""

from __future__ import annotations

import logging as _logging

# Upstream MDP — resolved per active framework (audit Table 2). The framework
# exposes the active backend's mdp_aliases module under ``robot_lab.framework.mdp``;
# we pull each name explicitly via ``getattr`` to avoid ``from x import *`` on
# a non-module attribute.
from robot_lab.framework import mdp as _fw_mdp

for _name in _fw_mdp.__all__:
    globals()[_name] = getattr(_fw_mdp, _name)
del _fw_mdp, _name

_log = _logging.getLogger(__name__)

# robot_lab-local MDP additions. Each file may reference framework-mdp symbols
# that only the IL backend exposes (e.g. ``mdp.UniformVelocityCommand``,
# ``mdp.joint_deviation_l1``); on mjlab the import / class-body lookup fails.
# Treat both ImportError (missing module) and AttributeError (missing alias on
# the namespace) as "skip silently" — see module docstring.
for _mod in ("commands", "curriculums", "events", "observations", "rewards", "utils"):
    try:
        exec(f"from .{_mod} import *", globals())  # noqa: S102 - controlled, fixed list
    except (ImportError, AttributeError) as _exc:  # pragma: no cover - exercised under mjlab
        _log.debug("velocity.mdp.%s skipped: %s", _mod, _exc)
