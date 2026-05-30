"""Backend detection for the dual-framework adapter layer.

Reads the ``ROBOT_LAB_FRAMEWORK`` env var on first call. The user MUST set it
explicitly (no implicit default) -- both backends are first-class and choosing
one silently would surprise users.

A programmatic ``set_framework()`` is allowed only before the first
``get_framework()``; after that, the choice is locked for the process. This is
how the dispatcher and tests select the backend.
"""

from __future__ import annotations

import os
from typing import Literal  # noqa: F401  -- forward-compat: spec.py extends this module with FrameworkRequired = Literal[...]

FRAMEWORK_ENV_VAR = "ROBOT_LAB_FRAMEWORK"
ALLOWED: tuple[str, ...] = ("isaaclab", "mjlab")

_cached: str | None = None
_override: str | None = None


def get_framework() -> str:
    global _cached
    if _cached is not None:
        return _cached
    value = _override or os.environ.get(FRAMEWORK_ENV_VAR)
    if value is None:
        raise RuntimeError(
            f"{FRAMEWORK_ENV_VAR} is not set. Set it to one of {ALLOWED}, "
            "or pass --framework {isaaclab,mjlab} to scripts/train.py / scripts/play.py."
        )
    if value not in ALLOWED:
        raise ValueError(
            f"Invalid {FRAMEWORK_ENV_VAR}={value!r}; expected one of {ALLOWED}"
        )
    _cached = value
    return _cached


def set_framework(name: str) -> None:
    """Override the framework before any get_framework() call. For tests/dispatcher only."""
    global _override
    if _cached is not None:
        raise RuntimeError(
            "Framework already resolved; set_framework() must run before first get_framework()."
        )
    if name not in ALLOWED:
        raise ValueError(f"Invalid framework {name!r}; expected one of {ALLOWED}")
    _override = name


def _reset_cache_for_tests() -> None:
    """Internal: reset cached state. ONLY for tests."""
    global _cached, _override
    _cached = None
    _override = None
