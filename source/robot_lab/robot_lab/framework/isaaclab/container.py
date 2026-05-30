"""Manager container factory — IsaacLab backend.

IsaacLab managers (RewardManager, ObservationManager, ...) expect their cfg
input to be an attribute-bearing object whose attributes are cfg term
instances (historically a ``@configclass``-decorated class instance). This
helper wraps a plain dict-of-terms into such an object so business code can
stay framework-neutral.
"""

from __future__ import annotations

from typing import Any

from isaaclab.utils import configclass


def manager_container(terms: dict[str, Any]) -> Any:
    """Build an attribute-style container that IsaacLab managers accept."""

    @configclass
    class _AnonContainer:
        pass

    inst = _AnonContainer()
    for name, term in terms.items():
        setattr(inst, name, term)
    return inst
