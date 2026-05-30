"""Observation group factory — IsaacLab.

IsaacLab's ``ObservationGroupCfg`` is a class users subclass (typically inside
a nested ``@configclass``); attributes are the terms. We build such a class on
the fly per call so business code can use the same dict-of-terms idiom across
backends.
"""

from __future__ import annotations

from typing import Any

from isaaclab.managers import ObservationGroupCfg
from isaaclab.utils import configclass


def make_observation_group(
    terms: dict[str, Any],
    *,
    concatenate_terms: bool = True,
    enable_corruption: bool = False,
) -> ObservationGroupCfg:
    @configclass
    class _Group(ObservationGroupCfg):
        def __post_init__(self):
            self.concatenate_terms = concatenate_terms
            self.enable_corruption = enable_corruption

    inst = _Group()
    for name, term in terms.items():
        setattr(inst, name, term)
    return inst


__all__ = ["make_observation_group"]
