"""Observation group factory — mjlab.

mjlab's ``ObservationGroupCfg`` accepts a ``terms`` dict directly. We wrap the
terms dict in :class:`AttrDict` so attribute-style access works downstream
(the business code treats group.terms.<name> uniformly across backends).
"""

from __future__ import annotations

from typing import Any

from mjlab.managers.observation_manager import ObservationGroupCfg

from robot_lab.framework.mjlab.container import AttrDict


def make_observation_group(
    terms: dict[str, Any],
    *,
    concatenate_terms: bool = True,
    enable_corruption: bool = False,
) -> ObservationGroupCfg:
    return ObservationGroupCfg(
        terms=AttrDict(terms),
        concatenate_terms=concatenate_terms,
        enable_corruption=enable_corruption,
    )


__all__ = ["make_observation_group"]
