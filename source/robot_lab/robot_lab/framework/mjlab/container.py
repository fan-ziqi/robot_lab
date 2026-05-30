"""Manager container factory — mjlab backend.

mjlab managers accept a plain dict of cfg terms. The robot_lab business code
prefers attribute access (``cfg.rewards.term_name.weight = ...``). This module
provides an :class:`AttrDict` that subclasses ``dict`` and adds attribute-style
get/set/del — both forms refer to the same underlying mapping.
"""

from __future__ import annotations

from typing import Any


class AttrDict(dict):
    """Dict subclass with attribute-style access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def manager_container(terms: dict[str, Any]) -> AttrDict:
    """Wrap a dict of cfg terms in an attribute-accessible dict for mjlab managers."""
    return AttrDict(terms)
