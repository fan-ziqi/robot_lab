"""Tests for IsaacLab manager_container.

We can't import ``isaaclab.utils.configclass`` directly in a generic Python
process — IsaacLab's ``utils/__init__.py`` pulls in USD bindings (``pxr``) that
require a live IsaacSim kernel. We monkeypatch ``configclass`` with a stdlib
``dataclasses.dataclass`` substitute and verify the dict-of-terms wrapping
behavior. The actual configclass code path is exercised at runtime by the
smoke training task (Task 9.6).
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def manager_container_with_stub(monkeypatch):
    """Inject a stub ``isaaclab.utils.configclass`` and import ``manager_container``."""
    # Build a fake isaaclab.utils module that exposes a configclass identical
    # in surface (decorator returning the class).
    fake_utils = types.ModuleType("isaaclab.utils")
    fake_utils.configclass = lambda cls: cls  # noqa: E731

    fake_il = types.ModuleType("isaaclab")
    fake_il.utils = fake_utils

    monkeypatch.setitem(sys.modules, "isaaclab", fake_il)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", fake_utils)

    # Reload container module so it re-binds configclass.
    sys.modules.pop("robot_lab.framework.isaaclab.container", None)
    from robot_lab.framework.isaaclab.container import manager_container
    return manager_container


def test_returns_object_with_setattr_terms(manager_container_with_stub):
    """Each dict key becomes an attribute on the returned instance."""
    class Term:
        def __init__(self, weight):
            self.weight = weight

    container = manager_container_with_stub({"a": Term(1.0), "b": Term(-2.0)})
    assert container.a.weight == 1.0
    assert container.b.weight == -2.0


def test_empty_dict_returns_empty_container(manager_container_with_stub):
    container = manager_container_with_stub({})
    # No raised error, no spurious attributes.
    assert not [a for a in vars(container) if not a.startswith("_")]


def test_attributes_mutable(manager_container_with_stub):
    """Returned container must allow IsaacLab-style ``cfg.term.weight = 3.0`` mutation."""
    class Term:
        weight = 0.0

    container = manager_container_with_stub({"r": Term()})
    container.r.weight = 3.0
    assert container.r.weight == 3.0
