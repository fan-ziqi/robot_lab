"""Tests for mjlab manager_container (AttrDict)."""

from __future__ import annotations

from robot_lab.framework.mjlab.container import AttrDict, manager_container


def test_returns_attrdict():
    container = manager_container({"a": 1})
    assert isinstance(container, AttrDict)
    assert isinstance(container, dict)


def test_attribute_access_reads_from_dict():
    class Term:
        def __init__(self, w):
            self.weight = w

    container = manager_container({"reward_a": Term(1.0)})
    assert container.reward_a.weight == 1.0
    assert container["reward_a"].weight == 1.0


def test_iter_as_dict():
    """mjlab managers iterate the cfg as a dict."""
    class Term:
        pass
    container = manager_container({"a": Term(), "b": Term()})
    assert set(container.keys()) == {"a", "b"}
    for name, term in container.items():
        assert name in {"a", "b"}


def test_pop_works():
    class Term:
        pass
    container = manager_container({"push_robot": Term()})
    container.pop("push_robot", None)
    assert "push_robot" not in container


def test_attribute_set_creates_entry():
    class Term:
        def __init__(self, w):
            self.weight = w
    container = manager_container({})
    container.new = Term(42.0)
    assert container["new"].weight == 42.0


def test_attribute_delete():
    class Term:
        pass
    container = manager_container({"x": Term()})
    del container.x
    assert "x" not in container


def test_attribute_missing_raises_attribute_error():
    container = manager_container({})
    try:
        _ = container.nope
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")
