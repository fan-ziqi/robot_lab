"""Tests for FrameworkRequired predicate."""

import pytest

from robot_lab.framework import detect, spec


def _set_fw(monkeypatch, name):
    monkeypatch.setenv(detect.FRAMEWORK_ENV_VAR, name)
    detect._reset_cache_for_tests()


def test_any_supported_everywhere(monkeypatch):
    _set_fw(monkeypatch, "isaaclab")
    assert spec.current_framework_supports("any") is True
    _set_fw(monkeypatch, "mjlab")
    assert spec.current_framework_supports("any") is True


def test_isaaclab_required(monkeypatch):
    _set_fw(monkeypatch, "isaaclab")
    assert spec.current_framework_supports("isaaclab") is True
    _set_fw(monkeypatch, "mjlab")
    assert spec.current_framework_supports("isaaclab") is False


def test_invalid_required_raises(monkeypatch):
    _set_fw(monkeypatch, "isaaclab")
    with pytest.raises(ValueError):
        spec.current_framework_supports("bogus")  # type: ignore[arg-type]


def test_framework_required_error_message():
    err = spec.FrameworkRequiredError(
        symbol="randomize_geom_friction",
        required="mjlab",
        current="isaaclab",
    )
    msg = str(err)
    assert "randomize_geom_friction" in msg
    assert "mjlab" in msg
    assert "isaaclab" in msg
