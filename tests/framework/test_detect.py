"""Tests for framework backend detection."""

import os

import pytest

from robot_lab.framework import detect


def test_unset_raises(monkeypatch):
    monkeypatch.delenv(detect.FRAMEWORK_ENV_VAR, raising=False)
    detect._reset_cache_for_tests()
    with pytest.raises(RuntimeError, match=detect.FRAMEWORK_ENV_VAR):
        detect.get_framework()


def test_env_var_isaaclab(monkeypatch):
    monkeypatch.setenv(detect.FRAMEWORK_ENV_VAR, "isaaclab")
    detect._reset_cache_for_tests()
    assert detect.get_framework() == "isaaclab"


def test_env_var_mjlab(monkeypatch):
    monkeypatch.setenv(detect.FRAMEWORK_ENV_VAR, "mjlab")
    detect._reset_cache_for_tests()
    assert detect.get_framework() == "mjlab"


def test_invalid_value_raises(monkeypatch):
    monkeypatch.setenv(detect.FRAMEWORK_ENV_VAR, "bogus")
    detect._reset_cache_for_tests()
    with pytest.raises(ValueError, match="bogus"):
        detect.get_framework()


def test_set_framework_before_first_get(monkeypatch):
    monkeypatch.delenv(detect.FRAMEWORK_ENV_VAR, raising=False)
    detect._reset_cache_for_tests()
    detect.set_framework("mjlab")
    assert detect.get_framework() == "mjlab"


def test_set_framework_after_resolution_raises(monkeypatch):
    monkeypatch.setenv(detect.FRAMEWORK_ENV_VAR, "isaaclab")
    detect._reset_cache_for_tests()
    detect.get_framework()  # caches
    with pytest.raises(RuntimeError, match="already resolved"):
        detect.set_framework("mjlab")
