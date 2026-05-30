"""Subprocess-isolated import smoke tests.

Each test launches a fresh interpreter with ``ROBOT_LAB_FRAMEWORK`` set to
the target backend, then imports the framework public API + a sampling of
business code. Subprocess isolation is needed because ``framework.detect``
caches the active framework on first call; switching mid-process is not
supported by design.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_PY = REPO_ROOT / ".venvs" / "isaaclab" / "bin" / "python"
MJLAB_PY = REPO_ROOT / ".venvs" / "mjlab" / "bin" / "python"


def _run(interpreter: Path, env_value: str, code: str) -> tuple[int, str]:
    proc = subprocess.run(
        [str(interpreter), "-c", code],
        env={
            "ROBOT_LAB_FRAMEWORK": env_value,
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.skipif(not MJLAB_PY.exists(), reason="mjlab venv not provisioned")
def test_framework_imports_mjlab():
    rc, out = _run(
        MJLAB_PY,
        "mjlab",
        "from robot_lab.framework import EnvCfg, RewardTermCfg, register_task; print('ok')",
    )
    assert rc == 0, out
    assert "ok" in out


@pytest.mark.skipif(not ISAACLAB_PY.exists(), reason="isaaclab venv not provisioned")
def test_framework_imports_isaaclab():
    # Static AST validation is what we can verify without booting SimApp.
    src_root = REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "framework"
    code = (
        f"import ast, pathlib\n"
        f"for p in pathlib.Path({str(src_root)!r}).rglob('*.py'):\n"
        f"    ast.parse(p.read_text())\n"
        f"print('ok')\n"
    )
    rc, out = _run(ISAACLAB_PY, "isaaclab", code)
    assert rc == 0, out
    assert "ok" in out


@pytest.mark.skipif(not MJLAB_PY.exists(), reason="mjlab venv not provisioned")
def test_business_imports_mjlab():
    rc, out = _run(
        MJLAB_PY,
        "mjlab",
        "import robot_lab.tasks; "
        "from robot_lab.tasks.manager_based.locomotion.velocity import mdp; "
        "print(len([n for n in dir(mdp) if not n.startswith('_')]))",
    )
    assert rc == 0, out
    # Should print a positive count of mdp namespace entries.
    last_line = out.strip().splitlines()[-1]
    assert int(last_line) > 0
