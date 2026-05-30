"""Integration smoke tests — opt-in via ``pytest -m smoke``.

These tests boot the simulator and run a few env steps. They're gated behind
the ``smoke`` marker because:
- IsaacLab side requires Isaac Sim (and its py3.11 SimApp bootstrap currently
  has a known compat bug, see Task 9.6 spec).
- mjlab side requires GPU + warp. Even on CPU it loads MuJoCo.

CI runs ``pytest`` without ``-m smoke`` so these tests skip by default.
Run locally with ``pytest -m smoke`` (or via Task 9.6 training run).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_PY = REPO_ROOT / ".venvs" / "isaaclab" / "bin" / "python"
MJLAB_PY = REPO_ROOT / ".venvs" / "mjlab" / "bin" / "python"

pytestmark = pytest.mark.smoke


def _run(interpreter: Path, env_value: str, code: str, timeout: int = 600) -> tuple[int, str]:
    proc = subprocess.run(
        [str(interpreter), "-c", code],
        env={
            "ROBOT_LAB_FRAMEWORK": env_value,
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "DISPLAY": os.environ.get("DISPLAY", ""),
        },
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.skipif(not MJLAB_PY.exists(), reason="mjlab venv not provisioned")
def test_smoke_mjlab_list_tasks():
    """All robot_lab tasks register cleanly (or skip-with-info) under mjlab."""
    code = textwrap.dedent(
        """
        import os
        os.environ["ROBOT_LAB_FRAMEWORK"] = "mjlab"
        import robot_lab.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks
        # All current robot_lab tasks are framework_required='isaaclab', so
        # mjlab side has nothing registered. The assertion is just that the
        # import + registration completed without raising.
        names = sorted(list_tasks())
        print(f"REGISTERED={len(names)}")
        """
    )
    rc, out = _run(MJLAB_PY, "mjlab", code, timeout=180)
    assert rc == 0, out
    assert "REGISTERED=" in out


@pytest.mark.skipif(not ISAACLAB_PY.exists(), reason="isaaclab venv not provisioned")
@pytest.mark.skip(reason="IsaacSim 5.1 + Python 3.11 SimApp bootstrap is broken on this host")
def test_smoke_isaaclab_unitree_a1_rough():
    """Skipped pending isaacsim 5.1+py3.11 bootstrap fix; see Task 9.6."""
    pass
