"""AST-based test: business code (tasks/, assets/) must not import isaaclab/mjlab directly.

Per spec §3.1 Rule 1, the framework adapter (``robot_lab.framework``) is the
ONLY place those imports are allowed. This test parses every Python file
under ``tasks/`` and ``assets/`` and flags direct top-level imports of the
forbidden prefixes.

Exempted contexts (the import IS allowed when wrapped in any of these):
- ``if TYPE_CHECKING:`` blocks (type hints only).
- ``try`` blocks (transitional shim — file needs to load under both venvs
  while migration is incremental).
- ``def`` / ``class`` bodies (lazy imports inside dispatch helpers like
  ``build_<robot>_cfg()`` are intentional).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUSINESS_DIRS = [
    REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "tasks",
    REPO_ROOT / "source" / "robot_lab" / "robot_lab" / "assets",
]
# Adapter-layer files that LIVE under business dirs but are part of the
# framework backend bridge — they MAY import isaaclab/mjlab directly.
ADAPTER_PATH_FRAGMENTS = (
    "/assets/builders/",  # builders/isaaclab.py and builders/mjlab.py
    # IL-only branches (lazy-loaded by IL register_task or only used under
    # IsaacLab venv): agent cfg files, the direct/g1_amp task, and any
    # task entry whose framework_required='isaaclab' guard means the file
    # only loads under IL.
    "/agents/",
    "/tasks/direct/",
    # Per-robot env_cfg files inherit the IL-style nested @configclass base
    # in velocity_env_cfg.py (the file itself is wrapped in try/except).
    # The subclasses are IL-only by design until Phase 7+ rewrites them as
    # mjlab-native factories.
    "/config/internal/",
    "/config/quadruped/",
    "/config/wheeled/",
    "/config/humanoid/",
    "/config/others/",
    "/beyondmimic/config/",
    "/loco_manipulation/tracking/config/",
    # IL-only scratch retained for oppo_v1 etc. (per spec §9.6).
    "/locomotion/velocity/_velocity_env_cfg.py",
    "/locomotion/velocity/_mdp/",
    # beyondmimic + loco_manipulation env / mdp packages remain IL-only;
    # they're loaded only via lazy entry-point strings on the IL backend.
    "/beyondmimic/tracking_env_cfg.py",
    "/beyondmimic/mdp/",
    "/loco_manipulation/tracking/tracking_env_cfg.py",
    "/loco_manipulation/tracking/mdp/",
)
FORBIDDEN_PREFIXES = (
    "isaaclab",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_assets",
    "mjlab",
)


def _is_inside(parents: list[ast.AST], target_types: tuple[type, ...]) -> bool:
    """Return True if any ancestor is one of the target node types."""
    return any(isinstance(p, target_types) for p in parents)


def _is_in_type_checking_block(parents: list[ast.AST]) -> bool:
    """True if the import sits inside `if TYPE_CHECKING:` body."""
    for p in parents:
        if isinstance(p, ast.If):
            test = p.test
            # Match `if TYPE_CHECKING:` or `if typing.TYPE_CHECKING:`.
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return True
            if (
                isinstance(test, ast.Attribute)
                and test.attr == "TYPE_CHECKING"
            ):
                return True
    return False


def _scan_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    offenders: list[str] = []

    # Walk with parent tracking.
    stack: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        stack.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)
        stack.pop()
        # Process current node (post-order so parents are still on stack)
        # Actually we need to check at visit time.

    def walk(node: ast.AST, parents: list[ast.AST]) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Allow imports inside try, if-TYPE_CHECKING, function, or class bodies.
            if (
                _is_inside(parents, (ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                or _is_in_type_checking_block(parents)
            ):
                return
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in FORBIDDEN_PREFIXES:
                        offenders.append(f"line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    return
                top = node.module.split(".", 1)[0]
                if top in FORBIDDEN_PREFIXES:
                    offenders.append(f"line {node.lineno}: from {node.module} import ...")

        for child in ast.iter_child_nodes(node):
            walk(child, parents + [node])

    walk(tree, [])
    return offenders


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in BUSINESS_DIRS:
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            posix = p.as_posix()
            if any(frag in posix for frag in ADAPTER_PATH_FRAGMENTS):
                continue
            files.append(p)
    return files


_ALL_FILES = _iter_python_files()


@pytest.mark.parametrize("path", _ALL_FILES, ids=[str(p.relative_to(REPO_ROOT)) for p in _ALL_FILES])
def test_no_direct_framework_import(path: Path):
    offenders = _scan_file(path)
    assert not offenders, (
        f"\n{path.relative_to(REPO_ROOT)} imports forbidden modules outside "
        f"try/TYPE_CHECKING blocks:\n  " + "\n  ".join(offenders)
    )
