"""Top-level play dispatcher. See scripts/train.py for the dispatch contract."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _venv_python(framework: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / ".venvs" / framework / "bin" / "python"


def _maybe_exec_in_venv(framework: str) -> None:
    target = _venv_python(framework)
    if not target.exists():
        print(
            f"[robot_lab] venv not found at {target}.\n"
            f"[robot_lab] Run: ./setup_env.sh {framework}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        if Path(sys.executable).resolve() == target.resolve():
            return
    except OSError:
        pass
    os.execv(str(target), [str(target), *sys.argv])


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--framework",
        choices=["isaaclab", "mjlab"],
        required=True,
        help="Simulator backend to use (required; no default).",
    )
    pre.add_argument(
        "--rl",
        choices=["rsl_rl", "skrl", "cusrl"],
        default="rsl_rl",
        help="RL library to use (the matching framework must support it).",
    )
    args, rest = pre.parse_known_args()

    _maybe_exec_in_venv(args.framework)
    os.environ["ROBOT_LAB_FRAMEWORK"] = args.framework
    sys.argv = [sys.argv[0]] + rest

    # Make 'scripts.<fw>.<rl>.play' importable when invoked via the dispatcher.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_path = f"scripts.{args.framework}.reinforcement_learning.{args.rl}.play"
    mod = importlib.import_module(module_path)
    if hasattr(mod, "main"):
        mod.main()
    elif hasattr(mod, "cli_entry"):
        mod.cli_entry()
    else:
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Cannot resolve {module_path}")
        runpy.run_path(spec.origin, run_name="__main__")


if __name__ == "__main__":
    main()
