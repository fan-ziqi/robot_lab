"""Simulator launcher hook — mjlab (no-op)."""

from __future__ import annotations

from typing import Any


def launch_app(args: Any) -> None:
    """mjlab does not require a separate app launch."""
    del args
    return None


__all__ = ["launch_app"]
