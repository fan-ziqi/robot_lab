"""Framework-required guard types."""

from __future__ import annotations

from typing import Literal

from robot_lab.framework import detect

FrameworkRequired = Literal["isaaclab", "mjlab", "any"]
_VALID: tuple[FrameworkRequired, ...] = ("isaaclab", "mjlab", "any")


def current_framework_supports(required: FrameworkRequired) -> bool:
    if required not in _VALID:
        raise ValueError(f"Invalid framework_required={required!r}; expected one of {_VALID}")
    if required == "any":
        return True
    return detect.get_framework() == required


class FrameworkRequiredError(RuntimeError):
    """Raised when business code asks for a symbol the active backend does not support."""

    def __init__(self, symbol: str, required: str, current: str) -> None:
        self.symbol = symbol
        self.required = required
        self.current = current
        super().__init__(
            f"{symbol!r} requires framework={required!r}, but current framework is {current!r}."
        )
