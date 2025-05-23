# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from conch.platforms.platform import Platform, PlatformEnum, detect_current_platform

_current_platform = None

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str) -> Any:
    if name == "current_platform":
        global _current_platform  # noqa: PLW0603
        if _current_platform is None:
            _current_platform = detect_current_platform()
        return _current_platform

    if name in globals():
        return globals()[name]

    error_msg = f"No attribute named '{name}' exists in {__name__}."
    raise AttributeError(error_msg)


__all__ = [
    "Platform",
    "PlatformEnum",
    "current_platform",
    "detect_current_platform",
]
