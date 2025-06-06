# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Platform enum."""

import enum
from dataclasses import dataclass
from typing import Final

import torch


class PlatformEnum(enum.Enum):
    """Enum for different platforms."""

    NVIDIA = enum.auto()
    AMD = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    UNSPECIFIED = enum.auto()


@dataclass
class Platform:
    """Class representing a platform."""

    platform_enum: PlatformEnum
    device: str

    def name(self) -> str:
        """Return the name of the platform."""
        return self.platform_enum.name

    def is_nvidia(self) -> bool:
        """Check if the platform is NVIDIA."""
        return self.platform_enum == PlatformEnum.NVIDIA

    def is_amd(self) -> bool:
        """Check if the platform is AMD."""
        return self.platform_enum == PlatformEnum.AMD

    def is_unspecified(self) -> bool:
        """Check if the platform is unspecified."""
        return self.platform_enum == PlatformEnum.UNSPECIFIED

    def has_cuda(self) -> bool:
        """Check if the platform has CUDA support."""
        return self.is_nvidia() or self.is_amd()

    def supports_fp8(self) -> bool:
        """Check if the platform supports FP8."""
        if torch.cuda.is_available():
            # Triton: fp8e4nv data type is not supported on CUDA arch < 89
            min_fp8_capability_triton: Final = 89

            capability = torch.cuda.get_device_capability()
            actual_capability = capability[0] * 10 + capability[1]
            return actual_capability >= min_fp8_capability_triton

        # Until we officially support XPU/CPU, just assume True so that we can uncover any issues
        return True

    def get_device_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()

        return "unknown"


def detect_current_platform() -> Platform:
    """Detect the current platform."""
    # We could do much more sophisticated things to detect the platform,
    # but this is an easy first-cut
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if torch.version.cuda is not None:
            return Platform(PlatformEnum.NVIDIA, "cuda")

        if torch.version.hip is not None:
            return Platform(PlatformEnum.AMD, "cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return Platform(PlatformEnum.XPU, "xpu")

    if hasattr(torch, "cpu") and torch.cpu.is_available():
        return Platform(PlatformEnum.CPU, "cpu")

    return Platform(PlatformEnum.UNSPECIFIED, "cpu")
