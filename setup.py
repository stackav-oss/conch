# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

import os
import subprocess
from typing import Final, Literal

from setuptools import setup

_REQUIREMENTS: Final = [
    "numpy>=1.26.4",
]

_DEFAULT_PLATFORM_REQUIREMENTS: Final = [
    "torch>=2.7.0",
    "triton>=3.1.0",
]

# --extra-index-url https://download.pytorch.org/whl/rocm6.3
_ROCM_PLATFORM_REQUIREMENTS: Final = [
    "torch==2.7.0+rocm6.3",
    "pytorch-triton-rocm>=3.1.0",
]

_XPU_PLATFORM_REQUIREMENTS: Final = [
    "torch>=2.7.0",
    "pytorch-triton-xpu>=3.2.0",
]

_PLATFORM_REQUIREMENTS: Final = {
    "cpu": _DEFAULT_PLATFORM_REQUIREMENTS,
    "cuda": _DEFAULT_PLATFORM_REQUIREMENTS,
    "rocm": _ROCM_PLATFORM_REQUIREMENTS,
    "xpu": _XPU_PLATFORM_REQUIREMENTS,
}


def get_default_dependencies() -> list[str]:
    """Determine the appropriate dependencies based on detected hardware."""
    platform = get_platform()
    return _REQUIREMENTS + _PLATFORM_REQUIREMENTS[platform]


def get_optional_dependencies() -> dict[str, list[str]]:
    """Get optional dependency groups."""
    return {
        "dev": [
            "click>=8.1.8",
            "coverage>=7.8.0",
            "einops>=0.8.0",
            "matplotlib>=3.10.1",
            "mypy>=1.15.0",
            "pandas>=2.2.3",
            "pre-commit>=4.2.0",
            "pytest>=8.3.4",
            "ruff>=0.4.10",
        ],
    }


def get_platform() -> str | Literal["cuda", "rocm", "cpu", "xpu"]:
    """
    Detect whether the system has NVIDIA or AMD GPU without torch dependency.
    """
    if (override_platform := os.environ.get("CONCH_WHEEL_BUILD_PLATFORM", None)) is not None:
        override_platform = override_platform.strip().lower()
        print(f"Overriding platform to {override_platform}")
        return override_platform

    try:
        subprocess.run(["nvidia-smi"], check=True)
        print("NVIDIA GPU detected")
        return "cuda"
    except (subprocess.SubprocessError, FileNotFoundError):
        print("NVIDIA GPU *NOT* detected...")

    try:
        subprocess.run(["rocm-smi"], check=True)
        print("AMD GPU detected")
        return "rocm"
    except (subprocess.SubprocessError, FileNotFoundError):
        print("AMD GPU *NOT* detected...")

    print("Warning: XPU/CPU support is currently experimental in conch!")

    try:
        subprocess.run(["xpu-smi"], check=True)
        print("Intel GPU detected")
        return "xpu"
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Intel GPU *NOT* detected...")

    print("No GPU detected, defaulting to CPU backend")
    return "cpu"


setup(  # type: ignore[no-untyped-call]
    name="conch-triton-kernels",
    install_requires=get_default_dependencies(),
    extras_require=get_optional_dependencies(),
    setup_requires=["wheel"],
)
