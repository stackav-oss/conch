# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

from typing import Final

from setuptools import setup

_TORCH_VERSION: Final = "2.6.0"

_REQUIREMENTS: Final = [
    "numpy>=1.26.4",
]

# For CPU:
# --extra-index-url https://download.pytorch.org/whl/cpu
_DEFAULT_PLATFORM_REQUIREMENTS: Final = [
    f"torch=={_TORCH_VERSION}",
    "triton>=3.1.0",
]

# For ROCm:
# --extra-index-url https://download.pytorch.org/whl/rocm6.2.4
_ROCM_PLATFORM_REQUIREMENTS: Final = [
    f"torch=={_TORCH_VERSION}+rocm6.2.4",
    "pytorch-triton-rocm>=3.1.0",
]

# For XPU:
# --extra-index-url https://download.pytorch.org/whl/xpu
_XPU_PLATFORM_REQUIREMENTS: Final = [
    f"torch=={_TORCH_VERSION}+xpu",
    "pytorch-triton-xpu>=3.2.0",
]

_PLATFORM_REQUIREMENTS: Final = {
    # --extra-index-url https://download.pytorch.org/whl/cpu
    "cpu": _DEFAULT_PLATFORM_REQUIREMENTS,
    # No extra index URL needed!
    "cuda": _DEFAULT_PLATFORM_REQUIREMENTS,
    # --extra-index-url https://download.pytorch.org/whl/rocm6.2.4
    "rocm": _ROCM_PLATFORM_REQUIREMENTS,
    # --extra-index-url https://download.pytorch.org/whl/xpu
    "xpu": _XPU_PLATFORM_REQUIREMENTS,
}


def get_default_dependencies() -> list[str]:
    """Determine the appropriate dependencies based on detected hardware."""
    return _REQUIREMENTS


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
    } | _PLATFORM_REQUIREMENTS


setup(  # type: ignore[no-untyped-call]
    name="conch-triton-kernels",
    install_requires=get_default_dependencies(),
    extras_require=get_optional_dependencies(),
    setup_requires=["wheel"],
)
