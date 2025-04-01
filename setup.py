import subprocess
from setuptools import setup
from typing import Final, Literal

_REQUIREMENTS: Final = [
    "numpy>=1.26.4",
]

_DEFAULT_PLATFORM_REQUIREMENTS: Final = [
    "torch>=2.5.1",
    "triton>=3.1.0",
]

# --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.2
_ROCM_PLATFORM_REQUIREMENTS: Final = [
    "amdsmi==6.2.2.post0",
    "torch==2.5.1+rocm6.2",
    "pytorch-triton-rocm>=3.1.0",
]

_XPU_PLATFORM_REQUIREMENTS: Final = [
    "torch>=2.6.0",
    "pytorch-triton-xpu>=3.2.0",
]

_PLATFORM_REQUIREMENTS: Final = {
    "cpu": _DEFAULT_PLATFORM_REQUIREMENTS,
    "cuda": _DEFAULT_PLATFORM_REQUIREMENTS,
    "rocm": _ROCM_PLATFORM_REQUIREMENTS,
    "xpu": _XPU_PLATFORM_REQUIREMENTS,
}


def get_default_dependencies():
    """Determine the appropriate dependencies based on detected hardware."""
    platform = get_platform()
    return _REQUIREMENTS + _PLATFORM_REQUIREMENTS[platform]


def get_optional_dependencies():
    """Get optional dependency groups."""
    return {
        "dev": [
            "click>=8.1.8",
            "einops>=0.8.0",
            "pytest>=8.3.4",
            "ruff>=0.4.10",
        ],
    }


def get_platform() -> Literal["cuda", "rocm", "cpu", "xpu"]:
    """
    Detect whether the system has NVIDIA or AMD GPU without torch dependency.
    """
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


setup(
    name="conch",
    install_requires=get_default_dependencies(),
    extras_require=get_optional_dependencies(),
    setup_requires=["wheel"],
)
