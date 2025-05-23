# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Environment variables."""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    CONCH_BENCH_ENABLE_ALL_REF: bool
    CONCH_ENABLE_BNB: bool
    CONCH_ENABLE_VLLM: bool


environment_variables: dict[str, Callable[[], Any]] = {
    # Enable the vLLM/bitsandbytes reference implementations for all applicable benchmarks
    "CONCH_BENCH_ENABLE_ALL_REF": lambda: (
        os.environ.get("CONCH_BENCH_ENABLE_ALL_REF", "0").strip().lower() in ("1", "true")
    ),
    # Enable bitsandbytes kernels for testing/benchmarking
    "CONCH_ENABLE_BNB": lambda: (os.environ.get("CONCH_ENABLE_BNB", "0").strip().lower() in ("1", "true")),
    # Enable vLLM kernels for testing/benchmarking
    "CONCH_ENABLE_VLLM": lambda: (os.environ.get("CONCH_ENABLE_VLLM", "0").strip().lower() in ("1", "true")),
}


def __getattr__(name: str) -> Any:
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    error_msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(error_msg)


def __dir__() -> list[str]:
    return list(environment_variables.keys())
