# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Inspired by: https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/setup.py

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(
    name: str,
    module: str,
    sources: list[str],
    sources_cuda: list[str] = [],
    extra_args: list[str] = [],
    extra_include_path: list[str] = [],
) -> CUDAExtension:  # type: ignore[valid-type]
    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if (torch.cuda.is_available() and torch.version.cuda is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_90,code=sm_90",
        ]
        sources += sources_cuda
    elif (torch.cuda.is_available() and torch.version.hip is not None) or os.getenv("FORCE_ROCM", "0") == "1":
        define_macros += [("WITH_ROCM", None)]
        extra_compile_args["hipcc"] = extra_args + [
            "-D__HIP_NO_HALF_OPERATORS__",
            "-D__HIP_NO_HALF_CONVERSIONS__",
            "-D__HIP_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda

    return CUDAExtension(  # type: ignore[no-any-return]
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name="conch_cuda_ext",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    license="Apache License 2.0",
    ext_modules=[
        make_cuda_ext(
            name="bev_pool",
            module="ops.vision.bev_pool",
            sources=[
                "bev_pool.cc",
                "bev_pool_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
