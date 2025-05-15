# Installation

## As editable

First, clone the Conch repo.

```bash
git clone https://github.com/stackav-oss/conch.git
cd conch
```

then install:

```bash
pip install -e .
```

## As wheel

You can install `conch` from PyPi via:

```bash
pip install conch-triton-kernels
```

**Note**: by default, without any extras specified, **this will not install `torch` or `triton`**.
This allows usage of Conch as long as Torch and Triton are already installed in your environment.

### Nvidia/CUDA

For Nvidia/CUDA platforms, you can specify the `[cuda]` extra to install `torch` and `triton` for Nvidia/CUDA platforms.

```bash
pip install "conch-triton-kernels[cuda]"
```

### AMD/ROCm

For AMD/ROCm platforms, you can specify the `[rocm]` extra to install `torch` and `triton` for AMD/ROCm platforms.
You must also specify the appropriate `--extra-index-url`.

```bash
pip install "conch-triton-kernels[rocm]" --extra-index-url https://download.pytorch.org/whl/rocm6.3
```
