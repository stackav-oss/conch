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

### Nvidia/CUDA

For Nvidia/CUDA platforms you can install `conch` from PyPi via:

```bash
pip install conch-triton-kernels
```

### AMD/ROCm

For AMD/ROCm, we do not currently have a wheel on PyPi, but you can easily build one.
After cloning the Conch repo, run this command from the repository root:

```bash
./scripts/wheel/build.sh rocm
```

The resulting wheel file will be generated under `dist/rocm/`.

```bash
pip install dist/rocm/conch_triton_kernels-{version}-py3-none-any.whl
```
