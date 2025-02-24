# Conch

This repository contains Triton kernels written by Stack AV for Triton-only vLLM inference.


## Getting started

### Supported platforms

Supported platforms:
- Nvidia A10, CUDA 12.2
- Nvidia H100, CUDA 12.2
- AMD MI300X, ROCm 6.2.2
  - Note: vLLM does not have AMD-GPU-enabled pre-built wheels, so you currently cannot use vLLM dependencies with an AMD GPU.

Work-in-progress platforms:
- [XPU](https://github.com/intel/intel-xpu-backend-for-triton)
- [CPU](https://github.com/triton-lang/triton-cpu)


### Configure environment

**Note**: we assume that you have Python 3.10+ already installed on your system.

1. Install [direnv](https://direnv.net/).
2. Enable direnv.

```bash
direnv allow .
```


### Install requirements

```bash
# Install requirements (choose either requirements-cuda.txt or requirements-rocm.txt, depending on your platform)
pip install -r requirements-cuda.txt
```


#### Optional: User environment

Extra environment variables can be placed in `configuration/user.sh` (not tracked by git).
For example:
```
export ROCM_PATH=/opt/rocm
```


#### Optional: Install vLLM

Some unit tests/benchmarks allow comparison to CUDA implementations of operations from vLLM.
In order to use them, you can install vLLM via:

```bash
pip install -r requirements-vllm.txt
```

This is entirely optional, as all operations have PyTorch-only implementations.


## Running Triton kernel tests

```bash
python -m pytest tests/
```


## Running Triton kernel benchmarks

```bash
python benchmarks/paged_attention_benchmark.py
```

or

```bash
./scripts/run_benchmarks.sh
```

## Open-source credits

- [GemLite](https://github.com/mobiusml/gemlite)
- [vLLM](https://github.com/vllm-project/vllm)
