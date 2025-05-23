# Getting Started: Developer Environment

## Clone repo

```bash
git clone https://github.com/stackav-oss/conch.git
cd conch
```

Unless otherwise specified, all commands are intended to be run from the root of the Conch repository.

## Configure Environment

**Note**: we assume that you have Python 3.10+ already installed on your system.

1. Install [direnv](https://direnv.net/).
1. Enable direnv.

```bash
direnv allow .
```

Direnv is not necessarily required, we primarily use it to manage activation/deactivation of the virtual environment for the project.
If you cannot (or do not want to) install direnv, you'll just need to manually activate/deactivate the virtual environment.

### Optional: User environment

Extra environment variables can be placed in `tools/env/user.sh` (not tracked by git).
For example:

```bash
export ROCM_PATH=/opt/rocm
```

## Install project

To install the project as an editable, clone this repository and run this command from the repo root directory.

```bash
pip install -e ".[dev]"
```

By default this does not install `torch` or `triton`.
You can specify an extra for your platform (either `cuda` or `rocm`) to install the appropriate versions of those packages for your accelerator.
For ROCm/AMD support, you'll need to add `--extra-index-url https://download.pytorch.org/whl/rocm6.2.4`.

```bash
pip install -e ".[dev, cuda]"
```

```bash
pip install -e ".[dev, rocm]" --extra-index-url https://download.pytorch.org/whl/rocm6.2.4
```

## Testing

After installation, to run the Triton kernel tests, execute this command:

```bash
pytest
```

You can also specify the path to a specific test file to run one test individually, for example:

```bash
pytest tests/copy_blocks_test.py
```

If you have other `pytest` installations in your `$PATH`, you can also run `pytest` via:

```bash
python -m pytest
```

## Benchmarks

To benchmark all Triton kernels, execute this script:

```bash
./tools/benchmarks/run_all.sh
```

You can also run benchmarks individually, for example:

```bash
python benchmarks/paged_attention_benchmark.py
```

### Optional: Benchmarking against vLLM

Most unit tests/benchmarks allow comparison to CUDA implementations of operations from vLLM (rather than PyTorch-reference implementations).
In order to use them, you can install vLLM (`pip install vllm`) and set the environment variable `CONCH_ENABLE_VLLM=1`.

```bash
pip install vllm==0.8.5
CONCH_ENABLE_VLLM=1 python benchmarks/paged_attention_benchmark.py
```
