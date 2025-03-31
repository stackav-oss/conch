# Getting Started: Developer Environment

## Configure Environment

**Note**: we assume that you have Python 3.10+ already installed on your system.

1. Install [direnv](https://direnv.net/).
2. Enable direnv.

```bash
direnv allow .
```

Direnv is not necessarily required, we primarily use it to manage activation/deactivation of the virtual environment for the project.
If you cannot (or do not want to) install direnv, you'll just need to manually activate/deactivate the virtual environment.

### Optional: User environment

Extra environment variables can be placed in `configuration/user.sh` (not tracked by git).
For example:

```
export ROCM_PATH=/opt/rocm
```

## Install project

To install the project as an editable, clone this repository and run this command from the repo root directory.

```
pip install -e ".[dev]"
```

**Note**: For ROCm/AMD support, you'll need to add `--extra-index-url https://download.pytorch.org/whl/rocm6.2`.

## Testing

After installation, to run the Triton kernel tests, execute this command:

```bash
python -m pytest tests/
```

You can also specify the path to a specific test file to run one test individually, for example:

```bash
python -m pytest tests/copy_blocks_test.py
```

## Benchmarks

To benchmark all Triton kernels, execute this script:

```bash
./scripts/run_benchmarks.sh
```

You can also run benchmarks individually, for example:

```bash
python benchmarks/paged_attention_benchmark.py
```

### Optional: Benchmarking against vLLM

Most unit tests/benchmarks allow comparison to CUDA implementations of operations from vLLM (rather than PyTorch-reference implementations).
In order to use them, you can install vLLM (`pip install vllm`) and set the environment variable `CONCH_ENABLE_VLLM=1`.

```bash
pip install vllm==0.6.4
CONCH_ENABLE_VLLM=1 python benchmarks/paged_attention_benchmark.py
```
