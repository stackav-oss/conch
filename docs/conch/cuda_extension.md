# CUDA Extension

In some cases, there are not easily-installable external packages that provide a CUDA-accelerated reference implementation for benchmarking/testing.
For those situations we optionally provide a CUDA extension for Conch.

To build this extension, run the following commands:

```bash
cd cuda_ext/
python setup.py develop
# This should also work:
# pip install wheel
# pip install -e . --no-build-isolation
```

**Note**: After installing the CUDA extension, you may need to reinstall Conch.

```bash
# Go back to the root of the Conch repo
cd ..
pip install -e .
```

To enable these reference implementations, when applicable, use the following environment variable:

```bash
CONCH_ENABLE_CUDA_EXT=1
```
