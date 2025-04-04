# Conch Structure

## Kernels

This directory contains the Triton kernels for each operation supported by Conch.
Subdirectories should separate kernels based on their "category" (e.g. activation functions, quantization, etc.).
We do not currently have any best-practices on this subdirectory structure.

Each kernel file should follow the same best-practices/structure:

- Kernel files should minimize imports from other files in the repo
  - Some may require `conch.platforms` for platform-specific code (e.g. FP8 dtype on Nvidia vs. AMD)
  - Some may require additional files from `conch.third_party` (e.g. `vllm.scalar_type` for quantization kernels)
- The interface to call the kernel should be named `{operation}_launcher`
  - The launcher should not allocate memory
  - The launcher can include some input validation, but can largely assume that its arguments are valid
    - `assert` is acceptable in kernels, because we assume that error-checking should have happened in the operation public interface
- Kernel function names should be prefixed with an underscore
  - These functions should not be called directly, except by the launcher
- Kernel files may include multiple kernels and launchers

## Ops

This directory contains public interfaces for the Triton kernels.
Each file under `conch/kernels/` should have a corresponding file under `conch/ops/`.
The subdirectory structure between `kernels/` and `ops/` should be identical.

Each operation file should follow the same best-practices/structure:

- Operation files should import the kernel launcher (not the private kernel implementations)
- The public interface to call the kernel should generally be named `{operation}` and match the file name
  - This may not always be the case, as some operation files may include public interfaces for multiple kernels
- The public interface may allocate an output buffer for memory or any other intermediate memory required by the kernel launcher
- The public interface should do all necessary error checking, ideally by raising exceptions (and not `assert`)

## Platforms

This directory contains utilities for handling multiple platforms (i.e. Nvidia, AMD, XPU, etc.).
Currently we have a simple platform-detection mechanism based on PyTorch.

## Reference

This directory contains reference implementations of the operations supported by Conch.
Similar to the `kernels/` and `ops/` directory, there is a subdirectory structure for categories of operations.
The reference implementations **must** be written in pure PyTorch.
If there is an implementation from another library (e.g. vLLM), its use should be gated by an environment variable toggle (e.g. `CONCH_ENABLE_VLLM`).
Reference implementations with multiple backends (e.g. PyTorch and vLLM) should have a unified public interface.

These reference implementations are necessary for unit testing and benchmarking our Triton implementations.
For most cases, Triton should be significantly faster than pure-PyTorch implementations.

## Third Party

This directory contains any third-party code from other projects.
For instance, we leverage some libraries/utilities from vLLM for our quantization work (`conch.third_party.vllm.quant_utils` and `conch.third_party.vllm.scalar_type`). Adding code from other projects in this directory allows us to avoid an explicit dependency on those projects.

Third-party code should be added in a subfolder named after the name of the project.

## Utils

This directory contains any miscellaneous utilities.
