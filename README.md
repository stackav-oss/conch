# Conch :shell:

_A "standard library" of Triton kernels_.

## What is Conch?

Conch is a central repository of Triton kernels for accelerating common AI operations.
We strive to provide performant, well-written kernels that can be easily integrated into other projects.
We also strive to support multiple hardware platforms (currently Nvidia and AMD).

## Key Features

We support each of the following operations.
Each operation is complete with a PyTorch-only reference implementation (and sometimes a reference implementation provided by another library, like vLLM), a microbenchmark, and a unit test.

- Activation functions
  - SiLU and mul
  - GeLU and mul
- Attention
  - Paged Attention (Flash-Decoding with Paged KV Cache)
- Embedding
  - Rotary embedding
- Normalization
  - Gemma-style RMS norm
  - Llama-style RMS norm
- Quantization
  - FP8 static quantization
  - Int8 static quantization
  - GEMM
    - Mixed-precision
    - Scaled
  - bitsandbytes
    - NF4/FP4/8-bit blockwise quantize/dequantize
- vLLM
  - KV cache operations
    - Copy blocks
    - Reshape and cache

## Performance

TODO(jmanning-stackav): Add graphics for relative performance of kernels vs. CUDA baselines.

## Supported platforms

Supported platforms:

- Nvidia A10, CUDA 12.2
- Nvidia H100, CUDA 12.2
- AMD MI300X, ROCm 6.2.2

Work-in-progress platforms:

- [XPU](https://github.com/intel/intel-xpu-backend-for-triton)
- [CPU](https://github.com/triton-lang/triton-cpu)

## Getting Started

Check out the [developer instructions](./docs/getting_started/developer_environment.md) to get started!

## Open-source credits

We were inspired by and leverage components of the following libraries:

- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [GemLite](https://github.com/mobiusml/gemlite)
- [vLLM](https://github.com/vllm-project/vllm)

## License

Copyright 2025 [Stack AV Co](https://stackav.com/).
Licensed under the [Apache License, Version 2.0](./LICENSE).
