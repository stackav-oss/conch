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
  - GeLU and mul
  - SiLU and mul
- Attention
  - Paged Attention (Flash-Decoding with Paged KV Cache)
- Embedding
  - Rotary embedding
- Normalization
  - Gemma-style RMS norm
  - Llama-style RMS norm
- Quantization
  - bitsandbytes
    - NF4/FP4/8-bit blockwise quantize/dequantize
  - FP8 static quantization
  - Int8 static quantization
  - GEMM
    - Mixed-precision
    - Scaled
- vLLM
  - KV cache operations
    - Copy blocks
    - Reshape and cache

## Performance

The goal of Conch is not to claim that our operations are faster than CUDA implementations.
Our goal is to write Triton operations that are _as fast as_ the state-of-the-art CUDA implementations.
This allows developers on any hardware platform (Nvidia, AMD, etc.) access to the same, performant kernels.

Below is a table comparing the relative performance of our Triton kernels to CUDA baselines (on H100).
The listed runtime is the median runtime from 10,000 iterations on our microbenchmarks.
**Note**: it's difficult to express the performance of a kernel with a single number (performance will vary with input sizes, data types, etc.).
We tried our best to choose representative parameters for a fair comparison.
Most relevant parameters are specified via CLI parameters to the microbenchmarks (`benchmarks/`), so feel free to collect your own results based on your use case.
CUDA runtimes collected via vLLM and bitsandbytes (`vllm==0.6.4` and `bitsandbytes==0.45.4`).

| Operation | CUDA Runtime | Triton Runtime | Triton Speedup |
| --- | --- | --- | --- |
| GeLU, Tanh, and Mul | 0.493 ms | 0.466 ms | 1.06 |
| SiLU and Mul | 0.063 ms | 0.047 ms | 1.34 |
| Paged Attention | 0.090 ms | 0.083 ms | 1.08 |
| Rotary Embedding | 0.107 ms | 0.103 ms | 1.04 |
| RMS Norm (Gemma-style) | 0.392 ms | 0.029 ms | 13.52 |
| RMS Norm (Llama-style) | 0.044 ms | 0.018 ms | 2.44 |
| bitsandbytes: Dequantize | 0.074 ms | 4.487 ms | 0.02 |
| bitsandbytes: Quantize | 0.377 ms | 4.819 ms | 0.08 |
| FP8 Static Quantization | 0.035 ms | 0.090 ms | 0.39 |
| Int8 Static Quantization | 0.056 ms | 0.094 ms | 0.60 |
| Mixed-precision GEMM [Int4 x FP16] | 0.432 ms | 1.437 ms | 0.30 |
| Scaled GEMM [Int8 x BF16] | 0.204 ms | 0.285 ms | 0.72 |
| vLLM: Copy Blocks | 2.231 ms | 1.807 ms | 1.23 |
| vLLM: Reshape and Cache | 0.057 ms | 0.010 ms | 5.70 |

For additional analysis of kernel performance, check out our [performance docs](./docs/performance/).

## Supported platforms

Supported platforms:

- Nvidia A10, CUDA 12.2
- Nvidia H100, CUDA 12.2
- AMD MI300X, ROCm 6.2.2

Work-in-progress platforms:

- [XPU](https://github.com/intel/intel-xpu-backend-for-triton)
- [CPU](https://github.com/triton-lang/triton-cpu)

## Getting Started

### Users

Check out the [installation instructions](./docs/getting_started/installation.md) to get started!

### Developers

Check out the [developer instructions](./docs/getting_started/developer_environment.md) to get started!

## Open-source credits

We were inspired by and leverage components of the following libraries:

- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [GemLite](https://github.com/mobiusml/gemlite)
- [vLLM](https://github.com/vllm-project/vllm)

## License

Copyright 2025 [Stack AV Co](https://stackav.com/).
Licensed under the [Apache License, Version 2.0](./LICENSE).
