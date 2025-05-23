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
  - Varlen Attention (Prefill/decode attention with paged KV cache)
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

Below is a table comparing the relative performance of our Triton kernels to CUDA baselines (on NVIDIA A10).
The listed runtime is the median runtime from 10,000 iterations on our microbenchmarks.
**Note**: it's difficult to express the performance of a kernel with a single number (performance will vary with input sizes, data types, etc.).
We tried our best to choose representative parameters for a fair comparison.
Most relevant parameters are specified via CLI parameters to the microbenchmarks (`benchmarks/`), so feel free to collect your own results based on your use case.
CUDA runtimes collected via vLLM and bitsandbytes (`vllm==0.8.5` and `bitsandbytes==0.45.5`).

| Operation | CUDA Runtime | Triton Runtime | Triton Speedup |
| --- | --- | --- | --- |
| GeLU, Tanh, and Mul | 2.835 ms | 2.851 ms | 0.99 |
| SiLU and Mul | 0.260 ms | 0.209 ms | 1.24 |
| Paged Attention | 0.374 ms | 0.344 ms | 1.09 |
| Rotary Embedding | 0.579 ms | 0.600 ms | 0.96 |
| RMS Norm (Gemma-style) | 1.392 ms | 0.141 ms | 9.87 |
| RMS Norm (Llama-style) | 0.117 ms | 0.072 ms | 1.63 |
| bitsandbytes: Dequantize | 0.175 ms | 10.950 ms | 0.02 |
| bitsandbytes: Quantize | 0.671 ms | 12.667 ms | 0.05 |
| Int8 Static Quantization | 0.167 ms | 0.164 ms | 1.02 |
| Scaled GEMM [Int8 x BF16] | 2.130 ms | 4.441 ms | 0.48 |
| vLLM: Copy Blocks | 8.550 ms | 9.933 ms | 0.86 |
| vLLM: Reshape and Cache | 0.245 ms | 0.024 ms | 10.21 |

For additional analysis of kernel performance, check out our [performance docs](./docs/performance/).

## Supported platforms

Supported platforms:

- Nvidia A10, CUDA 12.2
- Nvidia H100, CUDA 12.2
- AMD MI300X, ROCm 6.2.4

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
