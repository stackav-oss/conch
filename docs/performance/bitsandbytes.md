# `bitsandbytes` Triton Kernel Performance

The performance of the Triton implementations of `bitsandbytes` quantize/dequantize kernels is significantly worse than the CUDA baselines.
The primary reason for this is because of a Triton language limitation.

## Explanation

In the CUDA implementation of these kernels, for each element in a block of the tensor, they apply a mapping to/from the quantized/dequantized representations (e.g. `quantize_nf4` to convert from FP16 to NF4, or `dequantize_nf4` to convert from NF4 to FP16).
It is difficult to do this in Triton because Triton does not generally operate on scalars individually; it is meant to operate on "blocks" on an input tensor.
For example, most of the built-in operations in Triton (e.g. `tl.abs()`, `tl.exp()`, multiply, add, etc.) are automatically broadcasted to each element of the loaded block.
But there is no way to apply a generic function not supported by `triton.langugage` (especially one that cannot be represented mathmetically, like the mapping of an FP16 value to NF4) on a block.
Instead, we must load each element of the block as a scalar, apply the mapping function on that scalar, and then store the result as a scalar.
This _severely_ degrades the performance of the Triton kernel compared to its CUDA counterpart.

## A Potential Fix

A potential fix for this would be to extend Triton to allow registration of a scalar operation as a blockwise operation.
Below is a rough sketch of what the syntax could look like:

```py
@triton.jit
@triton.register_custom_blockwise
def _nf4_quantize(value: float) -> int:
    # See kernel implementation: conch/kernels/quantization/bitsandbytes/quantize_blockwise.py
    # ...
```

If a scalar operation is registered in this way, Triton could automatically "broadcast" it to each element of a block.
For example:

```py
@triton.jit
def _quantize_blockwise_kernel(
    x_ptr: tl.tensor,
    out_ptr: tl.tensor,
    cxpr_blocksize: tl.constexpr,
) -> None:
    # Skipping additional logic for brevity

    # Load input block
    block_offsets = tl.arange(0, cxpr_blocksize)
    x = tl.load(x_ptr + block_offsets)

    # Apply custom blockwise function (syntax for illustration only)
    # Operation accepts a block as a parameter and returns a block
    x_q = triton.custom._nf4_quantize(x)

    # Store output block
    tl.store(out_ptr + block_offsets, x_q)
```

At this time, we are unaware of such a proposal to extend Triton and are unaware of the implementation concerns/feasability.
But nevertheless, this case is an interesting one where, due to Triton language limitations, Triton cannot perform on-par with CUDA.
