# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""NMS benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.vision.nms import nms as nms_conch
from conch.platforms import current_platform
from conch.reference.vision.nms import nms as nms_ref
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


def _create_tensors_with_iou(num_boxes: int, iou_thresh: float) -> tuple[torch.Tensor, torch.Tensor]:
    # force last box to have a pre-defined iou with the first box
    # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
    # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
    # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
    # Adjust the threshold upward a bit with the intent of creating
    # at least one box that exceeds (barely) the threshold and so
    # should be suppressed.
    boxes = torch.rand(num_boxes, 4) * 100
    boxes[:, 2:] += boxes[:, :2]
    boxes[-1, :] = boxes[0, :]
    x0, y0, x1, y1 = boxes[-1].tolist()
    iou_thresh += 1e-5
    boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
    scores = torch.rand(num_boxes)
    return boxes, scores


@click.command()
@click.option(
    "--num-boxes",
    required=False,
    type=int,
    default=1000,
    help="Number of boxes to create",
)
@click.option(
    "--iou-threshold",
    required=False,
    type=float,
    default=0.2,
    help="IoU threshold for boxes to be kept",
)
@click.option(
    "--vectorize-ref",
    is_flag=True,
    help="Flag to enable vectorization in the reference implementation",
)
@click.option(
    "--gpu-ref",
    is_flag=True,
    help="Flag to enable GPU reference implementation",
)
@click.option(
    "--iteration-time-ms",
    required=False,
    type=int,
    default=10000,
    help="Time in milliseconds to run benchmark",
)
@click.option(
    "--warmup-time-ms",
    required=False,
    type=int,
    default=1000,
    help="Time in milliseconds to warmup before recording times",
)
@click.option(
    "--absolute-tolerance",
    required=False,
    type=float,
    default=1e-3,
    help="Absolute tolerance to match with",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Flag for printing verbose output",
)
@click.option(
    "--gpu",
    required=False,
    type=str,
    default=current_platform.device,
    help="Device to run on",
)
@click.option(
    "--csv",
    is_flag=True,
    help="Flag for printing results in CSV format",
)
@click.option(
    "--compile-ref",
    is_flag=True,
    help="Flag to torch.compile() the reference impl",
)
@click.option(
    "--compile-conch",
    is_flag=True,
    help="Flag to torch.compile() the Conch impl",
)
def main(
    num_boxes: int,
    iou_threshold: float,
    vectorize_ref: bool,
    gpu_ref: bool,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
    compile_ref: bool,
    compile_conch: bool,
) -> None:
    """Benchmark NMS.

    Args:
        num_boxes: Number of boxes to create.
        iou_threshold: IoU threshold for boxes to be kept.
        vectorize_ref: Flag to enable vectorization in the reference implementation.
        gpu_ref: Flag to enable GPU reference implementation.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
        compile_ref: Flag to torch.compile() the reference implementation.
        compile_conch: Flag to torch.compile() the Conch implementation.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "num_boxes": num_boxes,
            "iou_threshold": iou_threshold,
        },
    )

    boxes, scores = _create_tensors_with_iou(num_boxes, iou_threshold)

    reference_vectorized_fn = None
    reference_gpu_fn = None
    if vectorize_ref:
        # Use vectorized reference implementation if requested
        from conch.reference.vision.nms import _nms_pytorch_vectorized

        reference_vectorized_fn = _nms_pytorch_vectorized
    if gpu_ref:
        # Use GPU reference implementation if requested
        from torchvision.ops.boxes import nms as nms_torchvision  # type: ignore[import-untyped]

        reference_gpu_fn = nms_torchvision

    reference_compiled_fn = None
    reference_vectorized_compiled_fn = None
    if compile_ref:
        # Compile the reference implementation if requested
        reference_compiled_fn = torch.compile(nms_ref)
        if vectorize_ref:
            reference_vectorized_compiled_fn = torch.compile(reference_vectorized_fn)

    conch_compiled_fn = torch.compile(nms_conch) if compile_conch else None

    # Get reference output
    reference_output = nms_ref(boxes, scores, iou_threshold)

    # Test Conch implementation
    conch_output = nms_conch(boxes, scores, iou_threshold)

    # Accuracy checks
    if not torch.allclose(conch_output, reference_output, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Ref kept: {len(reference_output)}, Conch kept: {len(conch_output)}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {reference_output}", file=sys.stderr)
            print(f"Conch output: {conch_output}", file=sys.stderr)
    else:
        print(f"Reference vs Conch: Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    # Benchmark implementations
    baseline_result = benchmark_it(
        lambda: nms_ref(
            boxes,
            scores,
            iou_threshold,
        ),
        tag="PyTorch Reference",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: nms_conch(
            boxes,
            scores,
            iou_threshold,
        ),
        tag="Conch",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    reference_compiled_result = None
    reference_vectorized_result = None
    reference_vectorized_compiled_result = None
    reference_gpu_result = None
    conch_compiled_result = None

    if reference_compiled_fn:
        reference_compiled_result = benchmark_it(
            lambda: reference_compiled_fn(
                boxes,
                scores,
                iou_threshold,
            ),
            tag="PyTorch Reference (Compiled)",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    if reference_vectorized_fn:
        reference_vectorized_result = benchmark_it(
            lambda: reference_vectorized_fn(
                boxes,
                scores,
                iou_threshold,
            ),
            tag="PyTorch Reference (Vectorized)",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    if reference_vectorized_compiled_fn:
        reference_vectorized_compiled_result = benchmark_it(
            lambda: reference_vectorized_compiled_fn(  # type: ignore[call-arg]
                boxes,  # type: ignore[arg-type]
                scores,
                iou_threshold,
            ),
            tag="PyTorch Reference (Vectorized, Compiled)",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    if reference_gpu_fn:
        reference_gpu_result = benchmark_it(
            lambda: reference_gpu_fn(
                boxes,
                scores,
                iou_threshold,
            ),
            tag="PyTorch GPU Reference",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    if conch_compiled_fn:
        conch_compiled_result = benchmark_it(
            lambda: conch_compiled_fn(
                boxes,
                scores,
                iou_threshold,
            ),
            tag="Conch (Compiled)",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    conch_result.print_parameters(csv=csv)
    conch_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)
    if reference_compiled_result:
        reference_compiled_result.print_results(csv=csv)
    if reference_vectorized_result:
        reference_vectorized_result.print_results(csv=csv)
    if reference_vectorized_compiled_result:
        reference_vectorized_compiled_result.print_results(csv=csv)
    if reference_gpu_result:
        reference_gpu_result.print_results(csv=csv)
    if conch_compiled_result:
        conch_compiled_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
