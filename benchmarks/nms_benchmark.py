# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""NMS benchmark."""

import sys
from typing import Final

import click
import torch

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
def main(
    num_boxes: int,
    iou_threshold: float,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark NMS.

    Args:
        num_boxes: Number of boxes to create.
        iou_threshold: IoU threshold for boxes to be kept.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
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

    reference_output = nms_ref(boxes, scores, iou_threshold)

    compiled_fn = torch.compile(nms_ref)
    compiled_output = compiled_fn(boxes, scores, iou_threshold)

    if not torch.allclose(compiled_output, reference_output, atol=absolute_tolerance):
        print(f"WARNING: Reference and compiled results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(reference_output - compiled_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {reference_output}", file=sys.stderr)
            print(f"Compiled output: {compiled_output}", file=sys.stderr)
    else:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    # Benchmark Reference vs. Compiled implementations
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

    compiled_result = benchmark_it(
        lambda: compiled_fn(
            boxes,
            scores,
            iou_threshold,
        ),
        tag="Compiled",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    # Print results
    compiled_result.print_parameters(csv=csv)
    compiled_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
