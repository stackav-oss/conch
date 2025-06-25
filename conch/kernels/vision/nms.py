# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of Non-Maximum Suppression (NMS).

Kernel based on CUDA torchvision NMS implementation:
https://github.com/pytorch/vision/blob/0721867e42841171254c7acaa45fbaf8ee16d3d7/torchvision/csrc/ops/cuda/nms_kernel.cu
"""

from typing import Any

import torch
import triton
import triton.language as tl


@triton.autotune(  # type: ignore[misc]
    configs=[
        triton.Config({"cxpr_block_size": 64}),
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
        triton.Config({"cxpr_block_size": 512}),
        triton.Config({"cxpr_block_size": 1024}),
    ],
    key=["num_boxes"],
)
@triton.jit  # type: ignore[misc]
def _calculate_iou_kernel(
    # Tensors
    boxes_ptr: tl.tensor,  # [N, 4]
    iou_matrix_ptr: tl.tensor,  # [N, N]
    # Scalars
    num_boxes: int,
    # Strides
    boxes_stride: int,
    iou_matrix_stride: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
) -> None:
    """Calculate IoU matrix between all pairs of boxes.

    Args:
        boxes_ptr: Pointer to boxes tensor, shape: (N, 4) in (x1, y1, x2, y2) format.
        iou_matrix_ptr: Pointer to IoU matrix tensor, shape: (N, N).
        num_boxes: Number of boxes.
        boxes_stride: Stride for boxes tensor.
        iou_matrix_stride: Stride for IoU matrix tensor.
        cxpr_block_size: Block size for processing.
    """
    row_idx = tl.program_id(0)
    col_block_start = tl.program_id(1) * cxpr_block_size

    # Load the reference box (row_idx)
    box1_offset = row_idx * boxes_stride
    box1_x1 = tl.load(boxes_ptr + box1_offset + 0)
    box1_y1 = tl.load(boxes_ptr + box1_offset + 1)
    box1_x2 = tl.load(boxes_ptr + box1_offset + 2)
    box1_y2 = tl.load(boxes_ptr + box1_offset + 3)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)

    # Process a block of columns
    col_offsets = col_block_start + tl.arange(0, cxpr_block_size)
    col_mask = col_offsets < num_boxes

    # Load boxes in the current block
    box2_offsets = col_offsets * boxes_stride
    box2_x1 = tl.load(boxes_ptr + box2_offsets + 0, mask=col_mask)
    box2_y1 = tl.load(boxes_ptr + box2_offsets + 1, mask=col_mask)
    box2_x2 = tl.load(boxes_ptr + box2_offsets + 2, mask=col_mask)
    box2_y2 = tl.load(boxes_ptr + box2_offsets + 3, mask=col_mask)

    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Calculate intersection
    inter_x1 = tl.maximum(box1_x1, box2_x1)
    inter_y1 = tl.maximum(box1_y1, box2_y1)
    inter_x2 = tl.minimum(box1_x2, box2_x2)
    inter_y2 = tl.minimum(box1_y2, box2_y2)

    # Check if there's valid intersection
    inter_w = tl.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tl.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Calculate union and IoU
    union_area = box1_area + box2_area - inter_area
    iou = tl.where(union_area > 0.0, inter_area / union_area, 0.0)

    # Store IoU values
    iou_output_offsets = row_idx * iou_matrix_stride + col_offsets
    tl.store(iou_matrix_ptr + iou_output_offsets, iou, mask=col_mask)


@triton.autotune(  # type: ignore[misc]
    configs=[
        triton.Config({"cxpr_block_size": 64}),
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
        triton.Config({"cxpr_block_size": 512}),
        triton.Config({"cxpr_block_size": 1024}),
    ],
    key=["num_boxes"],
)
@triton.jit  # type: ignore[misc]
def _nms_suppression_kernel(
    # Tensors
    sorted_indices_ptr: tl.tensor,  # [N]
    iou_matrix_ptr: tl.tensor,  # [N, N]
    keep_mask_ptr: tl.tensor,  # [N]
    # Scalars
    num_boxes: int,
    iou_threshold: float,
    # Strides
    iou_matrix_stride: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
    cxpr_num_boxes_padded: tl.constexpr,
) -> None:
    """NMS suppression kernel.

    Args:
        sorted_indices_ptr: Pointer to sorted indices tensor, shape: (N,).
        iou_matrix_ptr: Pointer to precomputed IoU matrix, shape: (N, N).
        keep_mask_ptr: Pointer to keep mask tensor, shape: (N,).
        num_boxes: Number of boxes.
        iou_threshold: IoU threshold for suppression.
        iou_matrix_stride: Stride for IoU matrix tensor.
        cxpr_block_size: Block size for processing.
        cxpr_num_boxes_padded: Padded number of boxes for block processing.
    """
    # Sequential NMS: for each box in sorted order, suppress later boxes
    # Iterate through sorted indices
    for i in range(num_boxes):
        # Get the current box index from sorted indices
        current_box_idx = tl.load(sorted_indices_ptr + i)

        # Check if current box is still kept
        is_kept = tl.load(keep_mask_ptr + current_box_idx)
        if is_kept:
            # IoU row offset for the current box
            iou_row_offset = current_box_idx * iou_matrix_stride

            # Iterate blockwise through the columns
            for block_idx in range(triton.cdiv(cxpr_num_boxes_padded, cxpr_block_size)):
                # Only need to consider later boxes, so start from i + 1 (i is the current box index)
                block_start = i + 1 + block_idx * cxpr_block_size
                # Only process if the start of the block is within bounds
                if block_start < num_boxes:
                    # Masked load of indices for the target boxes in the current block
                    target_box_offsets = block_start + tl.arange(0, cxpr_block_size)
                    target_box_mask = target_box_offsets < num_boxes
                    target_box_indices = tl.load(sorted_indices_ptr + target_box_offsets, mask=target_box_mask)

                    # Load IoU values for the current block
                    iou_values = tl.load(iou_matrix_ptr + iou_row_offset + target_box_indices, mask=target_box_mask)

                    # Suppress boxes with lower scores that have high IoU
                    suppression_mask = tl.where(iou_values > iou_threshold, True, False)

                    # Conditionally store suppression result for high-IoU boxes
                    tl.store(keep_mask_ptr + target_box_indices, False, mask=suppression_mask)


def nms_launcher(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_matrix: torch.Tensor,
    keep_mask: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Launch NMS kernel.

    Args:
        boxes: Boxes tensor of shape (N, 4) in (x1, y1, x2, y2) format.
        scores: Scores tensor of shape (N,).
        iou_matrix: IoU matrix of shape (N, N).
        keep_mask: Mask tensor of shape (N,) indicating which boxes are kept.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Tensor: Indices of kept boxes, sorted by decreasing score.
    """
    assert boxes.dim() == 2 and boxes.size(1) == 4, "Boxes must have shape (N, 4)"
    assert scores.dim() == 1, "Scores must be 1D"
    assert boxes.size(0) == scores.size(0), "Number of boxes and scores must match"
    assert boxes.is_contiguous(), "Boxes tensor must be contiguous"
    assert scores.is_contiguous(), "Scores tensor must be contiguous"

    num_boxes = boxes.size(0)

    # Sort boxes by scores in descending order
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    # Calculate IoU of each each block against all other boxes in parallel. Process other boxes
    # blockwise in chunks of size `cxpr_block_size`.
    def stage1_grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (num_boxes, triton.cdiv(num_boxes, meta["cxpr_block_size"]))

    # Calculate IoU matrix using Triton kernel
    _calculate_iou_kernel[stage1_grid](
        # Tensors
        boxes_ptr=boxes,
        iou_matrix_ptr=iou_matrix,
        # Scalars
        num_boxes=num_boxes,
        # Strides
        boxes_stride=boxes.stride(0),
        iou_matrix_stride=iou_matrix.stride(0),
    )

    # For the suppression stage, we need to process sequentially, but we'll still take
    # advantage of parallelism by processing in blocks in one program.
    stage2_grid = (1,)
    _nms_suppression_kernel[stage2_grid](
        # Tensors
        sorted_indices_ptr=sorted_indices,
        iou_matrix_ptr=iou_matrix,
        keep_mask_ptr=keep_mask,
        # Scalars
        num_boxes=num_boxes,
        iou_threshold=iou_threshold,
        # Strides
        iou_matrix_stride=iou_matrix.stride(0),
        # Constexprs
        cxpr_num_boxes_padded=triton.next_power_of_2(num_boxes),
    )

    # Extract indices of kept boxes
    return sorted_indices[keep_mask[sorted_indices]]
