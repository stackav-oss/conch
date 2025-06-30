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
        triton.Config({"cxpr_block_size": 8}),
        triton.Config({"cxpr_block_size": 16}),
        triton.Config({"cxpr_block_size": 32}),
        triton.Config({"cxpr_block_size": 64}),
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
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
    
      0 1 2 3 4 5
    0 1 . . . . .
    1 x 1 . . . .
    2 x x 1 . . .
    3 x x x 1 . .
    4 x x x x 1 .
    5 x x x x x 1
    
    ---------------------
    |....|....|....|....|
    |X...|....|....|....|
    |XX..|....|....|....|
    |XXX.|....|....|....|
    ---------------------
    |XXXX|....|....|....|
    |XXXX|X...|....|....|
    |XXXX|XX..|....|....|
    |XXXX|XXX.|....|....|
    ---------------------
    |XXXX|XXXX|X...|....|
    |XXXX|XXXX|XX..|....|
    |XXXX|XXXX|XXX.|....|
    |XXXX|XXXX|XXXX|....|
    ---------------------

    Args:
        boxes_ptr: Pointer to boxes tensor, shape: (N, 4) in (x1, y1, x2, y2) format.
        iou_matrix_ptr: Pointer to IoU matrix tensor, shape: (N, N).
        num_boxes: Number of boxes.
        boxes_stride: Stride for boxes tensor.
        iou_matrix_stride: Stride for IoU matrix tensor.
        cxpr_block_size: Block size for processing.
    """
    row_idx = tl.program_id(0)
    # row_block_start = tl.program_id(0) * cxpr_block_size
    col_block_start = tl.program_id(1) * cxpr_block_size
    
    # row_block_end = row_block_start + cxpr_block_size
    # col_block_end = col_block_start + cxpr_block_size

    # if row_idx > tl.program_id(1):
    if row_idx >= col_block_start + cxpr_block_size:
        return
    # if row_block_start >= col_block_start + cxpr_block_size:
    #     # print("Skipping block: row_block_start >= col_block_start + cxpr_block_size: ", tl.program_id(0))
    #     return

    # print("row_block_start = ", row_block_start)
    # print("col_block_start = ", col_block_start)
    
    # Process a block of rows
    # row_offsets = row_block_start + tl.arange(0, cxpr_block_size)
    # row_mask = row_offsets < num_boxes

    # Load the reference box (row_idx)
    box1_offset = row_idx * boxes_stride
    # Shape: (cxpr_block_size,)
    # box1_offsets = row_offsets * boxes_stride
    # box1_x1 = tl.load(boxes_ptr + box1_offsets + 0, mask=row_mask, other=0.0)
    # box1_y1 = tl.load(boxes_ptr + box1_offsets + 1, mask=row_mask, other=0.0)
    # box1_x2 = tl.load(boxes_ptr + box1_offsets + 2, mask=row_mask, other=0.0)
    # box1_y2 = tl.load(boxes_ptr + box1_offsets + 3, mask=row_mask, other=0.0)
    box1_x1 = tl.load(boxes_ptr + box1_offset + 0)
    box1_y1 = tl.load(boxes_ptr + box1_offset + 1)
    box1_x2 = tl.load(boxes_ptr + box1_offset + 2)
    box1_y2 = tl.load(boxes_ptr + box1_offset + 3)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    # if row_block_start == 0 and col_block_start == 0:
    #     print("box1_x1 = ", box1_x1)
    #     print("box1_y1 = ", box1_y1)
    #     print("box1_x2 = ", box1_x2)
    #     print("box1_y2 = ", box1_y2)
    #     print("box1_area = ", box1_area)

    # Process a block of columns
    col_offsets = col_block_start + tl.arange(0, cxpr_block_size)
    col_mask = col_offsets < num_boxes

    # Load boxes in the current block
    # Shape: (cxpr_block_size,)
    box2_offsets = col_offsets * boxes_stride
    box2_x1 = tl.load(boxes_ptr + box2_offsets + 0, mask=col_mask, other=0.0)
    box2_y1 = tl.load(boxes_ptr + box2_offsets + 1, mask=col_mask, other=0.0)
    box2_x2 = tl.load(boxes_ptr + box2_offsets + 2, mask=col_mask, other=0.0)
    box2_y2 = tl.load(boxes_ptr + box2_offsets + 3, mask=col_mask, other=0.0)

    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    # if row_block_start == 0 and col_block_start == 0:
    #     print("box2_x1 = ", box2_x1)
    #     print("box2_y1 = ", box2_y1)
    #     print("box2_x2 = ", box2_x2)
    #     print("box2_y2 = ", box2_y2)
    #     print("box2_area = ", box2_area)

    # Calculate intersection
    # inter_x1 = tl.maximum(box1_x1[:, None], box2_x1)
    # inter_y1 = tl.maximum(box1_y1[:, None], box2_y1)
    # inter_x2 = tl.minimum(box1_x2[:, None], box2_x2)
    # inter_y2 = tl.minimum(box1_y2[:, None], box2_y2)
    # inter_x1 = tl.maximum(box1_x1[None, :], box2_x1)
    # inter_y1 = tl.maximum(box1_y1[None, :], box2_y1)
    # inter_x2 = tl.minimum(box1_x2[None, :], box2_x2)
    # inter_y2 = tl.minimum(box1_y2[None, :], box2_y2)

    # inter_x1 = tl.maximum(box1_x1[:, None], box2_x1[None, :])
    # inter_y1 = tl.maximum(box1_y1[:, None], box2_y1[None, :])
    # inter_x2 = tl.minimum(box1_x2[:, None], box2_x2[None, :])
    # inter_y2 = tl.minimum(box1_y2[:, None], box2_y2[None, :])

    inter_x1 = tl.maximum(box1_x1, box2_x1)
    inter_y1 = tl.maximum(box1_y1, box2_y1)
    inter_x2 = tl.minimum(box1_x2, box2_x2)
    inter_y2 = tl.minimum(box1_y2, box2_y2)
    
    # if row_block_start == 0 and col_block_start == 0:
    #     # Debugging output
    #     print("inter_x1 = ", inter_x1)
    #     print("inter_y1 = ", inter_y1)
    #     print("inter_x2 = ", inter_x2)
    #     print("inter_y2 = ", inter_y2)

    # Check if there's valid intersection
    inter_w = tl.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tl.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    # if row_block_start == 0 and col_block_start == 0:
    #     print("inter_area = ", inter_area)

    # Calculate union and IoU
    # Shape: (cxpr_block_size, cxpr_block_size)
    union_area = box1_area + box2_area - inter_area
    iou = tl.where(union_area > 0.0, inter_area / union_area, 0.0)

    # Store IoU values
    iou_output_offsets = row_idx * iou_matrix_stride + col_offsets
    # iou_output_offsets = row_offsets[:, None] * iou_matrix_stride + col_offsets[None, :]
    # iou_output_mask = row_mask[:, None] & col_mask[None, :] & (row_offsets[:, None] <= col_offsets[None, :])
    # iou_output_mask = row_mask[:, None] & col_mask[None, :]
    # iou_output_mask = row_mask[:, None] & col_mask[None, :]
    # if row_block_start == 0 and col_block_start == 0:
    #     print("iou = ", iou)
    # print("iou_output_offsets = ", iou_output_offsets)
    # print("iou_output_mask = ", iou_output_mask)
    # tl.store(iou_matrix_ptr + iou_output_offsets, iou, mask=(row_mask[:, None] & col_mask[None, :]))
    # tl.store(iou_matrix_ptr + iou_output_offsets, iou, mask=iou_output_mask)
    tl.store(iou_matrix_ptr + iou_output_offsets, iou, mask=col_mask)


@triton.autotune(  # type: ignore[misc]
    configs=[
        # triton.Config({"cxpr_block_size": 4}),
        # triton.Config({"cxpr_block_size": 8}),
        triton.Config({"cxpr_block_size": 32}),
        triton.Config({"cxpr_block_size": 64}),
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
        triton.Config({"cxpr_block_size": 512}),
        triton.Config({"cxpr_block_size": 1024}),
        triton.Config({"cxpr_block_size": 2048}),
        triton.Config({"cxpr_block_size": 4096}),
    ],
    key=["num_boxes"],
)
@triton.jit  # type: ignore[misc]
def _nms_suppression_kernel(
    # Tensors
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
        iou_matrix_ptr: Pointer to precomputed IoU matrix, shape: (N, N).
        keep_mask_ptr: Pointer to keep mask tensor, shape: (N,).
        num_boxes: Number of boxes.
        iou_threshold: IoU threshold for suppression.
        iou_matrix_stride: Stride for IoU matrix tensor.
        cxpr_block_size: Block size for processing.
        cxpr_num_boxes_padded: Padded number of boxes for block processing.
    """
    # Sequential NMS: for each box in sorted order, suppress later boxes
    for current_box_idx in range(num_boxes):
        # Check if current box is still kept
        is_kept = tl.load(keep_mask_ptr + current_box_idx)
        if is_kept:
            # IoU row offset for the current box
            # iou_row_offset = current_box_idx * iou_matrix_stride
            row_indices = tl.full([cxpr_block_size], current_box_idx, dtype=tl.int64)
            # print("row_indices = ", row_indices)

            # Iterate blockwise through the columns
            for block_idx in range(triton.cdiv(cxpr_num_boxes_padded, cxpr_block_size)):
                # Only need to consider later boxes, so start from current_box + 1
                block_start = current_box_idx + 1 + block_idx * cxpr_block_size
                # Only process if the start of the block is within bounds
                if block_start < num_boxes:
                    # Masked load of indices for the target boxes in the current block
                    # target_box_offsets = block_start + tl.arange(0, cxpr_block_size)
                    # target_box_mask = target_box_offsets < num_boxes
                    # target_box_indices = tl.load(sorted_indices_ptr + target_box_offsets, mask=target_box_mask)
                    target_box_offsets = block_start + tl.arange(0, cxpr_block_size)
                    target_box_mask = target_box_offsets < num_boxes

                    adjusted_row_offsets = tl.where(row_indices < target_box_offsets, row_indices, target_box_offsets)
                    adjusted_col_offsets = tl.where(row_indices < target_box_offsets, target_box_offsets, row_indices)
                    # target_box_indices = tl.load(sorted_indices_ptr + target_box_offsets, mask=target_box_mask)
                    
                    # print("target_box_offsets = ", target_box_offsets)
                    # print("column_indices = ", column_indices)
                    
                    # print("adjusted_row_offsets = ", adjusted_row_offsets)
                    # print("adjusted_col_offsets = ", adjusted_col_offsets)
                    # print("target_box_mask = ", target_box_mask)

                    # Load IoU values for the current block
                    # iou_values = tl.load(iou_matrix_ptr + iou_row_offset + target_box_indices, mask=target_box_mask, other=0.0)
                    #
                    # iou_values = tl.load(iou_matrix_ptr + adjusted_row_offsets * iou_matrix_stride + adjusted_col_offsets, mask=target_box_mask, other=0.0)
                    # iou_values = tl.load(iou_matrix_ptr + adjusted_col_offsets * iou_matrix_stride + adjusted_row_offsets, mask=target_box_mask, other=0.0)
                    iou_values = tl.load(iou_matrix_ptr + (adjusted_row_offsets * iou_matrix_stride) + adjusted_col_offsets, mask=target_box_mask, other=0.0)
                    
                    # print("iou_values = ", iou_values)

                    # Suppress boxes with lower scores that have high IoU
                    # suppression_mask = tl.where(iou_values > iou_threshold, True, False) & target_box_mask
                    suppression_mask = tl.where(iou_values > iou_threshold, True, False)
                    
                    # print("column_indices = ", column_indices)
                    # print("suppress_mask = ", suppression_mask)

                    # Conditionally store suppression result for high-IoU boxes
                    tl.store(keep_mask_ptr + target_box_offsets, False, mask=suppression_mask)


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
    _, sorted_indices = torch.sort(scores, descending=True)
    sorted_boxes = boxes[sorted_indices]

    # Calculate IoU of each each block against all other boxes in parallel. Process other boxes
    # blockwise in chunks of size `cxpr_block_size`.
    def stage1_grid(meta: dict[str, Any]) -> tuple[int, int]:
        num_blocks = triton.cdiv(num_boxes, meta["cxpr_block_size"])
        # return (num_blocks, num_blocks)
        return (num_boxes, num_blocks)

    # Calculate IoU matrix using Triton kernel
    _calculate_iou_kernel[stage1_grid](
        # Tensors
        boxes_ptr=sorted_boxes,
        iou_matrix_ptr=iou_matrix,
        # Scalars
        num_boxes=num_boxes,
        # Strides
        boxes_stride=boxes.stride(0),
        iou_matrix_stride=iou_matrix.stride(0),
    )
    
    # print(f"{iou_matrix = }")

    # For the suppression stage, we need to process sequentially, but we'll still take
    # advantage of parallelism by processing in blocks in one program.
    stage2_grid = (1,)
    _nms_suppression_kernel[stage2_grid](
        # Tensors
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
    return sorted_indices[keep_mask]
