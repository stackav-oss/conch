# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of Non-Maximum Suppression (NMS).

Kernel based on CUDA torchvision NMS implementation:
https://github.com/pytorch/vision/blob/0721867e42841171254c7acaa45fbaf8ee16d3d7/torchvision/csrc/ops/cuda/nms_kernel.cu
"""

import torch
import triton
import triton.language as tl


@triton.autotune(  # type: ignore[misc]
    configs=[
        triton.Config({"cxpr_block_size": 16}),
        triton.Config({"cxpr_block_size": 32}),
        triton.Config({"cxpr_block_size": 64}),
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
    ],
    key=["num_boxes"],
)
@triton.jit  # type: ignore[misc]
def _create_iou_mask_kernel(
    # Tensors
    boxes_ptr: tl.tensor,  # [N, 4]
    iou_mask_ptr: tl.tensor,  # [N, N]
    # Scalars
    num_boxes: int,
    iou_threshold: float,
    # Strides
    boxes_stride: int,
    iou_mask_stride: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
) -> None:
    """Determine if IoU between all pairs of boxes exceeds the given threshold.

    Note: we only populate the upper-triangular portion of the IoU mask.
    For example: for N = 16 boxes, and cxpr_block_size = 4, the IoU mask will look like this:

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
    |XXXX|XXXX|XXXX|X...|
    |XXXX|XXXX|XXXX|XX..|
    |XXXX|XXXX|XXXX|XXX.|
    |XXXX|XXXX|XXXX|XXXX|
    ---------------------

    The `X`s represent the unpopulated portion of the matrix, while the `.`s represent
    the populated portion.

    Args:
        boxes_ptr: Pointer to boxes tensor, sorted by scores, shape: (N, 4) in (x1, y1, x2, y2) format.
        iou_mask_ptr: Pointer to IoU mask tensor, shape: (N, N).
        num_boxes: Number of boxes.
        iou_threshold: IoU threshold for determining if two boxes overlap.
        boxes_stride: Stride for boxes tensor.
        iou_mask_stride: Stride for IoU mask tensor.
        cxpr_block_size: Block size for processing.
    """
    # What row of the matrix are we processing?
    # Each row corresponds to a box, and we process one row at a time.
    row_index = tl.program_id(0)

    # Load the reference box
    box1_offset = row_index * boxes_stride
    box1_x1 = tl.load(boxes_ptr + box1_offset + 0)
    box1_y1 = tl.load(boxes_ptr + box1_offset + 1)
    box1_x2 = tl.load(boxes_ptr + box1_offset + 2)
    box1_y2 = tl.load(boxes_ptr + box1_offset + 3)

    # Calculate area of the reference box
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)

    # Process all of the columns, blockwise
    for col_block_start in range(row_index, num_boxes, cxpr_block_size):
        # Column offsets for the current block
        col_offsets = col_block_start + tl.arange(0, cxpr_block_size)
        col_mask = col_offsets < num_boxes

        # Load boxes in the current block
        # Shape: (cxpr_block_size,)
        box2_offsets = col_offsets * boxes_stride
        box2_x1 = tl.load(boxes_ptr + box2_offsets + 0, mask=col_mask, other=0.0)
        box2_y1 = tl.load(boxes_ptr + box2_offsets + 1, mask=col_mask, other=0.0)
        box2_x2 = tl.load(boxes_ptr + box2_offsets + 2, mask=col_mask, other=0.0)
        box2_y2 = tl.load(boxes_ptr + box2_offsets + 3, mask=col_mask, other=0.0)

        # Calculate areas of the boxes
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
        # Shape: (cxpr_block_size,)
        union_area = box1_area + box2_area - inter_area
        iou = tl.where(union_area > 0.0, inter_area / union_area, 0.0)

        # Create a mask for IoU values that exceed the threshold
        # Shape: (cxpr_block_size,)
        exceeds_threshold = iou > iou_threshold

        # Note: for debugging, if you want to store the actual IoU values instead of boolean,
        # you can store `iou` instead of `exceeds_threshold`. You'll also need to update the
        # `iou_mask_ptr` type to `boxes.dtype` or similar (instead of `torch.bool`).

        # Store IoU mask -> upper triangular part of the matrix
        iou_output_offsets = row_index * iou_mask_stride + col_offsets
        tl.store(iou_mask_ptr + iou_output_offsets, exceeds_threshold, mask=col_mask)


@triton.autotune(  # type: ignore[misc]
    configs=[
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
        triton.Config({"cxpr_block_size": 512}),
        triton.Config({"cxpr_block_size": 1024}),
        triton.Config({"cxpr_block_size": 2048}),
        triton.Config({"cxpr_block_size": 4096}),
        triton.Config({"cxpr_block_size": 8192}),
    ],
    key=["num_boxes"],
)
@triton.jit  # type: ignore[misc]
def _nms_suppression_kernel(
    # Tensors
    iou_mask_ptr: tl.tensor,  # [N, N]
    keep_mask_ptr: tl.tensor,  # [N]
    # Scalars
    num_boxes: tl.int32,
    # Strides
    iou_mask_stride: tl.int32,
    # Constexprs
    cxpr_block_size: tl.constexpr,
) -> None:
    """NMS suppression kernel.

    Args:
        iou_mask_ptr: Pointer to precomputed IoU mask, shape: (N, N).
        keep_mask_ptr: Pointer to keep mask tensor, shape: (N,).
        num_boxes: Number of boxes.
        iou_mask_stride: Stride for IoU mask tensor.
        cxpr_block_size: Block size for processing.
    """
    # Sequential NMS: for each box in sorted order, suppress later boxes
    for current_box_idx in range(num_boxes - 1):
        # Check if current box is still kept
        is_kept = tl.load(keep_mask_ptr + current_box_idx)
        if is_kept:
            # IoU mask row offset for the current box
            # Because the IoU mask is sorted by score, we will only consider boxes that come after the current box.
            # This means we only need to read the upper triangular part of the IoU mask.
            iou_row_offset = current_box_idx * iou_mask_stride

            # Only process boxes that come after the current box
            next_box_idx = current_box_idx + 1
            remaining_boxes = num_boxes - next_box_idx

            # Iterate blockwise through the columns
            for block_idx in range(tl.cdiv(remaining_boxes, cxpr_block_size)):
                # Masked load of indices for the target boxes in the current block
                block_start = next_box_idx + block_idx * cxpr_block_size
                target_box_offsets = block_start + tl.arange(0, cxpr_block_size)
                target_box_mask = target_box_offsets < num_boxes

                # Suppress boxes with lower scores that have high IoU
                suppression_mask = tl.load(
                    iou_mask_ptr + iou_row_offset + target_box_offsets, mask=target_box_mask, other=False
                )

                # Conditionally store suppression result for high-IoU boxes
                tl.store(keep_mask_ptr + target_box_offsets, False, mask=suppression_mask)

            # Potential race condition: we need to ensure all threads complete the store before the next
            # iteration otherwise we may load stale data for whether or not a box has been suppressed.
            # Aside: `debug_barrier` is a poor name for this function, because it is not only used for debugging,
            # but also to ensure synchronization between threads.
            tl.debug_barrier()


def nms_launcher(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Launch NMS kernel.

    Args:
        boxes: Boxes tensor of shape (N, 4) in (x1, y1, x2, y2) format.
        scores: Scores tensor of shape (N,).
        iou_mask: Mask tensor of shape (N, N), indicating if the IoU between two boxes exceeds the threshold.
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
    _, sorted_indices = torch.sort(scores, dim=0, stable=True, descending=True)
    sorted_boxes = boxes[sorted_indices].contiguous()

    # For each box, create a mask indicating which boxes have IoU with it that exceeds the threshold.
    # Process other boxes blockwise, in chunks of size `cxpr_block_size`.
    stage1_grid = (num_boxes,)

    # Create IoU mask in parallel, only upper-triangular part of the matrix is populated.
    _create_iou_mask_kernel[stage1_grid](
        # Tensors
        boxes_ptr=sorted_boxes,
        iou_mask_ptr=iou_mask,
        # Scalars
        num_boxes=num_boxes,
        iou_threshold=iou_threshold,
        # Strides
        boxes_stride=sorted_boxes.stride(0),
        iou_mask_stride=iou_mask.stride(0),
    )

    # For the suppression stage, we need to process sequentially, but we'll still take
    # advantage of parallelism by processing in blocks in one program.
    stage2_grid = (1,)
    _nms_suppression_kernel[stage2_grid](
        # Tensors
        iou_mask_ptr=iou_mask,
        keep_mask_ptr=keep_mask,
        # Scalars
        num_boxes=num_boxes,
        # Strides
        iou_mask_stride=iou_mask.stride(0),
    )

    # Extract indices of kept boxes
    return sorted_indices[keep_mask]
