# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference implementation of non-max suppression."""

import torch

from conch import envs


def _calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in (x1, y1, x2, y2) format.
        boxes2: Tensor of shape (M, 4) in (x1, y1, x2, y2) format.

    Returns:
        Tensor of shape (N, M) containing IoU values.
    """
    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Expand dimensions for broadcasting
    boxes1_expanded = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2_expanded = boxes2.unsqueeze(0)  # (1, M, 4)

    # Calculate intersection coordinates
    inter_x1 = torch.max(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])  # (N, M)
    inter_y1 = torch.max(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])  # (N, M)
    inter_x2 = torch.min(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])  # (N, M)
    inter_y2 = torch.min(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])  # (N, M)

    # Calculate intersection area
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h  # (N, M)

    # Calculate union area
    area1_expanded = area1.unsqueeze(1)  # (N, 1)
    area2_expanded = area2.unsqueeze(0)  # (1, M)
    union_area = area1_expanded + area2_expanded - inter_area  # (N, M)

    # Calculate IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    return iou


def _nms_pytorch_iterative(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Sort boxes by scores in descending order
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    # Keep track of which boxes to keep
    keep = []

    # Process boxes in order of decreasing score
    while sorted_indices.numel() > 0:
        # Take the box with highest score
        current_idx = sorted_indices[0]
        # keep.append(current_idx.item())
        keep.append(current_idx)

        if sorted_indices.numel() == 1:
            break

        # Get remaining boxes
        remaining_indices = sorted_indices[1:]
        current_box = boxes[current_idx].unsqueeze(0)  # (1, 4)
        remaining_boxes = boxes[remaining_indices]  # (K, 4)

        # Calculate IoU between current box and all remaining boxes
        ious = _calculate_iou(current_box, remaining_boxes)  # (1, K)
        ious = ious.squeeze(0)  # (K,)

        # Keep only boxes with IoU below threshold
        keep_mask = ious <= iou_threshold
        sorted_indices = remaining_indices[keep_mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _nms_pytorch_vectorized(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Vectorized implementation of NMS that pre-computes all IoUs.

    This version is more memory-intensive but can be faster for medium-sized inputs
    by reducing the number of iterations.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    num_boxes = boxes.size(0)

    # Sort boxes by scores in descending order
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    # Pre-compute all pairwise IoUs
    iou_matrix = _calculate_iou(boxes, boxes)  # (N, N)

    # Initialize keep mask
    keep_mask = torch.ones(num_boxes, dtype=torch.bool, device=boxes.device)

    # Process boxes in order of decreasing score
    for i in range(num_boxes):
        current_idx = sorted_indices[i]

        if not keep_mask[current_idx]:
            continue

        # Suppress boxes with higher IoU than threshold
        # Only consider boxes with lower scores (higher indices in sorted order)
        for j in range(i + 1, num_boxes):
            next_idx = sorted_indices[j]
            if keep_mask[next_idx] and iou_matrix[current_idx, next_idx] > iou_threshold:
                keep_mask[next_idx] = False

    # Extract kept indices in original score order
    return sorted_indices[keep_mask[sorted_indices]]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, vectorize: bool = False) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold
        vectorize (bool): whether to enable vectorized NMS implementation

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if envs.CONCH_ENABLE_TORCHVISION:
        from torchvision.ops.boxes import nms as nms_torchvision  # type: ignore[import-untyped]

        return nms_torchvision(boxes, scores, iou_threshold)  # type: ignore[no-any-return]

    if vectorize:
        return _nms_pytorch_vectorized(boxes, scores, iou_threshold)

    return _nms_pytorch_iterative(boxes, scores, iou_threshold)
