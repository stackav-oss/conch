# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test non-max suppression."""

from typing import Final

import pytest
import torch

from conch.ops.vision.nms import nms as nms_conch
from conch.platforms import current_platform
from conch.reference.vision.nms import nms as nms_ref
from conch.third_party.vllm.utils import seed_everything


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


@pytest.mark.parametrize("num_boxes", [10, 100, 1000, 4000])
@pytest.mark.parametrize("iou_threshold", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", range(3))
def test_nms_conch_vs_reference(
    num_boxes: int,
    iou_threshold: float,
    seed: int,
) -> None:
    """Test that Triton NMS gives same results as reference implementation."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    boxes, scores = _create_tensors_with_iou(num_boxes, iou_threshold)

    # Get results from both implementations
    keep_ref = nms_ref(boxes, scores, iou_threshold)
    keep_conch = nms_conch(boxes, scores, iou_threshold)

    # Results should be identical (same indices, same order)
    torch.testing.assert_close(keep_ref, keep_conch)


def test_nms_edge_cases() -> None:
    """Test NMS edge cases."""
    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    # Test empty input
    empty_boxes = torch.empty(0, 4, device=device)
    empty_scores = torch.empty(0, device=device)

    keep_ref = nms_ref(empty_boxes, empty_scores, 0.5)
    keep_conch = nms_conch(empty_boxes, empty_scores, 0.5)

    assert len(keep_ref) == 0
    assert len(keep_conch) == 0

    # Test single box
    single_box = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)
    single_score = torch.tensor([0.9], device=device)

    keep_ref = nms_ref(single_box, single_score, 0.5)
    keep_conch = nms_conch(single_box, single_score, 0.5)

    torch.testing.assert_close(keep_ref, keep_conch)
    assert len(keep_ref) == 1
    assert len(keep_conch) == 1


@pytest.mark.parametrize("iou_threshold", [0.0, 0.3, 0.7, 1.0])
def test_nms_identical_boxes(iou_threshold: float) -> None:
    """Test NMS with identical boxes at different score levels."""
    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    # Create multiple identical boxes with different scores
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        device=device,
    )

    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], device=device)

    keep_ref = nms_ref(boxes, scores, iou_threshold)
    keep_conch = nms_conch(boxes, scores, iou_threshold)

    # For identical boxes, only the highest scoring one should be kept (except when threshold is 1.0)
    if iou_threshold < 1.0:
        assert len(keep_ref) == 1
        assert len(keep_conch) == 1
        assert keep_ref[0] == 0  # Highest scoring box
        assert keep_conch[0] == 0
    else:
        # When threshold is 1.0, all boxes should be kept
        assert len(keep_ref) == boxes.size(0)
        assert len(keep_conch) == boxes.size(0)


def test_nms_no_overlap() -> None:
    """Test NMS with non-overlapping boxes."""
    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    # Create non-overlapping boxes
    boxes = torch.tensor(
        [
            [0.0, 0.0, 5.0, 5.0],
            [10.0, 10.0, 15.0, 15.0],
            [20.0, 20.0, 25.0, 25.0],
            [30.0, 30.0, 35.0, 35.0],
        ],
        device=device,
    )

    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], device=device)

    keep_ref = nms_ref(boxes, scores, 0.5)
    keep_conch = nms_conch(boxes, scores, 0.5)

    # All boxes should be kept since they don't overlap
    torch.testing.assert_close(keep_ref, keep_conch)
    assert len(keep_ref) == boxes.size(0)
    assert len(keep_conch) == boxes.size(0)
