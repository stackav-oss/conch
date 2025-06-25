# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test non-max suppression."""

import pytest
import torch

from conch.reference.vision.nms import nms as nms_ref


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


# @pytest.mark.parametrize("num_boxes", [1000])
@pytest.mark.parametrize("num_boxes", [10])
# @pytest.mark.parametrize("iou_threshold", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("iou_threshold", [0.2])
# @pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("seed", range(1))
def test_nms(
    num_boxes: int,
    iou_threshold: float,
    seed: int,
) -> None:
    """Test non-max suppression."""
    torch.random.manual_seed(seed)
    boxes, scores = _create_tensors_with_iou(num_boxes, iou_threshold)
    # keep_ref = self._reference_nms(boxes, scores, iou)
    keep = nms_ref(boxes, scores, iou_threshold)
    print(f"{boxes = }, {scores = }, {keep = }")
    assert False
    # torch.testing.assert_close(keep, keep_ref))
