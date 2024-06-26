import torch
import torch.nn as nn


def bbox_iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, eps: float) -> torch.Tensor:
    """
    :param bboxes1: Expected to be already in transposed format -> (4, N)
    :param bboxes2: Expected to be already in transposed format -> (4, N)
    :param eps: Param for preventing zero-division errors
    :return:
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps
    iou = inter_area / union_area

    return iou


def bbox_ciou(bboxes1: torch.Tensor, bboxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    bboxes1, bboxes2 should be tensors of shape (N, 4), with each box in (x1, y1, x2, y2) format
    """

    # transpose both to get xs and ys as vectors (below)
    bboxes1 = bboxes1.t()
    bboxes2 = bboxes2.t()
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2
    iou = bbox_iou(bboxes1, bboxes2, eps)
    b1_center_x = (b1_x1 + b1_x2) / 2
    b1_center_y = (b1_y1 + b1_y2) / 2
    b2_center_x = (b2_x1 + b2_x2) / 2
    b2_center_y = (b2_y1 + b2_y2) / 2
    center_distance = (b1_center_x - b2_center_x) ** 2 + (b1_center_y - b2_center_y) ** 2
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1
    aspect_ratio = 4 / (torch.pi ** 2) * torch.pow(torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps)), 2)
    v = aspect_ratio / (1 - iou + aspect_ratio + eps)
    ciou = iou - (center_distance / (enclose_diagonal + eps) + v)

    return ciou


class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ciou = bbox_ciou(preds, targets)
        loss = 1 - ciou

        return loss.mean()
