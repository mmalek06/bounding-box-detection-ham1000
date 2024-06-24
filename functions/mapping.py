import torch


def extract_bboxes(targets: list[dict]) -> list[torch.Tensor]:
    bboxes = []

    for target in targets:
        xs, ys, widths, heights = target["bbox"]

        for idx, _ in enumerate(xs):
            x1, y1, width, height = xs[idx], ys[idx], widths[idx], heights[idx]
            # Convert COCO format (x, y, width, height) to (x1, y1, x2, y2)
            x2, y2 = x1 + width, y1 + height

            bboxes.append(torch.IntTensor([x1, y1, x2, y2]))

    return bboxes
