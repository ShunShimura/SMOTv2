from .ultralytics import YOLOv10
from .ultralytics.engine.results import Results
import numpy as np

__all__ = [
    'YOLOv10',
    'Results',
    'non_maximum_suppression'
]

def non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on a set of bounding boxes.
    
    Args:
        boxes (np.ndarray): Array of bounding boxes, shape (N, 4). Each box is [x1, y1, x2, y2].
        scores (np.ndarray): Array of confidence scores, shape (N,).
        iou_threshold (float): IoU threshold for suppressing boxes.
        
    Returns:
        np.ndarray: Indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # Coordinates of bounding boxes
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Areas of bounding boxes
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by score in descending order

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the kept box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute width and height of overlap
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Intersection over Union (IoU)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep only boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)