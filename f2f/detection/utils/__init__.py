from typing import List, Optional, Tuple

import numpy as np
from numpy import ndarray


def nms(bboxes: ndarray, threshold: float) -> List[int]:
    if len(bboxes) == 0:
        return []
    L, T, R, B = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (R - L + 1) * (B - T + 1)
    order = bboxes[:, 4].argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        LL = np.maximum(L[i], L[order[1:]])
        TT = np.maximum(T[i], T[order[1:]])
        RR = np.minimum(R[i], R[order[1:]])
        BB = np.minimum(B[i], B[order[1:]])

        w = np.maximum(0.0, RR - LL + 1)
        h = np.maximum(0.0, BB - TT + 1)
        area = w * h
        ovr = area / (areas[i] + areas[order[1:]] - area)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


def distance2bbox(
    center: ndarray,
    distance: ndarray,
    max_shape: Optional[Tuple[int, ...]] = None,
) -> ndarray:
    """Decode distance prediction to bounding boxes.
    Args:
        center: anchor center, (n, 2), [x, y].
        distance: Distance from the given point to (left, top, right, bottom).
        max_shape: Shape of the image.
    Returns:
        bboxes: Decoded bboxes, (F, 4).
    """
    x1 = center[:, 0] - distance[:, 0]
    y1 = center[:, 1] - distance[:, 1]
    x2 = center[:, 0] + distance[:, 2]
    y2 = center[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    center: ndarray,
    distance: ndarray,
    max_shape: Optional[Tuple[int, ...]] = None,
) -> ndarray:
    """Decode distance prediction to keypoints.
    Args:
        center: anchor center, (n, 2), [x, y].
        distance: Distance from the given point to 5 facial keypoints.
        max_shape: Shape of the image.
    Returns:
        keypoints: Decoded keypoints, (F, 5, 2).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = center[:, i % 2] + distance[:, i]
        py = center[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape(-1, 5, 2)


def cal_order_by_area(
    height: int,
    width: int,
    bboxes: ndarray,
    center_weight: float = 0.0,
) -> ndarray:
    """
    Calculate order of bbox by area.

    Args:
        height: Height of image.
        width: Width of image.
        bboxes: Shape (F, ge 4), num of bbox and [L, T, R, B, ...etc].
        center_weight: Weight of center point. 0.5 is recommended.
            if 0, sort by area.
            elif positive, weighting close to center.
            elif negative, weighting far from center.
    Returns:
        order: Order of bbox.
    """
    area = (
        np.multiply(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1])
        ** 0.5
    )
    # weighting area by score
    if bboxes.shape[-1] > 4:
        score = bboxes[:, 4]
        area *= score**2
    if center_weight == 0.0:
        return np.argsort(area)[::-1]

    center_distance = (
        bboxes[:, :2]
        + bboxes[:, 2:4]
        - np.array([[width, height]], dtype=np.float32)
    ) / 2
    center_distance = np.linalg.norm(center_distance, axis=1)
    return np.argsort(area - center_weight * center_distance)[::-1]


def decode(
    loc: ndarray, priors: ndarray, variances: Tuple[float, float]
) -> ndarray:
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors,4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def s3fd_predictions(olist: List[ndarray]) -> ndarray:
    """
    olist: [cls1, reg1, cls2, reg2, ...], list of 12 tensors, output of S3FD
    """
    bboxlists = []
    variances = (0.1, 0.2)
    batch_size = olist[0].shape[0]
    for j in range(batch_size):
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = (
                    stride / 2 + windex * stride,
                    stride / 2 + hindex * stride,
                )
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].copy().reshape(1, 4)
                priors = np.array(
                    [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
                )
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0]
                bboxlist.append([x1, y1, x2, y2, score])

        bboxlists.append(bboxlist)

    bboxlists = np.array(bboxlists)
    return bboxlists
