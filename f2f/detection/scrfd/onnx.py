"""
SCRFD from insightface: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
insightface is licensed under the Apache License 2.0.

SCRFD: https://arxiv.org/abs/2105.04714
Backbone of SCRFD: ResNet-50-D https://arxiv.org/abs/1812.01187
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from PIL import Image

from f2f.core.onnx import BaseONNX
from f2f.detection.utils import (
    cal_order_by_area,
    distance2bbox,
    distance2kps,
    nms,
)
from f2f.utils.image import as_rgb_ndarray, square_resize

ONNX_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/scrfd_10g_bnkps-404e7c3a.onnx"


class SCRFDONNX(BaseONNX):
    resolution: int = 640
    conf_threshold: float = 0.02
    nms_threshold: float = 0.4
    fmc: int = 3
    _feat_stride_fpn: List[int] = [8, 16, 32]
    _num_anchors: int = 2

    def __init__(
        self,
        threshold: float = 0.5,
        onnx_path: str = ONNX_PATH,
        device: Union[str, int] = "cpu",
        use_padding_trick: bool = True,
    ) -> None:
        """
        Args:
            threshold: threshold for face detection
            onnx_path: path to onnx model
            device: device to run inference on
            use_padding_trick: add padding to bottom and right to support \
                images with a large face ratio. If False, detection rate of \
                images like FFHQ dataset is lower than 0.5.
        """
        super().__init__(onnx_path, device)
        self.anchor_centers = self.generate_anchor_centers()

        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in range [0.0, 1.0]")
        self.threshold = threshold
        self.use_padding_trick = use_padding_trick

    def generate_anchor_centers(self) -> Dict[int, ndarray]:
        anchor_centers: Dict[int, ndarray] = {}
        for stride in self._feat_stride_fpn:
            size = self.resolution // stride
            anchor_grid = np.mgrid[:size, :size][::-1] * stride
            anchor_center = np.repeat(anchor_grid, self._num_anchors, axis=-1)
            anchor_center = anchor_center.transpose(1, 2, 0).reshape(-1, 2)
            anchor_centers[stride] = anchor_center.astype(self.dtype)
        return anchor_centers

    def padding_trick(self, input: ndarray, ratio: float = 0.2) -> ndarray:
        """
        Pad the bottom and right side of the image to support images with a
        large face ratio.

        Args:
            input: (H, W, 3) RGB image in range [0, 255]
            ratio: padding ratio
        Returns:
            (H', W', 3) padded image in range [0, 255]
        """
        H, W = input.shape[:2]
        pad_size = int(min(H, W) * ratio)
        if abs(H - W) > pad_size:
            return input

        if H < W:
            pad_H = pad_size
            pad_W = 0
        elif H == W:
            pad_H = pad_W = pad_size
        else:
            pad_H = 0
            pad_W = pad_size
        return np.pad(input, ((0, pad_H), (0, pad_W), (0, 0)))

    def __call__(
        self, input: Union[ndarray, Image.Image], center_weight: float = 0.3
    ) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        """
        Args:
            input: (H, W, 3) RGB image in range [0, 255]
            center_weight: weight for center face
        Returns:
            bboxes: (N, 5) bounding boxes in format (x1, y1, x2, y2, score)
            keypoints: (N, 5, 2) keypoints in format (x, y)
        """
        input = as_rgb_ndarray(input)
        H_original, W_original = input.shape[:2]
        if self.use_padding_trick:
            input = self.padding_trick(input, ratio=0.2)
        H, W = input.shape[:2]

        ratio = self.resolution / max(H, W)
        resized_input = square_resize(input, self.resolution)

        # HWC to NCHW, normalize
        session_input = resized_input.transpose(2, 0, 1)[None] / 127.5 - 1.0
        bboxes, kpss = self.get_session_outputs(session_input)
        if bboxes.size == 0:
            return None, None

        # rescale outputs to original image size
        bboxes[:, :4] /= ratio
        kpss /= ratio

        # non maximum suppression
        keep = nms(bboxes, self.nms_threshold)
        bboxes = bboxes[keep]
        kpss = kpss[keep]

        # thresholding by score
        if self.threshold > 0.0:
            bboxes = bboxes[bboxes[:, -1] > self.threshold]
            if bboxes.size == 0:
                return None, None

        # sort by area and center
        area_order = cal_order_by_area(
            height=H_original,
            width=W_original,
            bboxes=bboxes,
            center_weight=center_weight,
        )
        reordered_bboxes = bboxes[area_order]
        kpss = kpss[area_order]
        return reordered_bboxes, kpss

    def get_session_outputs(self, input: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Args:
            input: (1, 3, resolution, resolution) RGB image in range [-1, 1]
                not support batch inference
        Returns:
            bboxes: ndarray (F_s08 + F_s16 + F_s32, 5)
            keypoints: ndarray (F_s08 + F_s16 + F_s32, 5, 2)
                F_s08: number of faces in stride 8
                F_s16: number of faces in stride 16
                F_s32: number of faces in stride 32
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []

        session_outputs = self.session_run(input)
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = session_outputs[idx][0]
            bbox_preds = session_outputs[idx + self.fmc][0] * stride
            kps_preds = session_outputs[idx + self.fmc * 2][0] * stride

            anchor_center = self.anchor_centers[stride]
            bboxes = distance2bbox(anchor_center, bbox_preds)
            kpss = distance2kps(anchor_center, kps_preds)

            pos_inds = np.where(scores >= self.conf_threshold)[0]
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)
        keypoints = np.vstack(kpss_list)
        return (
            np.hstack((bboxes, scores)),
            keypoints,
        )
