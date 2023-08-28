"""
S3FD from face_alignment https://github.com/1adrianb/face-alignment
face-alignment is licensed under the BSD 3-Clause License.
"""
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from PIL import Image

from f2f.core.onnx import BaseONNX
from f2f.detection.utils import cal_order_by_area, nms, s3fd_predictions
from f2f.utils.image import as_rgb_ndarray, square_resize

ONNX_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/s3fd-619a316812-ece4f2c1.onnx"


class S3FDONNX(BaseONNX):
    resolution: int = 640
    """input resolution (training resolution is 640)"""
    nms_threshold: float = 0.3

    def __init__(
        self,
        threshold: float = 0.5,
        onnx_path: str = ONNX_PATH,
        device: Union[str, int] = "cpu",
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        super().__init__(onnx_path, device, dtype)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in range [0.0, 1.0]")
        self.threshold = threshold

    def __call__(
        self, input: Union[ndarray, Image.Image], center_weight: float = 0.3
    ) -> Optional[ndarray]:
        """
        S3FD inference, only support single image inference.
        In test, there is no speed advantage when operating in batches.

        Args:
            input: (H, W, 3), RGB image in range [0, 255] or PIL.Image
            center_weight: weight for center face
        Returns:
            bboxes: None if no faces or (F, 5) bounding boxes, where F is the \
                number of faces
        """
        input = as_rgb_ndarray(input)
        H, W = input.shape[:2]

        ratio = self.resolution / max(H, W)
        resized_input = square_resize(input, self.resolution)

        # HWC to NCHW, normalize
        session_input = resized_input.transpose(2, 0, 1)[None] / 127.5 - 1.0
        s3fd_outputs = self.session_run(session_input)

        bboxes = s3fd_predictions(s3fd_outputs)[0]  # (F, 5)
        if bboxes.size == 0:
            return None

        # rescale bboxes to original image size
        bboxes[:, :4] /= ratio

        keep = nms(bboxes, self.nms_threshold)
        bboxes = bboxes[keep]

        # thresholding by score
        if self.threshold > 0.0:
            bboxes = bboxes[bboxes[:, -1] > self.threshold]
            if bboxes.size == 0:
                return None

        # sort by area and center
        area_order = cal_order_by_area(
            height=H,
            width=W,
            bboxes=bboxes,
            center_weight=center_weight,
        )
        return bboxes[area_order]
