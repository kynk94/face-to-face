from typing import Tuple, Union

import numpy as np
from numpy import ndarray
from PIL import Image

from f2f.core.exceptions import FaceNotFoundError
from f2f.core.onnx import BaseONNX
from f2f.detection.s3fd.onnx import S3FDONNX
from f2f.detection.scrfd.onnx import SCRFDONNX

ONNX_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/synthetic_resnet50d-01b3d553-f86e186d.onnx"


class FDSyntheticLM2dONNX(BaseONNX):
    # fmt: off
    flip_parts: Tuple[Tuple[int, int], ...] = (
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),  # jaw  # noqa
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # eyebrow
        (31, 35), (32, 34),  # nose
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # eye
        (48, 54), (49, 53), (50, 52), (61, 63), (60, 64), (67, 65), (58, 56), (59, 55)  # mouth  # noqa
    )
    # fmt: on
    resolution: int = 256
    FD_model: Union[S3FDONNX, SCRFDONNX]

    def __init__(
        self,
        detection_model: str = "scrfd",
        threshold: float = 0.5,
        onnx_path: str = str(ONNX_PATH),
        device: Union[str, int] = "cpu",
    ) -> None:
        self.detection_model = detection_model.lower()
        if self.detection_model == "s3fd":
            self.FD_model = S3FDONNX(threshold=threshold, device=device)
        elif self.detection_model == "scrfd":
            self.FD_model = SCRFDONNX(threshold=threshold, device=device)
        else:
            raise ValueError(f"Unsupported detection model: {detection_model}")
        super().__init__(onnx_path, device=device)

    @property
    def threshold(self) -> float:
        return self.FD_model.threshold

    def to(self, device: Union[str, int]) -> "FDSyntheticLM2dONNX":
        super().to(device)
        self.FD_model.to(device)
        return self

    def __call__(
        self, input: Union[ndarray, Image.Image], use_flip: bool = False
    ) -> Tuple[ndarray, ndarray]:
        """
        Args:
            input: (H, W, 3), RGB image in range [0, 255] or PIL.Image
            use_flip: whether to use flip augmentation
        Returns:
            bboxes: (F, 5) bounding boxes where F is the number of faces
            landmarks: (F, 68, 2) landmarks where F is the number of faces
        """
        if isinstance(self.FD_model, SCRFDONNX):
            bboxes, lm5s = self.FD_model(input)
        else:  # s3fd
            bboxes = self.FD_model(input)
        if bboxes is None:
            raise FaceNotFoundError("Face not found in image")
        landmarks = self.single_landmarks(input, bboxes, use_flip)
        return bboxes, landmarks

    def single_landmarks(
        self,
        input: Union[ndarray, Image.Image],
        bboxes: ndarray,
        use_flip: bool = False,
    ) -> ndarray:
        """
        Args:
            input: (H, W, 3) RGB image in range [0, 255]
            bboxes: (F, 5) bounding boxes where F is the number of faces
            use_flip: whether to use flip augmentation
        Returns:
            landmarks: (F, 68, 2) landmarks in input resolution where F is the \
                number of faces
        """
        if isinstance(input, ndarray):
            input = Image.fromarray(input.astype(np.uint8))

        landmarks = []
        for bbox in bboxes:
            L, T, R, B, score = bbox
            bbox_W = R - L
            bbox_H = B - T

            W_center = (L + R) / 2
            H_center = (T + B) / 2
            crop_size = max(bbox_W, bbox_H) * 1.5

            L_crop = int(W_center - crop_size / 2)
            T_crop = int(H_center - crop_size / 2)
            R_crop = int(W_center + crop_size / 2)
            B_crop = int(H_center + crop_size / 2)
            crop = input.crop((L_crop, T_crop, R_crop, B_crop))
            # This use PIL resize not torchvision, the result is slightly
            # different with torch version
            resized = crop.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )
            session_input = (
                np.array(resized, dtype=np.float32).transpose(2, 0, 1)[None]
                / 255.0
            )
            face_landmarks = self.session_run(session_input)[0].reshape(
                -1, 68, 2
            )
            if use_flip:
                flip_input = np.flip(session_input, axis=-1)
                flip_landmarks = self.session_run(flip_input)[0].reshape(
                    -1, 68, 2
                )
                face_landmarks = (
                    face_landmarks + self.flip_landmark(flip_landmarks)
                ) / 2
            face_landmarks = (face_landmarks + 1) * crop_size / 2
            face_landmarks[..., 0] = face_landmarks[..., 0] + L_crop
            face_landmarks[..., 1] = face_landmarks[..., 1] + T_crop
            landmarks.append(face_landmarks)
        return np.concatenate(landmarks, axis=0)

    def flip_landmark(self, landmark: ndarray) -> ndarray:
        """
        Args:
            landmark: (N, 68, 2) landmarks in range [-1, 1]
        Returns:
            (N, 68, 2) landmarks in range [-1, 1]
        """
        landmark[..., 0] *= -1
        s, t = zip(*self.flip_parts)
        temp = landmark[:, t].copy()
        landmark[:, t] = landmark[:, s]
        landmark[:, s] = temp
        return landmark
