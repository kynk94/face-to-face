import os
from typing import Any, Callable, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from numpy import ndarray
from torch import Tensor

from f2f.core.exceptions import FaceNotFoundError
from f2f.detection.s3fd.onnx import S3FDONNX
from f2f.detection.scrfd.onnx import SCRFDONNX
from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)
from f2f.utils.onnx_ops import OnnxExport

SYNTHETIC_LANDMARKS_URL = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/synthetic_resnet50d-01b3d553.ckpt"


@OnnxExport()
def onnx_export() -> None:
    model = SyntheticLM2dInference()
    input = torch.randn(1, 3, 256, 256)
    print(f"Exporting {model._get_name()} ONNX...")
    print(f"Use Input: {input.size()}")
    file_name = (
        os.path.splitext(os.path.basename(SYNTHETIC_LANDMARKS_URL))[0] + ".onnx"
    )
    onnx_path = os.path.join(get_onnx_cache_dir(), file_name)
    torch.onnx.export(
        model,
        input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["landmark"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "landmarks": {0: "batch_size"},
        },
    )
    exported_path = rename_file_with_hash(onnx_path)
    print(f"Exported to {exported_path}")


class SyntheticLM2d(nn.Module):
    """
    Landmark 2d model trained on synthetic data.
    Can compute gradient if needed.
    """

    resolution: int = 256

    def __init__(self, checkpoint: Optional[str] = None) -> None:
        super().__init__()
        self.backbone = timm.create_model("resnet50d", num_classes=68 * 2)

        if checkpoint is not None:
            checkpoint = url_to_local_path(checkpoint)
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
            self.load_state_dict(
                torch.load(checkpoint, map_location="cpu")["state_dict"]
            )
            print(
                f"Loaded checkpoint from {checkpoint}. "
                "Need to call `eval()` and `requires_grad_(False)` manually."
            )

    def assert_resolution(self, input: Tensor) -> None:
        if (
            input.size(-2) != self.resolution
            or input.size(-1) != self.resolution
        ):
            raise ValueError(
                f"Expected input size ({self.resolution}, {self.resolution}), "
                f"got {input.size(-2)}, {input.size(-1)}"
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: (N, 3, 256, 256) RGB image in range [-1, 1]
        Returns:
            landmarks: (N, 68, 2) landmarks in range [-1, 1]
        """
        self.assert_resolution(input)
        landmarks: Tensor = self.backbone(input)
        return landmarks.view(-1, 68, 2)


class SyntheticLM2dInference(SyntheticLM2d):
    """
    Landmark 2d model trained on synthetic data.
    If need gradient, use `SyntheticLM2d` instead.
    """

    # fmt: off
    flip_parts: Tuple[Tuple[int, int], ...] = (
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),  # jaw  # noqa
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # eyebrow
        (31, 35), (32, 34),  # nose
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # eye
        (48, 54), (49, 53), (50, 52), (61, 63), (60, 64), (67, 65), (58, 56), (59, 55)  # mouth  # noqa
    )
    # fmt: on

    def __init__(self, pretrained_path: str = SYNTHETIC_LANDMARKS_URL) -> None:
        super().__init__(pretrained_path)
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "SyntheticLM2dInference":
        return super().train(False)

    @torch.no_grad()
    def flip_forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.backbone(input.flip(dims=(-1,)))
        output = output.view(-1, 68, 2)
        output[..., 0] *= -1
        s, t = zip(*self.flip_parts)
        temp = output[:, t].clone()
        output[:, t] = output[:, s]
        output[:, s] = temp
        return output

    @torch.no_grad()
    def forward(self, input: Tensor, use_flip: bool = False) -> Tensor:
        """
        Args:
            input: (N, 3, 256, 256) bbox aligned RGB image in range [-1, 1]
            use_flip: whether to use flip augmentation
        Returns:
            landmarks: (N, 68, 2) landmarks in range [-1, 1]
        """
        self.assert_resolution(input)
        landmarks: Tensor = self.backbone(input)
        landmarks = landmarks.view(-1, 68, 2)
        if use_flip:
            fliped_landmarks = self.flip_forward(input)
            landmarks = (landmarks + fliped_landmarks) / 2
        return landmarks


class FDSyntheticLM2dInference(SyntheticLM2dInference):
    """
    Face detection + Landmark 2d model trained on synthetic data.
    """

    FD_model: Union[S3FDONNX, SCRFDONNX]

    def __init__(
        self,
        detection_model: str = "scrfd",
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.detection_model = detection_model.strip().lower()
        if self.detection_model == "s3fd":
            self.FD_model = S3FDONNX(threshold=threshold)
        elif self.detection_model == "scrfd":
            self.FD_model = SCRFDONNX(threshold=threshold)
        else:
            raise ValueError(f"Unsupported detection model: {detection_model}")

    @property
    def threshold(self) -> float:
        return self.FD_model.threshold

    def _apply(self, fn: Callable[..., Any]) -> "FDSyntheticLM2dInference":
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                null_tensor = torch.empty(0)
                device = getattr(fn(null_tensor), "device", "cpu")
            self.FD_model.to(device)
        return super()._apply(fn)

    @torch.no_grad()  # type: ignore
    def forward(
        self, input: Tensor, use_flip: bool = False
    ) -> Tuple[List[ndarray], List[Tensor]]:
        """
        Args:
            input: (N, 3, H, W) RGB image in range [-1, 1]
            use_flip: whether to use flip augmentation
        Returns:
            bboxes: list of (F, 5) bboxes in range [0, 1] where F is the \
                number of faces of each image
            landmarks: list of (F, 68, 2) landmarks in range [-1, 1] where F \
                is the number of faces of each image
        """
        N = input.size(0)
        np_input = input.cpu().permute(0, 2, 3, 1).mul(127.5).add(127.5).numpy()
        batch_bboxes = []
        batch_landmarks = []
        for i in range(N):
            if isinstance(self.FD_model, SCRFDONNX):
                bboxes, lm5s = self.FD_model(np_input[i])
            else:  # s3fd
                bboxes = self.FD_model(np_input[i])
            if bboxes is None:
                raise FaceNotFoundError(f"Face not found in image {i}")
            batch_bboxes.append(bboxes)
            batch_landmarks.append(
                self.single_landmarks(input[i], bboxes, use_flip)
            )
        return batch_bboxes, batch_landmarks

    def single_landmarks(
        self, input: Tensor, bboxes: ndarray, use_flip: bool = False
    ) -> Tensor:
        """
        Args:
            input: (3, H, W) RGB image in range [-1, 1]
            bboxes: (F, 5) bounding boxes where F is the number of faces
            use_flip: whether to use flip augmentation
        Returns:
            landmarks: (F, 68, 2) landmarks in input resolution where F is the \
                number of faces
        """
        H, W = input.size(-2), input.size(-1)

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
            crop = F.pad(input, (-L_crop, R_crop - W, -T_crop, B_crop - H))
            # This use torchvision resize not PIL, the result is slightly
            # different with onnx version
            resized = TVF.resize(
                crop.unsqueeze(0),
                size=(self.resolution, self.resolution),
                antialias=True,
            )
            face_landmarks = self.backbone(resized).view(-1, 68, 2)
            if use_flip:
                face_landmarks = (
                    face_landmarks + self.flip_forward(resized)
                ) / 2
            face_landmarks = (face_landmarks + 1) * crop_size / 2
            face_landmarks[..., 0] = face_landmarks[..., 0] + L_crop
            face_landmarks[..., 1] = face_landmarks[..., 1] + T_crop
            landmarks.append(face_landmarks)
        return torch.cat(landmarks, dim=0)
