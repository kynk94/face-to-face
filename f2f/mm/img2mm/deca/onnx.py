import math
from typing import Any, Dict, Union

import numpy as np
from numpy import ndarray
from PIL import Image

from f2f.core.onnx import BaseONNX
from f2f.detection.scrfd.onnx import SCRFDONNX
from f2f.utils.image import ARCFACE_LM5, umeyama_transform

EMOCA_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/EMOCA_v2_lr_mse_20_E-c797dff4.onnx"
MICA_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/mica-41b3e8ce-44c80379.onnx"


class EMOCAONNX(BaseONNX):
    resolution: int = 224

    def __init__(self, device: Union[str, int] = "cpu") -> None:
        self.FD = SCRFDONNX(device=device)
        super().__init__(EMOCA_PATH, device)

    def to(self, device: Union[str, int]) -> "EMOCAONNX":
        super().to(device)
        self.FD.to(device)
        return self

    def __call__(
        self, input: Union[ndarray, Image.Image]
    ) -> Dict[str, ndarray]:
        """
        Args:
            input: (H, W, 3), RGB image in range [0, 255] or PIL.Image
        Returns:
            outputs: dict of bbox, landmarks, emoca_outputs
                {
                    bboxes: (F, 5), F is the number of faces
                    crop_bboxes: (F, 4),
                    lm5: (F, 5, 2),
                    identity: (F, 100),
                    texture: (F, 50),
                    expression: (F, 50),
                    global_rotation: (F, 3),
                    jaw_rotation: (F, 3),
                    scale: (F, 1),
                    xy_translation: (F, 3),
                    light: (F, 9, 3),
                    detail: Optional, (F, 128),
                }
        """
        bboxes, lm5 = self.FD(input)
        crop_bboxes = self.calc_crop_by_bbox(bboxes)
        outputs: Dict[str, Any] = {
            "bboxes": bboxes,
            "crop_bboxes": crop_bboxes,
            "lm5": lm5,
        }
        for output_name in self.output_names:
            outputs[output_name] = []

        if isinstance(input, ndarray):
            input = Image.fromarray(input.astype(np.uint8))
        input = input.convert("RGB")
        for crop_bbox in crop_bboxes:
            crop = input.crop(crop_bbox)
            if crop.size != (self.resolution, self.resolution):
                crop = crop.resize(
                    (self.resolution, self.resolution), Image.Resampling.LANCZOS
                )
            session_input = (
                np.array(crop, dtype=np.float32).transpose(2, 0, 1)[None]
                / 255.0
            )
            emoca_outputs = self.session_run(session_input)

            # append outputs
            for k, v in zip(self.output_names, emoca_outputs):
                outputs[k].append(v)
        for output_name in self.output_names:
            outputs[output_name] = np.concatenate(outputs[output_name], axis=0)
        return outputs

    def calc_crop_by_bbox(self, bboxes: ndarray) -> ndarray:
        """
        Args:
            bboxes: (F, 5), F is the number of faces
        Returns:
            crop_bboxes: (F, 4)
        """
        if bboxes.ndim == 1:
            bboxes = bboxes[None]
        crop_bboxes = []
        for bbox in bboxes:
            L, T, R, B, *_ = bbox
            center_w = (R + L) / 2.0
            # Since SCRFD includes forehead, bottom should be weighted more.
            center_h = B * 0.6 + T * 0.4
            old_size = (R - L) * 1.1
            crop_size = old_size * 1.25

            CL = int(center_w - crop_size / 2.0)
            CT = int(center_h - crop_size / 2.0)
            CR = int(CL + math.ceil(crop_size))
            CB = int(CT + math.ceil(crop_size))
            crop_bbox = np.array((CL, CT, CR, CB), dtype=np.float32)
            crop_bboxes.append(crop_bbox)
        return np.stack(crop_bboxes, axis=0)

    def calc_crop_by_lm68(self, landmarks: ndarray) -> ndarray:
        """
        Args:
            landmarks: (F, 68, 2), F is the number of faces
        Returns:
            crop_bboxes: (F, 4)
        """
        crop_bboxes = []
        for lm in landmarks:
            L, T = lm.min(axis=0)
            R, B = lm.max(axis=0)
            center_w = (R + L) / 2.0
            center_h = (B + T) / 2.0
            old_size = (R - L + B - T) / 2 * 1.1
            crop_size = old_size * 1.25

            CL = int(center_w - crop_size / 2.0)
            CT = int(center_h - crop_size / 2.0)
            CR = int(CL + math.ceil(crop_size))
            CB = int(CT + math.ceil(crop_size))
            crop_bbox = np.array((CL, CT, CR, CB), dtype=np.float32)
            crop_bboxes.append(crop_bbox)
        return np.stack(crop_bboxes, axis=0)


class MICAONNX(BaseONNX):
    resolution: int = 112

    def __init__(self, device: Union[str, int] = "cpu") -> None:
        self.FD = SCRFDONNX(device=device)
        super().__init__(MICA_PATH, device)

    def to(self, device: Union[str, int]) -> "MICAONNX":
        super().to(device)
        self.FD.to(device)
        return self

    def __call__(
        self, input: Union[ndarray, Image.Image]
    ) -> Dict[str, ndarray]:
        """
        Args:
            input: (H, W, 3), RGB image in range [0, 255] or PIL.Image
        Returns:
            outputs: dict of bbox, landmarks, emoca_outputs
                {
                    bboxes: (F, 5), F is the number of faces
                    lm5: (F, 5, 2),
                    affine_matrix: (F, 3, 3),
                    identity: (F, 300),
                    arcface: (F, 512),
                }
        """
        bboxes, lm5 = self.FD(input)
        outputs: Dict[str, Any] = {
            "bboxes": bboxes,
            "lm5": lm5,
        }
        for output_name in self.output_names:
            outputs[output_name] = []

        if isinstance(input, ndarray):
            input = Image.fromarray(input.astype(np.uint8))
        input = input.convert("RGB")
        for _lm5 in lm5:
            M = umeyama_transform(_lm5, ARCFACE_LM5, estimate_scale=True)
            inv_M = np.linalg.inv(M).ravel()
            crop = input.transform(
                (self.resolution, self.resolution),
                Image.Transform.AFFINE,
                inv_M[:6],
                resample=Image.Resampling.BILINEAR,
            )
            session_input = (
                np.array(crop, dtype=np.float32).transpose(2, 0, 1)[None]
                / 127.5
                - 1.0
            )
            mica_outputs = self.session_run(session_input)
            outputs["affine_matrix"].append(M)
            for k, v in zip(self.output_names, mica_outputs):
                outputs[k].append(v)
        outputs["affine_matrix"] = np.stack(outputs["affine_matrix"], axis=0)
        for output_name in self.output_names:
            outputs[output_name] = np.concatenate(outputs[output_name], axis=0)
        return outputs


class EMICAONNX(EMOCAONNX):
    def __init__(self, device: Union[str, int] = "cpu") -> None:
        self.mica = MICAONNX(device=device)
        super().__init__(device)

    def to(self, device: Union[str, int]) -> "EMICAONNX":
        super().to(device)
        self.mica.to(device)
        return self

    def __call__(
        self, input: Union[ndarray, Image.Image]
    ) -> Dict[str, ndarray]:
        """
        Args:
            input: (H, W, 3), RGB image in range [0, 255] or PIL.Image
        Returns:
            outputs: dict of bbox, landmarks, emoca_outputs, mica_outputs
                {
                    bboxes: (F, 5), F is the number of faces
                    crop_bboxes: (F, 4),
                    lm5: (F, 5, 2),
                    identity: (F, 100),
                    texture: (F, 50),
                    expression: (F, 50),
                    global_rotation: (F, 3),
                    jaw_rotation: (F, 3),
                    scale: (F, 1),
                    xy_translation: (F, 3),
                    light: (F, 9, 3),
                    detail: Optional, (F, 128),
                    affine_matrix: (F, 3, 3),
                    mica_identity: (F, 300),
                    mica_arcface: (F, 512),
                }
        """
        emoca_outputs = super().__call__(input)
        mica_output_names = tuple(
            f"mica_{name}" for name in self.mica.output_names
        )
        mica_outputs: Dict[str, Any] = {
            output_name: []
            for output_name in ("affine_matrix", *mica_output_names)
        }

        if isinstance(input, ndarray):
            input = Image.fromarray(input.astype(np.uint8))
        input = input.convert("RGB")
        resolution = self.mica.resolution
        for lm5 in emoca_outputs["lm5"]:
            M = umeyama_transform(lm5, ARCFACE_LM5, estimate_scale=True)
            inv_M = np.linalg.inv(M).ravel()
            crop = input.transform(
                (resolution, resolution),
                Image.Transform.AFFINE,
                inv_M[:6],
                resample=Image.Resampling.BILINEAR,
            )
            session_input = (
                np.array(crop, dtype=np.float32).transpose(2, 0, 1)[None]
                / 127.5
                - 1.0
            )
            outputs = self.mica.session_run(session_input)
            mica_outputs["affine_matrix"].append(M)
            for k, v in zip(mica_output_names, outputs):
                mica_outputs[k].append(v)

        mica_outputs["affine_matrix"] = np.stack(
            mica_outputs["affine_matrix"], axis=0
        )
        for output_name in mica_output_names:
            mica_outputs[output_name] = np.concatenate(
                mica_outputs[output_name], axis=0
            )
        return {**emoca_outputs, **mica_outputs}
