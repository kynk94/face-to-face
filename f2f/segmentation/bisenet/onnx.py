from typing import Dict, Sequence, Union

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from PIL import Image

from f2f.core.onnx import BaseONNX
from f2f.utils.image import as_rgb_ndarray

ONNX_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/bisenet_celeba-468e13ca-60a13035.onnx"
CELEBA_PARTS = {
    "background": 0,
    "skin": 1,
    "l_brow": 2,
    "r_brow": 3,
    "l_eye": 4,
    "r_eye": 5,
    "eye_glass": 6,
    "l_ear": 7,
    "r_ear": 8,
    "ear_ring": 9,
    "nose": 10,
    "mouth": 11,
    "u_lip": 12,
    "l_lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}


class FaceBiSeNetONNX(BaseONNX):
    resolution: int = 512
    """trained on 448x448 random crops from 512x512 resized CelebAMask-HQ"""

    def __init__(
        self,
        onnx_path: str = ONNX_PATH,
        device: Union[str, int] = "cpu",
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        super().__init__(onnx_path, device, dtype)

    @property
    def parts(self) -> Dict[str, int]:
        copy_parts = CELEBA_PARTS.copy()
        return copy_parts

    def __call__(self, input: Union[ndarray, Image.Image]) -> ndarray:
        """
        Args:
            input: (512, 512, 3), RGB image in range [0, 255] or PIL.Image
        Returns:
            segmentation: (512, 512) segmentation mask in range [0, 18], \
                background + 18 classes
        """
        input = as_rgb_ndarray(input)
        session_input = input.transpose(2, 0, 1)[None] / 127.5 - 1.0
        segmentation = self.session_run(session_input)[0]
        return segmentation[0, 0]

    def segmentation_to_mask(
        self, segmentation: ndarray, parts: Sequence[Union[str, int]]
    ) -> ndarray:
        """
        Args:
            segmentation: (H, W) segmentation mask in range [0, 18], \
                background + 18 classes
            parts: list of part names or indices to include in the mask
        Returns:
            mask: (H, W) boolean mask
        """
        mask = np.zeros_like(segmentation, dtype=bool)
        for part in parts:
            if isinstance(part, str):
                part = self.parts[part]
            mask[segmentation == part] = True
        return mask
