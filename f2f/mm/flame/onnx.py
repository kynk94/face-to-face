import pickle
from typing import Optional, Tuple, Union, cast

import numpy as np
from numpy import ndarray

from f2f.core.onnx import BaseONNX
from f2f.mm.flame.utils import FLAME_MODELS, TEXTURE_COORD, load_flame
from f2f.utils import url_to_local_path

ONNX_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2020_generic-ffd4033d-28ea75c4.onnx"


class FLAMEONNX(BaseONNX):
    f: ndarray
    """shape (9976, 3) obj f, triangle faces"""
    v: ndarray
    """shape (5023, 3) obj v, vertex positions"""
    ft: ndarray
    """shape (9976, 3) obj ft, texture face indices"""
    vt: ndarray
    """shape (5023, 2) obj vt, texture coordinates"""

    def __init__(self, device: Union[str, int] = "cpu") -> None:
        super().__init__(ONNX_PATH, device)
        flame_model = load_flame(FLAME_MODELS._2020.value)
        with open(url_to_local_path(TEXTURE_COORD), "rb") as f:
            texture_coord = pickle.load(f, encoding="latin1")  # noqa: S301
        self.f = flame_model["f"].astype(np.int64)
        self.v = flame_model["v_template"].astype(np.float32)
        self.ft = texture_coord["ft"].astype(np.int64)
        self.vt = texture_coord["vt"].astype(np.float32)

    def __call__(
        self,
        identity_params: Optional[ndarray] = None,
        expression_params: Optional[ndarray] = None,
        global_rotation: Optional[ndarray] = None,
        neck_rotation: Optional[ndarray] = None,
        jaw_rotation: Optional[ndarray] = None,
        eyes_rotation: Optional[ndarray] = None,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Args:
            identity_params: (N, <=300)
            expression_params: (N, <=100)
            global_rotation: (N, 3)
            neck_rotation: (N, 3)
            jaw_rotation: (N, 3)
            eyes_rotation: (N, 6)
        Returns:
            vertices: (N, 5023, 3)
            dynamic_lm17: (N, 17, 3)
            static_lm68: (N, 68, 3)
        """
        N = 1
        sizes = (300, 100, 3, 3, 3, 6)
        inputs = (
            identity_params,
            expression_params,
            global_rotation,
            neck_rotation,
            jaw_rotation,
            eyes_rotation,
        )
        for input in inputs:
            if input is None:
                continue
            N = input.shape[0]
            break

        session_inputs = []
        for index, input in enumerate(inputs):
            size = sizes[index]
            if input is None:
                session_inputs.append(np.zeros((N, size), dtype=np.float32))
            elif input.shape[-1] > size:
                raise ValueError(
                    f"input {index} shape {input.shape} exceeds size {size}"
                )
            else:
                session_inputs.append(input)
        outputs = self.session_run(*session_inputs)
        return cast(Tuple[ndarray, ndarray, ndarray], tuple(outputs))
