import pickle
from typing import Callable, Optional, Tuple, Union, cast

import numpy as np
from numpy import ndarray

from f2f.core.onnx import BaseONNX
from f2f.mm.flame.utils import FLAME_MODELS, TEXTURE_COORD, load_flame
from f2f.utils import url_to_local_path
from f2f.utils.transforms_np import batch_rodrigues, rotation_6d_to_matrix

ONNX_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2020_generic-ffd4033d-555f4c69.onnx"


class FLAMEONNX(BaseONNX):
    f: ndarray
    """shape (9976, 3) obj f, triangle faces"""
    v: ndarray
    """shape (5023, 3) obj v, vertex positions"""
    ft: ndarray
    """shape (9976, 3) obj ft, texture face indices"""
    vt: ndarray
    """shape (5118, 2) obj vt, texture coordinates"""

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
            global_rotation: (N, 3) in rodrigues vector or (N, 6) in 6d rotation
            neck_rotation: (N, 3) in rodrigues vector or (N, 6) in 6d rotation
            jaw_rotation: (N, 3) in rodrigues vector or (N, 6) in 6d rotation
            eyes_rotation: (N, 6) in rodrigues vector (3 left, 3 right)\
                or (N, 12) in 6d rotation (6 left, 6 right)
                axis direction:
                    x: head center -> left ear (head pitch axis)
                    y: head center -> top (head yaw axis)
                    z: head center -> forward of the face (head roll axis)
        Returns:
            vertices: (N, 5023, 3)
            dynamic_lm17: (N, 17, 3)
            static_lm68: (N, 68, 3)
        """
        N = 1
        for input in (
            identity_params,
            expression_params,
            global_rotation,
            neck_rotation,
            jaw_rotation,
            eyes_rotation,
        ):
            if input is None:
                continue
            N = input.shape[0]
            break

        # shape parameters
        if identity_params is None:
            identity_params = np.zeros((N, 300), dtype=np.float32)
        if expression_params is None:
            expression_params = np.zeros((N, 100), dtype=np.float32)
        # rotation parameters
        if eyes_rotation is None:
            left_eye = right_eye = None
        elif eyes_rotation.ndim == 4:  # rotation matrix fomula, (N, 2, 3, 3)
            left_eye, right_eye = eyes_rotation[:, 0], eyes_rotation[:, 1]
        else:  # rodrigues vector or 6d rotation, (N, 6) or (N, 12)
            left_eye, right_eye = np.split(eyes_rotation, 2, axis=-1)
        rotation_matrix = self.calculate_rotation_matrix(
            global_rotation,
            neck_rotation,
            jaw_rotation,
            left_eye,
            right_eye,
            batch_size=N,
        )
        outputs = self.session_run(
            identity_params,
            expression_params,
            rotation_matrix[:, 0],  # global rotation
            rotation_matrix[:, 1],  # neck rotation
            rotation_matrix[:, 2],  # jaw rotation
            rotation_matrix[:, 3:],  # left and right eye rotation
        )
        return cast(Tuple[ndarray, ndarray, ndarray], tuple(outputs))

    def calculate_rotation_matrix(
        self, *rotations: Optional[ndarray], batch_size: Optional[int] = None
    ) -> ndarray:
        """
        Calculate the rotation matrix from the given rotation parameters.

        Args:
            *rotations: (N, 3) or (N, 6) or (N, 3, 3) or None
            batch_size: batch size of the output rotation matrix
        Returns:
            rotation_matrix: (N, J, 3, 3), J is the number of input rotations
        """
        N = batch_size or 1
        get_zero_rotation = self.zero_rotation_matrix
        calculation: Optional[Callable[..., ndarray]] = None
        for rotation in rotations:
            if rotation is None:
                continue
            N = rotation.shape[0]
            if rotation.shape[1:] == (6,):
                get_zero_rotation = self.zero_rotation_6d
                calculation = rotation_6d_to_matrix
            elif rotation.shape[1:] == (3,):
                get_zero_rotation = self.zero_rotation_rodrigues
                calculation = batch_rodrigues
            elif rotation.shape[1:] != (3, 3):
                raise ValueError(
                    "rotation shape must be (N, 3) or (N, 6) or (N, 3, 3), "
                    f"but got {rotation.shape}"
                )
            break

        filled_rotations = tuple(
            r if r is not None else get_zero_rotation(N) for r in rotations
        )
        rotation_matrix = np.concatenate(filled_rotations, axis=0)
        if calculation is not None:
            rotation_matrix = calculation(rotation_matrix)
        return rotation_matrix.reshape(N, -1, 3, 3)

    def zero_rotation_matrix(self, batch_size: int) -> ndarray:
        """
        Returns:
            zero_rotation_matrix: (batch_size, 3, 3)
        """
        return np.tile(np.eye(3, dtype=np.float32), (batch_size, 1, 1))

    def zero_rotation_6d(self, batch_size: int) -> ndarray:
        """
        Returns:
            zero_rotation_6d: (batch_size, 6)
        """
        zero_rotation = np.array(((1, 0, 0, 0, 1, 0),), dtype=np.float32)
        return np.tile(zero_rotation, (batch_size, 1))

    def zero_rotation_rodrigues(self, batch_size: int) -> ndarray:
        """
        Returns:
            zero_rotation_rodrigues: (batch_size, 3)
        """
        return np.zeros((batch_size, 3), dtype=np.float32)
