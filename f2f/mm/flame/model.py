"""
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import os
import pickle
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor

from f2f.mm.flame.lbs import (
    linear_blend_skinning,
    rot_mat_to_euler,
    vertices2landmarks,
)
from f2f.mm.flame.utils import FLAME_MODELS, TEXTURE_COORD, load_flame
from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)
from f2f.utils.onnx_ops import OnnxExport
from f2f.utils.transforms_torch import batch_rodrigues, rotation_6d_to_matrix

LANDMARK_EMBEDDING = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame_landmark-8095348e.npy"
TEXTURE_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame_texture-1cb46349.npz"


@OnnxExport()
def onnx_export() -> None:
    model = FLAME().eval()
    identity_params = torch.randn(1, 300, dtype=torch.float32)
    expression_params = torch.randn(1, 100, dtype=torch.float32)
    rodrigues_vector = torch.randn(5, 3, dtype=torch.float32)
    rotation_matrix = batch_rodrigues(rodrigues_vector)
    global_rotation = rotation_matrix[0:1]  # (1, 3, 3)
    neck_rotation = rotation_matrix[1:2]  # (1, 3, 3)
    jaw_rotation = rotation_matrix[2:3]  # (1, 3, 3)
    eyes_rotation = rotation_matrix[3:5].unsqueeze(0)  # (1, 2, 3, 3)

    file_name = (
        os.path.splitext(os.path.basename(FLAME_MODELS._2020.value))[0]
        + ".onnx"
    )
    onnx_path = os.path.join(get_onnx_cache_dir(), file_name)
    torch.onnx.export(
        model,
        (
            identity_params,
            expression_params,
            global_rotation,
            neck_rotation,
            jaw_rotation,
            eyes_rotation,
        ),
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=[
            "identity_params",
            "expression_params",
            "global_rotation",
            "neck_rotation",
            "jaw_rotation",
            "eyes_rotation",
        ],
        output_names=["vertices", "dynamic_lm17", "static_lm68"],
        dynamic_axes={
            "identity_params": {0: "batch_size", 1: "n_identity_params"},
            "expression_params": {0: "batch_size", 1: "n_expression_params"},
            "global_rotation": {0: "batch_size"},
            "neck_rotation": {0: "batch_size"},
            "jaw_rotation": {0: "batch_size"},
            "eyes_rotation": {0: "batch_size"},
            "vertices": {0: "batch_size"},
            "dynamic_lm17": {0: "batch_size"},
            "static_lm68": {0: "batch_size"},
        },
    )
    exported_path = rename_file_with_hash(onnx_path)
    print(f"Exported to {exported_path}")


def to_tensor(
    array: Union[Tensor, npt.NDArray], dtype: torch.dtype = torch.float32
) -> Tensor:
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(dtype=dtype)
    return array.to(dtype=dtype)


def to_np(array: Any, dtype: npt.DTypeLike = np.float32) -> npt.NDArray:
    if isinstance(array, Tensor):
        return array.detach().cpu().numpy().astype(dtype)
    if isinstance(array, np.ndarray):
        return array.astype(dtype)

    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/flame_pytorch/flame.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs a mesh and 2D/3D facial landmarks
    """  # noqa: E501

    f: Tensor
    """shape (9976, 3) obj f, triangle faces"""
    v: Tensor
    """shape (5023, 3) obj v, vertex positions"""
    ft: Tensor
    """shape (9976, 3) obj ft, texture face indices"""
    vt: Tensor
    """shape (5118, 2) obj vt, texture coordinates"""
    identity_coeff: Tensor
    """shape (5023, 3, 300)"""
    expression_coeff: Tensor
    """shape (5023, 3, 100)"""
    rotation_coeff: Tensor
    """shape (36, 15069) = (12 * 3, 5023 * 3)"""
    joint_regressor: Tensor
    """shape (5, 5023) vertex to joint regressor"""
    parents: Tensor
    """
    shape (5,) [-1, 0, 1, 1, 1], (global, neck, jaw, eye_left, eye_right),
    describes the kinematic tree for the model
    """
    lbs_weights: Tensor
    """
    shape (5023, 5) linear blend skinning weights that represent how much the
    rotation matrix of each part affects each vertex"""
    dynamic_lm_faces_idx: Tensor
    """
    shape (79, 17) list of 17 contour face indexes for y angle (-39, 39) degrees
    """
    dynamic_lm_bary_coords: Tensor
    """
    shape (79, 17, 3) list of 17 contour barycentric weights for y angle
    (-39, 39) degrees
    """
    static_lm_faces_idx: Tensor
    """shape (68,)"""
    static_lm_bary_coords: Tensor
    """shape (68, 3)"""
    neck_kin_chain: Tensor
    """
    shape (2,) [1, 0], kinematic tree chain of the neck to consider relative
    rotation
    """

    def __init__(
        self,
        model: FLAME_MODELS = FLAME_MODELS._2020,
    ):
        super().__init__()
        if not isinstance(model, FLAME_MODELS):
            raise ValueError()
        flame_model = load_flame(model.value)
        with open(url_to_local_path(TEXTURE_COORD), "rb") as f:
            texture_coord = pickle.load(f, encoding="latin1")  # noqa: S301

        self.dtype = torch.float32
        self.register_buffer(
            "f",
            to_tensor(
                to_np(flame_model["f"], dtype=np.int64), dtype=torch.long
            ),
        )
        # The vertices of the template model
        self.register_buffer(
            "v", to_tensor(to_np(flame_model["v_template"]), dtype=self.dtype)
        )

        # The texture coordinates of the template model
        self.register_buffer(
            "ft", to_tensor(to_np(texture_coord["ft"]), dtype=torch.long)
        )
        self.register_buffer(
            "vt", to_tensor(to_np(texture_coord["vt"]), dtype=self.dtype)
        )

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model["shapedirs"]), dtype=self.dtype)
        identity_coeff, expression_coeff = (
            shapedirs[..., :300],
            shapedirs[..., 300:],
        )
        self.register_buffer("identity_coeff", identity_coeff)
        self.register_buffer("expression_coeff", expression_coeff)

        # The pose components
        num_pose_basis = flame_model["posedirs"].shape[-1]
        posedirs = np.reshape(flame_model["posedirs"], (-1, num_pose_basis)).T
        self.register_buffer(
            "rotation_coeff", to_tensor(to_np(posedirs), dtype=self.dtype)
        )

        self.register_buffer(
            "joint_regressor",
            to_tensor(to_np(flame_model["J_regressor"]), dtype=self.dtype),
        )
        parents = to_tensor(to_np(flame_model["kintree_table"][0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights",
            to_tensor(to_np(flame_model["weights"]), dtype=self.dtype),
        )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            url_to_local_path(LANDMARK_EMBEDDING),
            allow_pickle=True,
            encoding="latin1",
        ).item()
        self.register_buffer(
            "dynamic_lm_faces_idx",
            lmk_embeddings["dynamic_lmk_faces_idx"].long(),
        )
        self.register_buffer(
            "dynamic_lm_bary_coords",
            lmk_embeddings["dynamic_lmk_bary_coords"].to(self.dtype),
        )
        self.register_buffer(
            "static_lm_faces_idx",
            torch.from_numpy(lmk_embeddings["full_lmk_faces_idx"]).long(),
        )
        self.register_buffer(
            "static_lm_bary_coords",
            torch.from_numpy(lmk_embeddings["full_lmk_bary_coords"]).to(
                self.dtype
            ),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def forward(
        self,
        identity_params: Optional[Tensor] = None,
        expression_params: Optional[Tensor] = None,
        global_rotation: Optional[Tensor] = None,
        neck_rotation: Optional[Tensor] = None,
        jaw_rotation: Optional[Tensor] = None,
        eyes_rotation: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
            N = input.size(0)
            break
        if identity_params is None:
            identity_params = torch.zeros(
                1, dtype=self.dtype, device=self.v.device
            ).expand(N, -1)
        if expression_params is None:
            expression_params = torch.zeros(
                1, dtype=self.dtype, device=self.v.device
            ).expand(N, -1)
        shape_parameters = torch.cat(
            (identity_params, expression_params), dim=-1
        )
        shape_coeff = torch.cat(
            (
                self.identity_coeff[..., : identity_params.size(-1)],
                self.expression_coeff[..., : expression_params.size(-1)],
            ),
            dim=-1,
        )

        if eyes_rotation is None:
            left_eye = right_eye = None
        elif eyes_rotation.ndim == 4:  # rotation matrix fomula, (N, 2, 3, 3)
            left_eye, right_eye = eyes_rotation[:, 0], eyes_rotation[:, 1]
        else:  # rodrigues vector or 6d rotation, (N, 6) or (N, 12)
            left_eye, right_eye = torch.split(
                eyes_rotation, eyes_rotation.size(-1) // 2, dim=-1
            )
        rotation_matrix = self.calculate_rotation_matrix(
            global_rotation,
            neck_rotation,
            jaw_rotation,
            left_eye,
            right_eye,
            batch_size=N,
        )  # (N, 5, 3, 3)

        vertices, _ = linear_blend_skinning(
            shape_parameters=shape_parameters,
            rotations=rotation_matrix,
            shape_mean=self.v.expand(N, -1, -1),
            shape_coeff=shape_coeff,
            rotation_coeff=self.rotation_coeff,
            J_regressor=self.joint_regressor,
            parents=self.parents,
            lbs_weights=self.lbs_weights,
            dtype=self.dtype,
        )

        (
            dynamic_lm_faces_idx,
            dynamic_lm_bary_coords,
        ) = self._find_dynamic_lm_idx_and_bcoords(rotation_matrix)
        dynamic_lm17 = vertices2landmarks(
            vertices, self.f, dynamic_lm_faces_idx, dynamic_lm_bary_coords
        )
        static_lm68 = vertices2landmarks(
            vertices,
            self.f,
            self.static_lm_faces_idx.repeat(N, 1),
            self.static_lm_bary_coords.repeat(N, 1, 1),
        )
        return vertices, dynamic_lm17, static_lm68

    def _find_dynamic_lm_idx_and_bcoords(
        self, rotation_matrix: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Selects the face contour depending on the reletive rotations of the head
        Args:
            rotation_matrix: shape (N, J, 3, 3)
        Returns:
            dynamic_lm_faces_idx:
                shape (N, 17), the contour face indexes
            dynamic_lm_bary_coords:
                shape (N, 17, 3), the contour face barycentric weights
        """
        N = rotation_matrix.size(0)
        rot_mats = torch.index_select(rotation_matrix, 1, self.neck_kin_chain)

        rel_rot_mat = torch.eye(
            3, device=rotation_matrix.device, dtype=self.dtype
        ).expand(N, -1, -1)
        for idx in range(self.neck_kin_chain.size(0)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dynamic_lm_faces_idx = torch.index_select(
            self.dynamic_lm_faces_idx, 0, y_rot_angle
        )
        dynamic_lm_bary_coords = torch.index_select(
            self.dynamic_lm_bary_coords, 0, y_rot_angle
        )
        return dynamic_lm_faces_idx, dynamic_lm_bary_coords

    def calculate_rotation_matrix(
        self, *rotations: Optional[Tensor], batch_size: Optional[int] = None
    ) -> Tensor:
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
        calculation: Optional[Callable[..., Tensor]] = None
        for rotation in rotations:
            if rotation is None:
                continue
            N = rotation.size(0)
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
        rotation_matrix = torch.cat(filled_rotations, dim=0)
        if calculation is not None:
            rotation_matrix = calculation(rotation_matrix)
        return rotation_matrix.view(N, -1, 3, 3)

    def zero_rotation_matrix(self, batch_size: int) -> Tensor:
        """return (N, 3, 3) tensor"""
        eye = torch.eye(3, dtype=self.dtype, device=self.v.device)
        zero_rotation_matrix = eye.expand(batch_size, -1, -1)
        return zero_rotation_matrix

    def zero_rotation_6d(self, batch_size: int) -> Tensor:
        """return (N, 6) tensor"""
        eye = torch.eye(3, dtype=self.dtype, device=self.v.device)
        zero_rotation_6d = eye[:2].ravel().expand(batch_size, -1)
        return zero_rotation_6d

    def zero_rotation_rodrigues(self, batch_size: int) -> Tensor:
        """return (N, 3) tensor"""
        zero_rotation_rodrigues = torch.zeros(
            3, dtype=self.dtype, device=self.v.device
        ).expand(batch_size, -1)
        return zero_rotation_rodrigues


class FLAMETexture(nn.Module):
    """
    current FLAME texture are adapted from BFM Texture Model
    """

    mean: Tensor
    """(3, 512, 512) in RGB [0, 1]"""
    basis: Tensor
    """(3, 512, 512, 200) in RGB"""

    def __init__(self) -> None:
        super().__init__()
        # mean: (512, 512, 3), tex_dir: (512, 512, 3, 200)
        tex_space = np.load(url_to_local_path(TEXTURE_PATH))
        # BFM texture is in BGR order, flip to RGB
        mean = np.transpose(tex_space["mean"], (2, 0, 1)) / 127.5 - 1.0
        basis = np.transpose(tex_space["tex_dir"], (2, 0, 1, 3)) / 127.5
        mean = torch.from_numpy(np.flip(mean, 0).copy()).float()
        basis = torch.from_numpy(np.flip(basis, 0).copy()).float()
        self.register_buffer("mean", mean)
        self.register_buffer("basis", basis)

    def forward(self, texture_params: Tensor) -> Tensor:
        """
        texture_params: (N, <=200)
        return: (N, 3, 512, 512), in RGB [-1, 1], not clamped.
        """
        basis = self.basis[..., : texture_params.size(-1)]
        differences = torch.einsum("chwi,ni->nchw", basis, texture_params)
        return self.mean + differences
