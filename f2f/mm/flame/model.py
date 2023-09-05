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
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor

from f2f.mm.flame.lbs import (
    batch_rodrigues,
    linear_blend_skinning,
    rot_mat_to_euler,
    vertices2landmarks,
)
from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)
from f2f.utils.onnx_ops import OnnxExport


class FLAME_MODELS(Enum):
    _2020 = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2020_generic-efcd14cc.pkl"
    _2023 = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2023-8fb1af0d.pkl"
    _2023_NO_JAW = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2023_no_jaw-b291fcd1.pkl"


LANDMARK_EMBEDDING = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame_landmark-8095348e.npy"
TEXTURE_COORD = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame_texture_coord-f8d5f559.pkl"


@OnnxExport()
def onnx_export() -> None:
    model = FLAME().eval()
    identity_params = torch.zeros(1, 300, dtype=torch.float32)
    expression_params = torch.zeros(1, 100, dtype=torch.float32)
    global_rotation = torch.zeros(1, 3, dtype=torch.float32)
    neck_rotation = torch.zeros(1, 3, dtype=torch.float32)
    jaw_rotation = torch.zeros(1, 3, dtype=torch.float32)
    eyes_rotation = torch.zeros(1, 6, dtype=torch.float32)

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
    """shape (5023, 2) obj vt, texture coordinates"""
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
    zero_rotation: nn.Parameter
    """shape (1, 3)"""
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
        with open(url_to_local_path(model.value), "rb") as f:
            flame_model = pickle.load(f, encoding="latin1")  # noqa: S301
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

        zero_rotation = torch.zeros(
            [1, 3], dtype=self.dtype, requires_grad=False
        )
        self.register_buffer("zero_rotation", zero_rotation)

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
        args:
            identity_params: (N, <=300)
            expression_params: (N, <=100)
            global_rotation: (N, 3) in radians or (N, 4) in quaternions
            neck_rotation: (N, 3) in radians or (N, 4) in quaternions
            jaw_rotation: (N, 3) in radians or (N, 4) in quaternions
            eyes_rotation: (N, 6) in radians (3 for left, 3 for right) or \
                (N, 8) in quaternions
                axis direction:
                    x: head center -> left ear (head pitch axis)
                    y: head center -> top (head yaw axis)
                    z: head center -> forward of the face (head roll axis)
        return:
            vertices: (N, 5023, 3)
            dynamic_lm17: (N, 17, 3)
            static_lm68: (N, 68, 3)
        """
        if identity_params is None and expression_params is None:
            raise ValueError(
                "Either identity_params or expression_params should be provided"
            )
        if identity_params is None:
            expression_params = cast(Tensor, expression_params)
            N = expression_params.size(0)
            N_id = self.identity_coeff.size(-1)
            N_exp = expression_params.size(-1)
            identity_params = torch.zeros(
                (1, N_id),
                dtype=self.dtype,
                device=expression_params.device,
            ).expand(N, -1)
        else:
            N = identity_params.size(0)
            N_id = identity_params.size(-1)
            if expression_params is None:
                N_exp = self.expression_coeff.size(-1)
                expression_params = torch.zeros(
                    (1, N_exp),
                    dtype=self.dtype,
                    device=identity_params.device,
                ).expand(N, -1)
            else:
                N_exp = expression_params.size(-1)
        if global_rotation is None:
            global_rotation = self.zero_rotation.expand(N, -1)
        if neck_rotation is None:
            neck_rotation = self.zero_rotation.expand(N, -1)
        if jaw_rotation is None:
            jaw_rotation = self.zero_rotation.expand(N, -1)
        if eyes_rotation is None:
            eyes_rotation = self.zero_rotation.repeat(1, 2).expand(N, -1)

        rotations = torch.cat(
            (global_rotation, neck_rotation, jaw_rotation, eyes_rotation),
            dim=-1,
        )
        vertices, _ = linear_blend_skinning(
            shape_parameters=torch.cat(
                (identity_params, expression_params), dim=1
            ),
            rotations=rotations,
            shape_mean=self.v.expand(N, -1, -1),
            shape_coeff=torch.cat(
                (
                    self.identity_coeff[..., :N_id],
                    self.expression_coeff[..., :N_exp],
                ),
                dim=-1,
            ),
            rotation_coeff=self.rotation_coeff,
            J_regressor=self.joint_regressor,
            parents=self.parents,
            lbs_weights=self.lbs_weights,
            rot2mat=True,
            dtype=self.dtype,
        )

        (
            dynamic_lm_faces_idx,
            dynamic_lm_bary_coords,
        ) = self._find_dynamic_lm_idx_and_bcoords(rotations)
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
        self, rotations: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Selects the face contour depending on the reletive rotations of the head
        Args:
            rotations: shape (N, 15), rotations of global 3, eyes 6, jaw 3, \
                neck 3
        Returns:
            dynamic_lm_faces_idx:
                shape (N, 17), the contour face indexes
            dynamic_lm_bary_coords:
                shape (N, 17, 3), the contour face barycentric weights
        """
        N = rotations.size(0)
        aa_pose = torch.index_select(
            rotations.view(N, -1, 3), 1, self.neck_kin_chain
        )
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=self.dtype).view(
            N, -1, 3, 3
        )

        rel_rot_mat = torch.eye(
            3, device=rotations.device, dtype=self.dtype
        ).expand(N, -1, -1)
        for idx in range(len(self.neck_kin_chain)):
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


class FLAMETexture(nn.Module):
    """
    current FLAME texture are adapted from BFM Texture Model
    """

    texture_mean: Tensor
    """(1, 1, 786432 = 512*512*3)"""
    texture_basis: Tensor
    """(1, 786432 = 512*512*3, 200)"""
    verts_uv: Tensor
    """(1, 5118, 2)"""
    faces_uv: Tensor
    """(1, 9976, 3)"""

    def __init__(self, texture_path: str) -> None:
        super().__init__()
        # mean: (512, 512, 3), tex_dir: (512, 512, 3, 200)
        tex_space = np.load(url_to_local_path(texture_path))
        # BFM texture is in BGR order, flip to RGB
        texture_mean = np.flip(tex_space["mean"], axis=2).reshape(1, 1, -1)
        texture_basis = np.flip(tex_space["tex_dir"], axis=2).reshape(
            1, -1, 200
        )
        texture_mean = torch.from_numpy(texture_mean).float()
        texture_basis = torch.from_numpy(texture_basis).float()
        self.register_buffer("texture_mean", texture_mean)
        self.register_buffer("texture_basis", texture_basis)

        verts_uv = torch.from_numpy(tex_space["vt"]).float().reshape(1, -1, 2)
        faces_uv = (
            torch.from_numpy(tex_space["ft"].astype(np.int64))
            .long()
            .reshape(1, -1, 3)
        )
        self.register_buffer("verts_uv", verts_uv)
        self.register_buffer("faces_uv", faces_uv)

    @property
    def vt(self) -> Tensor:
        """(1, 5118, 2)"""
        return self.verts_uv

    @property
    def ft(self) -> Tensor:
        """(1, 9976, 3)"""
        return self.faces_uv

    def forward(self, input: Tensor) -> Tensor:
        """
        input: (N, texture_params<=200)
        return: (N, 3, 512, 512)
        """
        if input.ndim == 2:
            input = input.unsqueeze(1)
        if input.size(-1) > 200:
            raise ValueError("texture_params must <= 200")
        texture_basis = self.texture_basis[..., input.size(-1)]
        texture = self.texture_mean + (texture_basis * input).sum(-1)
        texture = texture.reshape(input.size(0), 512, 512, 3).permute(
            0, 3, 1, 2
        )
        return texture

    def get_texture_kwargs(
        self, input: Tensor, mode: str = "pytorch3d"
    ) -> Dict[str, Tensor]:
        """
        Args:
            input: (N, texture_params<=200)
        Returns:
            Dict[
                maps: (N, 3, 512, 512),
                faces_uvs: (N, 9976, 3),
                verts_uvs: (N, 5118, 2),
            ]
        """
        if mode != "pytorch3d":
            raise NotImplementedError("only support pytorch3d mode")

        batch_size = input.shape[0]
        texture = self.forward(input)
        texture = texture.permute(0, 2, 3, 1).div(255.0).clamp(0.0, 1.0)
        faces_uvs = self.faces_uv.expand(batch_size, -1, -1)
        verts_uvs = self.verts_uv.expand(batch_size, -1, -1)
        return {"maps": texture, "faces_uvs": faces_uvs, "verts_uvs": verts_uvs}
