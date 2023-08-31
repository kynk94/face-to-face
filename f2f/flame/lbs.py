# -*- coding: utf-8 -*-

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

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from f2f.utils.onnx_ops import onnx_atan2


def rot_mat_to_euler(rot_mats: Tensor) -> Tensor:
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(
        rot_mats[:, 0, 0] * rot_mats[:, 0, 0]
        + rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
    )
    return onnx_atan2(-rot_mats[:, 2, 0], sy)


def vertices2landmarks(
    vertices: Tensor,
    faces: Tensor,
    lm_faces_idx: Tensor,
    lm_bary_coords: Tensor,
) -> Tensor:
    """
    Calculates landmarks by barycentric interpolation

    Args:
        vertices: (N, V, 3), input vertices
        faces: (F, 3), faces of the mesh
        lm_faces_idx: (L,), The indices of the faces used to calculate the \
            landmarks.
        lm_bary_coords: (L, 3), barycentric coordinates used to interpolate \
            the landmarks
    Returns:
        landmarks: (B, L, 3), the landmarks coordinates for each mesh in batch.
    """
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces.long(), 0, lm_faces_idx.view(-1)).view(
        batch_size, -1, 3
    )

    lmk_faces += (
        torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1)
        * num_verts
    )

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lm_bary_coords])
    return landmarks


def linear_blend_skinning(
    shape_parameters: Tensor,
    rotations: Tensor,
    shape_mean: Tensor,
    shape_coeff: Tensor,
    rotation_coeff: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    rot2mat: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """
    Performs Linear Blend Skinning with the given shape and rotation parameters

    Args:
        shape_parameters: (N, B), shape parameters
        rotations: (N, (J+1)*3), rotation parameters in radian axis-angle format
        shape_mean: (N, V, 3), template mesh that will be deformed
        shape_coeff: (1, V, NB), PCA shape displacements
        rotation_coeff: (P, V, 3), pose PCA coefficients
        J_regressor: (V, J), regressor that maps from vertices to joints
        parents: (J), indices of the parents for each joint
        lbs_weights: (N, V, J), weights of the linear blend skinning
        rot2mat: If True, the rotations are converted to a rotation matrix
        dtype: data type of the output
    Returns:
        vertices: (N, V, 3), deformed mesh vertices
        joints: (N, J, 3), joint locations
    """
    N = max(shape_parameters.size(0), rotations.size(0))
    device = shape_parameters.device

    # Add shape contribution
    v_shaped = shape_mean + blend_shapes(shape_parameters, shape_coeff)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # J: 5 joints = (global, neck, jaw, left_eye, right_eye)
    # rot_mats: (N, J, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device)
    if rot2mat:
        rotation_matrix = batch_rodrigues(
            rotations.view(-1, 3), dtype=dtype
        ).view(N, -1, 3, 3)
    else:
        rotation_matrix = rotations.view(N, -1, 3, 3)
    rotation_feature = (rotation_matrix[:, 1:, :, :] - ident).view([N, -1])
    # (N x P) x (P, V * 3) -> N x V x 3
    rotation_offsets = torch.matmul(rotation_feature, rotation_coeff).view(
        N, -1, 3
    )

    # apply rotation offsets to the blend shape
    v_posed = rotation_offsets + v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rotation_matrix, J, parents)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.expand(N, -1, -1)
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.view(N, J_regressor.size(0), 16)).view(N, -1, 4, 4)

    homogen_coord = torch.ones(
        (N, v_posed.size(1), 1), dtype=dtype, device=device
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    """
    Calculates the 3D joint locations from the vertices.

    Args:
        J_regressor: (J, V), regressor array used to calculate the \
            joints from the position of the vertices
        vertices: (B, V, 3), the mesh vertices
    Returns:
        (B, J, 3), location of the joints
    """

    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(betas: Tensor, shape_displacement: Tensor) -> Tensor:
    """
    Calculates the per vertex displacement due to the blend shapes

    Args:
        betas: (B, num_betas), shape coefficients
        shape_displacement: (V, 3, num_betas)
    Returns:
        (B, V, 3), The per-vertex displacement due to shape deformation.
    """
    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_displacement[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,mkl->bmk", [betas, shape_displacement])
    return blend_shape


def batch_rodrigues(
    rot_vecs: Tensor, epsilon: float = 1e-8, dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Calculates the rotation matrices for a batch of rotation vectors.
    See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Args:
        rot_vecs: (N, 3), array of N axis-angle vectors
    Returns:
        rotation_matrix: (N, 3, 3), rotation matrices for the given axis-angle
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    # fmt: off
    K = torch.cat([
        zeros,   -rz,    ry,
           rz, zeros,   -rx,
          -ry,    rx, zeros
    ], dim=1).view((batch_size, 3, 3))
    # fmt: on

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rotation_matrix = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rotation_matrix


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    """Creates a batch of transformation matrices
    Args:
        R: (B, 3, 3) array of a batch of rotation matrices
        t: (B, 3, 1) array of a batch of translation vectors
    Returns:
        T: (B, 4, 4) Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat(
        [F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2
    )


def batch_rigid_transform(
    rot_mats: Tensor, joints: Tensor, parents: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Applies a batch of rigid transformations to the joints.

    Args:
        rot_mats: (B, N, 3, 3), rotation matrices
        joints : (B, N, 3), locations of joints
        parents : (B, N), The kinematic tree of each object
    Returns:
        posed_joints: (B, N, 3), The locations of the joints after applying \
            the pose rotations.
        rel_transforms : (B, N, 4, 4), The relative (with respect to the root \
            joint) rigid transformations for all the joints.
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(
            transform_chain[parents[i]], transforms_mat[:, i]
        )
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms
