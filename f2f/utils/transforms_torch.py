import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def batch_rodrigues(rot_vecs: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    Calculates the rotation matrices for a batch of rotation vectors.
    See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Args:
        rot_vecs: (N, 3), array of N axis-angle vectors
    Returns:
        rotation_matrix: (N, 3, 3), rotation matrices for the given axis-angle
    """

    batch_size = rot_vecs.size(0)
    dtype = rot_vecs.dtype
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
