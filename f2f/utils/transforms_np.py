import numpy as np
from numpy import ndarray


def rotation_6d_to_matrix(d6: ndarray) -> ndarray:
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
    eps = 1e-12  # same as torch
    a1, a2 = d6[..., :3], d6[..., 3:]
    a1_norm: ndarray = np.linalg.norm(a1, ord=2, axis=-1, keepdims=True)
    b1 = a1 / a1_norm.clip(eps, None)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1
    b2_norm: ndarray = np.linalg.norm(b2, ord=2, axis=-1, keepdims=True)
    b2 = b2 / b2_norm.clip(eps, None)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: ndarray) -> ndarray:
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
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def batch_rodrigues(rot_vecs: ndarray, epsilon: float = 1e-8) -> ndarray:
    """
    Calculates the rotation matrices for a batch of rotation vectors.
    See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Args:
        rot_vecs: (N, 3), array of N axis-angle vectors
    Returns:
        rotation_matrix: (N, 3, 3), rotation matrices for the given axis-angle
    """

    batch_size = rot_vecs.shape[0]
    dtype = rot_vecs.dtype

    angle: ndarray = np.linalg.norm(rot_vecs, ord=2, axis=1, keepdims=True)
    angle = angle.clip(epsilon, None)
    rot_dir = rot_vecs / angle

    cos = np.expand_dims(np.cos(angle), 1)
    sin = np.expand_dims(np.sin(angle), 1)

    # Bx1 arrays
    rx, ry, rz = np.split(rot_dir, 3, axis=-1)

    zeros = np.zeros((batch_size, 1), dtype=dtype)
    # fmt: off
    K = np.concatenate([
        zeros,   -rz,    ry,
           rz, zeros,   -rx,
          -ry,    rx, zeros
    ], axis=1).reshape(batch_size, 3, 3)
    # fmt: on

    ident = np.expand_dims(np.eye(3, dtype=dtype), 0)
    rotation_matrix = ident + sin * K + (1 - cos) * (K @ K)
    return rotation_matrix
