from typing import Any, Optional, Sequence, Union, overload

import cv2
import numpy as np
from numpy import ndarray
from PIL import ExifTags, Image

ARCFACE_LM5 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def as_rgb_ndarray(input: Union[ndarray, Image.Image]) -> ndarray:
    """
    Args:
        input: (H, W, C), image in range [0, 255] or PIL.Image. C is 1, 3 or 4.
    Returns:
        (H, W, 3), RGB image in range [0, 255]
    """
    if isinstance(input, Image.Image):
        input = np.array(input.convert("RGB"), dtype=np.float32, copy=False)
    if input.ndim == 2:
        input = np.expand_dims(input, axis=-1)
    if input.shape[-1] == 1:
        input = np.repeat(input, 3, axis=-1)
    elif input.shape[-1] == 4:
        input = input[..., :3]
    return input.astype(np.float32, copy=False)


def square_resize(
    image: Union[ndarray, Image.Image],
    resolution: int,
    fill: int = 0,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    center_pad: bool = False,
) -> ndarray:
    """
    Resize method for models that require square input,
    keeping the aspect ratio by filling the background.

    Args:
        image: (H, W, 3), RGB image in range [0, 255] or PIL.Image
        resolution: target resolution
        fill: value to fill the background
        resample: PIL.Image resampling method
        center_pad: whether to center pad the image
            if True, pad all sides equally to make the image square.
            if False, pad only the right and bottom sides.
    Returns:
        (resolution, resolution, 3), resized image in range [0, 255]
    """
    image = as_rgb_ndarray(image)
    H, W = image.shape[:2]

    if H == W:
        if resolution == H:
            return image
        resized_image = Image.fromarray(image.astype(np.uint8)).resize(
            (resolution, resolution), resample=resample
        )
        return np.array(resized_image, dtype=np.float32)

    ratio = resolution / max(H, W)
    new_H, new_W = round(H * ratio), round(W * ratio)
    resized_image = Image.fromarray(image.astype(np.uint8)).resize(
        (new_W, new_H), resample=resample
    )
    if center_pad:
        # pad all sides to square resolution
        H_div, H_mod = divmod(resolution - new_H, 2)
        W_div, W_mod = divmod(resolution - new_W, 2)
        pad_width = ((H_div, H_div + H_mod), (W_div, W_div + W_mod), (0, 0))
    else:
        # pad right and bottom sides to square resolution
        pad_width = ((0, resolution - new_H), (0, resolution - new_W), (0, 0))
    image = np.pad(
        resized_image,
        pad_width=pad_width,
        constant_values=fill,
    )
    return image.astype(np.float32)


def change_keys(
    dictionary: dict, key_pair: dict, strip_str: bool = False
) -> dict:
    new_dictionary = {}
    for k, v in dictionary.items():
        new_key = key_pair.get(k, k)
        if strip_str and isinstance(v, str):
            v = v.strip()
        new_dictionary[new_key] = v
    return new_dictionary


def read_str_exif(image: Union[str, Image.Image]) -> dict:
    if isinstance(image, str):
        image = Image.open(image)
    exif = image.getexif()
    str_exif = change_keys(exif, ExifTags.TAGS)
    exif_offset = exif.get(ExifTags.IFD.Exif)
    if not isinstance(exif_offset, dict):
        exif_offset = exif.get_ifd(ExifTags.IFD.Exif)
    if exif_offset:
        str_exif[ExifTags.TAGS[ExifTags.IFD.Exif]] = change_keys(
            exif_offset, ExifTags.TAGS
        )
    gps_info = exif.get(ExifTags.IFD.GPSInfo)
    if not isinstance(gps_info, dict):
        gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if gps_info:
        str_exif[ExifTags.TAGS[ExifTags.IFD.GPSInfo]] = change_keys(
            gps_info, ExifTags.GPSTAGS
        )
    return str_exif


def exif_to_pil_exif(
    exif: Union[dict, Image.Exif], strip_str: bool = False
) -> Image.Exif:
    if isinstance(exif, Image.Exif):
        return exif

    str_tags = {v: k for k, v in ExifTags.TAGS.items()}
    str_gps_tags = {v: k for k, v in ExifTags.GPSTAGS.items()}
    exif = change_keys(exif, str_tags, strip_str=strip_str)
    exif_offset = exif.get(ExifTags.IFD.Exif)
    if exif_offset:
        if isinstance(exif_offset, dict):
            exif_offset = change_keys(
                exif_offset, str_tags, strip_str=strip_str
            )
        exif[ExifTags.IFD.Exif] = exif_offset
    gps_info = exif.get(ExifTags.IFD.GPSInfo)
    if gps_info:
        if isinstance(gps_info, dict):
            gps_info = change_keys(gps_info, str_gps_tags, strip_str=strip_str)
        exif[ExifTags.IFD.GPSInfo] = gps_info
    pil_exif = Image.Exif()
    pil_exif.update(exif)
    return pil_exif


def save_image_with_exif(
    path: str,
    image: Union[str, Image.Image],
    exif: Optional[Union[dict, Image.Exif]] = None,
    **params: Any
) -> None:
    if isinstance(image, str):
        image = Image.open(image)

    if exif is None:
        exif = read_str_exif(image)
    exif = exif_to_pil_exif(exif, strip_str=True)
    image.save(path, exif=exif, **params)


def rotate_image_upright(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)

    exif = read_str_exif(image)
    orientation = exif.get("Orientation")
    if orientation is None:
        return image

    # Flip
    if orientation in {2, 4}:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        orientation -= 1
    elif orientation in {5, 7}:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        orientation += 1

    # Rotate
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(-90, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    exif["Orientation"] = 1

    pil_exif = exif_to_pil_exif(exif, strip_str=True)
    pil_exif._loaded = True
    image._exif = pil_exif
    return image


def umeyama_transform(
    src: ndarray, dst: ndarray, estimate_scale: bool = True
) -> ndarray:
    """
    Copy from `scikit-image/skimage/transform/_geometric.py`.
    Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


@overload
def face_align_lm5(
    image: ndarray,
    source_lm5: ndarray,
    target_lm5: ndarray = ARCFACE_LM5,
    resolution: Union[Sequence[int], int] = (112, 112),
) -> ndarray:
    ...


@overload
def face_align_lm5(
    image: Image.Image,
    source_lm5: ndarray,
    target_lm5: ndarray = ARCFACE_LM5,
    resolution: Union[Sequence[int], int] = (112, 112),
) -> Image.Image:
    ...


def face_align_lm5(
    image: Union[ndarray, Image.Image],
    source_lm5: ndarray,
    target_lm5: ndarray = ARCFACE_LM5,
    resolution: Union[Sequence[int], int] = (112, 112),
) -> Union[ndarray, Image.Image]:
    M = umeyama_transform(source_lm5, target_lm5, estimate_scale=True)
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(image, ndarray):
        return cv2.warpAffine(image, M[:2], resolution, borderValue=0.0)

    inv_M = np.linalg.inv(M).ravel()
    return image.transform(
        resolution,
        Image.Transform.AFFINE,
        inv_M[:6],
        resample=Image.Resampling.BILINEAR,
    )
