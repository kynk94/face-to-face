from typing import Any, Optional, Union

import numpy as np
from numpy import ndarray
from PIL import ExifTags, Image


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
