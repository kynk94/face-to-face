from typing import Union

import numpy as np
from numpy import ndarray
from PIL import Image


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
