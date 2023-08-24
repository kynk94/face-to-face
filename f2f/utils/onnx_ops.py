from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.autograd.grad_mode import _DecoratorContextManager

IN_ONNX_EXPORT = False


class OnnxExport(_DecoratorContextManager):
    def __init__(self) -> None:
        self.prev = False

    def __enter__(self) -> None:
        global IN_ONNX_EXPORT
        IN_ONNX_EXPORT = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global IN_ONNX_EXPORT
        IN_ONNX_EXPORT = self.prev


def onnx_atan2(
    input: Tensor, other: Tensor, *, out: Optional[Tensor] = None
) -> Tensor:
    """
    As of July 2023, ONNX does not support torch.atan2, so need to implement it.
    https://gist.github.com/nikola-j/b5bb6b141b8d9920318677e1bba70466?permalink_comment_id=4550495#gistcomment-4550495
    """  # noqa: E501
    if not IN_ONNX_EXPORT:
        return torch.atan2(input=input, other=other, out=out)

    # Create a pi tensor with the same device and data type as y
    pi = torch.tensor(np.pi, device=input.device, dtype=input.dtype)
    half_pi = pi / 2
    eps = 1e-6

    # Compute the arctangent of input/other
    ans = torch.atan(input / (other + eps), out=out)

    # Create boolean tensors representing positive, negative, and zero values
    # of input and other
    input_positive = input > 0
    input_negative = input < 0
    other_negative = other < 0
    other_zero = other == 0

    # Adjust ans based on the positive, negative, and zero values of input
    # and other
    ans += torch.where(
        input_positive & other_negative, pi, torch.zeros_like(ans)
    )  # Quadrants I and II
    ans -= torch.where(
        input_negative & other_negative, pi, torch.zeros_like(ans)
    )  # Quadrants III and IV
    ans = torch.where(
        input_positive & other_zero, half_pi, ans
    )  # Positive input-axis
    ans = torch.where(
        input_negative & other_zero, -half_pi, ans
    )  # Negative input-axis

    return ans
