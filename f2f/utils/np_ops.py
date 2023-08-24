import numpy as np
from numpy import ndarray


def softmax(input: ndarray, dim: int = 1) -> ndarray:
    _max = np.max(input, axis=dim, keepdims=True)
    e_input = np.exp(input - _max)
    sum = np.sum(e_input, axis=dim, keepdims=True)
    output = e_input / sum
    return output
