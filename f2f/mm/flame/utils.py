import pickle
from enum import Enum
from typing import Any, Dict

import numpy as np

from f2f.utils import url_to_local_path


class FLAME_MODELS(Enum):
    _2020 = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2020_generic-ffd4033d.pkl"
    _2023 = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2023-8fb1af0d.pkl"
    _2023_NO_JAW = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame2023_no_jaw-b291fcd1.pkl"


TEXTURE_COORD = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/flame_texture_coord-f8d5f559.pkl"


def load_flame(path: str) -> Dict[str, Any]:
    path = url_to_local_path(path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")  # noqa: S301
    except ModuleNotFoundError:
        raise
    except ImportError:
        setattr(np, "bool", bool)
        setattr(np, "int", int)
        setattr(np, "float", float)
        setattr(np, "complex", np.complex_)
        setattr(np, "object", object)
        setattr(np, "unicode", np.unicode_)
        setattr(np, "str", str)
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")  # noqa: S301
