from f2f import detection, landmark, segmentation, utils
from f2f.core import TORCH_AVAILABLE

__all__ = ["detection", "landmark", "segmentation", "utils"]

if TORCH_AVAILABLE:
    from f2f import mm, recognition  # noqa: F401

    __all__.extend(["mm", "recognition"])
