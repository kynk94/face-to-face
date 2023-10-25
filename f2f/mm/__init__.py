from f2f.core import TORCH_AVAILABLE

from .img2mm import EMOCAONNX

__all__ = ["EMOCAONNX"]

if TORCH_AVAILABLE:
    from .flame import FLAME, FLAME_MODELS, FLAMETexture  # noqa: F401

    __all__.extend(["FLAME", "FLAME_MODELS", "FLAMETexture"])
