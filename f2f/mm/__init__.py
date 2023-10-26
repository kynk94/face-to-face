from f2f.core import TORCH_AVAILABLE

from .flame import FLAME_MODELS, FLAMEONNX
from .img2mm import EMICAONNX, EMOCAONNX, MICAONNX

__all__ = ["FLAMEONNX", "FLAME_MODELS", "EMICAONNX", "EMOCAONNX", "MICAONNX"]

if TORCH_AVAILABLE:
    from .flame.model import FLAME, FLAMETexture  # noqa: F401
    from .img2mm import MICA, EMOCAEncoder  # noqa: F401

    __all__.extend(["EMOCAEncoder", "FLAME", "FLAMETexture", "MICA"])
