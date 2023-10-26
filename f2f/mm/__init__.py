from f2f.core import TORCH_AVAILABLE

from .img2mm import EMICAONNX, EMOCAONNX, MICAONNX

__all__ = ["EMICAONNX", "EMOCAONNX", "MICAONNX"]

if TORCH_AVAILABLE:
    from .flame import FLAME, FLAME_MODELS, FLAMETexture  # noqa: F401
    from .img2mm import MICA, EMOCAEncoder  # noqa: F401

    __all__.extend(
        ["EMOCAEncoder", "FLAME", "FLAME_MODELS", "FLAMETexture", "MICA"]
    )
