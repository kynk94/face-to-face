from f2f.core import TORCH_AVAILABLE

from .deca import EMICAONNX, EMOCAONNX, MICAONNX

__all__ = ["EMICAONNX", "EMOCAONNX", "MICAONNX"]

if TORCH_AVAILABLE:
    from .deca.emoca import EMOCAEncoder  # noqa: F401
    from .deca.mica import MICA  # noqa: F401

    __all__.extend(["EMOCAEncoder", "MICA"])
