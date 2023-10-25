from f2f.core import TORCH_AVAILABLE

from .deca import EMOCAONNX

__all__ = ["EMOCAONNX"]

if TORCH_AVAILABLE:
    from .deca.emoca import EMOCAEncoder  # noqa: F401

    __all__.extend(["EMOCAEncoder"])
