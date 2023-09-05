from f2f.core import TORCH_AVAILABLE

__all__ = []

if TORCH_AVAILABLE:
    from .arcface import ARCFACE_MODELS, Arcface  # noqa: F401

    __all__ = ["ARCFACE_MODELS", "Arcface"]
