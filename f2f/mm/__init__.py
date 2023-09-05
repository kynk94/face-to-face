from f2f.core import TORCH_AVAILABLE

__all__ = []

if TORCH_AVAILABLE:
    from .flame import FLAME, FLAMETexture  # noqa: F401

    __all__.extend(["FLAME", "FLAMETexture"])
