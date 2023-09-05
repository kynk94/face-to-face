from f2f.core import TORCH_AVAILABLE

from .bisenet import FaceBiSeNetONNX

__all__ = ["FaceBiSeNetONNX"]

if TORCH_AVAILABLE:
    from .bisenet.model import FaceBiSeNet  # noqa: F401

    __all__.append("FaceBiSeNet")
