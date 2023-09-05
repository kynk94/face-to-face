from f2f.core import TORCH_AVAILABLE

from .synthetic import FDSyntheticLM2dONNX

__all__ = ["FDSyntheticLM2dONNX"]

if TORCH_AVAILABLE:
    from .synthetic.model import (  # noqa: F401
        FDSyntheticLM2dInference,
        SyntheticLM2d,
        SyntheticLM2dInference,
    )

    __all__.extend(
        ["FDSyntheticLM2dInference", "SyntheticLM2d", "SyntheticLM2dInference"]
    )
