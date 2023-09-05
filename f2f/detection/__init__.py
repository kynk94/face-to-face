from f2f.core import TORCH_AVAILABLE

from .s3fd import S3FDONNX
from .scrfd import SCRFDONNX

__all__ = ["S3FDONNX", "SCRFDONNX"]

if TORCH_AVAILABLE:
    from .s3fd.model import S3FD  # noqa: F401

    __all__.append("S3FD")
