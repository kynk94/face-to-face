import os
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import onnxruntime
from numpy import ndarray

from f2f.utils import url_to_local_path


class BaseONNX:
    __onnx_path: Optional[str] = None
    __session: Optional[onnxruntime.InferenceSession] = None
    __input_names: Tuple[str, ...]
    __output_names: Tuple[str, ...]
    __device: str = "cpu"
    __dtype: npt.DTypeLike = np.float32

    def __init__(
        self,
        onnx_path: str,
        device: Union[str, int] = "cpu",
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        """
        Initialize ONNX model.

        Args:
            onnx_path: Path to ONNX model.
            device: Device to run inference on. (default: "cpu")

        Initialization is done in two steps:
            1: `set_session` is called to initialize inference session.
                (cpu session is initialized with given `onnx_path`.)
                Can add custom logic by overriding `set_session`.
            2: `to` is called to set device.
                if `device` is not "cpu", `__set_providers` is called.
        """
        self.__dtype = dtype
        self.onnx_path = onnx_path
        self.to(device)

    @property
    def onnx_path(self) -> str:
        if self.__onnx_path is None:
            raise RuntimeError("onnx_path is not initialized")
        return self.__onnx_path

    @onnx_path.setter
    def onnx_path(self, onnx_path: str) -> None:
        onnx_path = onnx_path.strip()
        if not onnx_path:
            raise ValueError("onnx_path cannot be empty")
        onnx_path = url_to_local_path(onnx_path)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"onnx_path {onnx_path} does not exist")
        if onnx_path == self.__onnx_path:
            return
        self.__onnx_path = onnx_path
        self.set_session()

    def __set_providers(self, device: str) -> None:
        if device == "cpu":
            self.session.set_providers(providers=["CPUExecutionProvider"])
            return

        index = int(device.split(":")[-1])
        providers = [
            (
                "CUDAExecutionProvider",
                {"device_id": index, "cudnn_conv_use_max_workspace": "1"},
            ),
            "CPUExecutionProvider",
        ]
        self.session.set_providers(providers=providers)

    def __set_session(self) -> None:
        self.__session = onnxruntime.InferenceSession(
            self.__onnx_path, providers=["CPUExecutionProvider"]
        )
        self.__session._sess
        self.__input_names = tuple(i.name for i in self.__session.get_inputs())
        self.__output_names = tuple(
            o.name for o in self.__session.get_outputs()
        )
        if self.device == "cpu":
            return
        self.__set_providers(self.device)

    def set_session(self) -> None:
        """
        Set session with current `onnx_path`.
        If `onnx_path` is changed, this method is called automatically.
        Can add custom logic by overriding this method.
        """
        self.__set_session()

    @property
    def device(self) -> str:
        return self.__device

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.__dtype

    @property
    def session(self) -> onnxruntime.InferenceSession:
        if self.__session is None:
            raise RuntimeError("session is not initialized")
        return self.__session

    @property
    def input_names(self) -> Tuple[str, ...]:
        return self.__input_names

    @property
    def output_names(self) -> Tuple[str, ...]:
        return self.__output_names

    def session_run(self, *args: Any) -> List[npt.NDArray]:
        """
        Run inference session with given inputs.
        """
        if len(args) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs, got {len(args)}"
            )
        session_inputs = {
            input_name: arg.astype(self.dtype, copy=False)
            if isinstance(arg, ndarray)
            else arg
            for input_name, arg in zip(self.input_names, args)
        }
        return self.session.run(self.output_names, session_inputs)

    def to(self, device: Union[str, int]) -> "BaseONNX":
        if device in {"gpu", "cuda"}:
            device = "cuda:0"
        elif isinstance(device, int):
            device = f"cuda:{device}"
        elif (
            not isinstance(device, str)
            and device.__class__.__name__ == "device"
        ):
            device = str(device)
        if device == self.__device:
            return self
        if re.match(r"^cuda:\d+$", device) is None and device != "cpu":
            raise ValueError(f"device {device} is not supported")

        self.__device = device
        self.__set_providers(self.device)
        return self

    def cpu(self) -> "BaseONNX":
        return self.to("cpu")

    def cuda(self) -> "BaseONNX":
        return self.to("cuda")

    def float(self) -> "BaseONNX":
        self.__dtype = np.float32
        return self

    def half(self) -> "BaseONNX":
        self.__dtype = np.float16
        return self
