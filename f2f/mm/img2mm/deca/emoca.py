import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from f2f.mm.img2mm.deca.deca import DECAEncoder, ResNetEncoder
from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)
from f2f.utils.onnx_ops import OnnxExport

E_FLAME = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/EMOCA_v2_lr_mse_20_E_flame-2aca4be6.ckpt"
E_EXPRESSION = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/EMOCA_v2_lr_mse_20_E_expression-d5fea8e4.ckpt"
E_DETAIL = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/EMOCA_v2_lr_mse_20_E_detail-91d9a887.ckpt"


@OnnxExport()
def onnx_export(use_detail: bool = False) -> None:
    model = EMOCAEncoder(use_detail=use_detail, checkpoint=E_FLAME).eval()
    input = torch.randn(1, 3, model.resolution, model.resolution)
    print(f"Exporting {model._get_name()} ONNX...")
    print(f"Use Input: {input.size()}")
    output_names = list(model.feature_index._fields)
    if use_detail:
        file_name = "EMOCA_v2_lr_mse_20_E_use_detail.onnx"
        output_names += ["detail"]
    else:
        file_name = "EMOCA_v2_lr_mse_20_E.onnx"
    onnx_path = os.path.join(get_onnx_cache_dir(), file_name)
    torch.onnx.export(
        model,
        input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            **{k: {0: "batch_size"} for k in output_names},
        },
    )
    exported_path = rename_file_with_hash(onnx_path)
    print(f"Exported to {exported_path}")


class EMOCAEncoder(DECAEncoder):
    def __init__(
        self,
        use_detail: bool = False,
        checkpoint: Optional[str] = E_FLAME,
    ) -> None:
        super().__init__(
            use_detail=use_detail,
            checkpoint=None,
        )
        expression_start, expression_end = self.feature_index.expression
        n_expression = expression_end - expression_start
        self.E_expression = ResNetEncoder(n_expression)
        if checkpoint is not None:
            self.load(checkpoint)
            self.eval()

    def load(self, checkpoint: str) -> None:
        checkpoint = url_to_local_path(checkpoint)
        state_dict: Dict[str, Any] = torch.load(checkpoint, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        refine_state_dict: Dict[str, Any] = defaultdict(dict)

        for key, val in state_dict.items():
            key = key.replace("deca.", "")
            module_key = key.split(".")[0]
            if module_key not in {
                "E_flame",
                "E_detail",
                "E_expression",
            }:
                continue
            refine_state_dict[module_key][
                key.replace(f"{module_key}.", "")
            ] = val
        if "E_flame" not in refine_state_dict:
            refine_state_dict["E_flame"] = torch.load(
                url_to_local_path(E_FLAME), map_location="cpu"
            )
        if "E_expression" not in refine_state_dict:
            refine_state_dict["E_expression"] = torch.load(
                url_to_local_path(E_EXPRESSION), map_location="cpu"
            )
        self.E_flame.load_state_dict(refine_state_dict["E_flame"])
        self.E_expression.load_state_dict(refine_state_dict["E_expression"])
        if self.E_detail is not None:
            if "E_detail" not in refine_state_dict:
                refine_state_dict["E_detail"] = torch.load(
                    url_to_local_path(E_DETAIL), map_location="cpu"
                )
            self.E_detail.load_state_dict(refine_state_dict["E_detail"])
        print(f"{self._get_name()} Loaded weights from {checkpoint}")

    def encode_flame(self, input: Tensor) -> Dict[str, Tensor]:
        coeffs = super().encode_flame(input)
        exp_coeffs = self.E_expression(input)
        coeffs["expression"] = exp_coeffs
        return coeffs
