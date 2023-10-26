import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from f2f.recognition.arcface.iresnet import iresnet100
from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)
from f2f.utils.onnx_ops import OnnxExport

CHECKPOINT_PATH = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/mica-41b3e8ce.tar"


@OnnxExport()
def onnx_export() -> None:
    model = MICA(checkpoint=CHECKPOINT_PATH).eval()
    input = torch.randn(1, 3, model.resolution, model.resolution)
    print(f"Exporting {model._get_name()} ONNX...")
    print(f"Use Input: {input.size()}")
    output_names = ["identity", "arcface"]
    file_name = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0] + ".onnx"
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


class MICA(nn.Module):
    resolution = 112

    def __init__(self, checkpoint: Optional[str] = CHECKPOINT_PATH) -> None:
        super().__init__()
        self.arcface = iresnet100(fp16=True)
        self.mapping = MappingNetwork(512, 300, 300, 3)
        if checkpoint is not None:
            self.load(checkpoint)
            self.eval()

    def train(self, mode: bool = True) -> "MICA":
        return super().train(False)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input: shape (B, 3, 112, 112), range [-1, 1]
        Returns:
            identity_coeff: shape (B, 300) FLAME identity coefficients.
            arcface: shape (B, 512) representing the embedding.
        """
        arcface = F.normalize(self.arcface(input))
        identity_coeff = self.mapping(arcface)
        return identity_coeff, arcface

    def load(self, checkpoint: str) -> None:
        checkpoint = url_to_local_path(checkpoint)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.arcface.load_state_dict(state_dict["arcface"])
        mapping_statedict = {}
        for key, val in state_dict["flameModel"].items():
            if not key.startswith("regressor."):
                continue
            key = key.replace("regressor.", "")
            mapping_statedict[key] = val
        self.mapping.load_state_dict(mapping_statedict)
        print(f"{self._get_name()} Loaded weights from {checkpoint}")


class MappingNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 300,
        output_dim: int = 300,
        n_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.network = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for layer in self.network:
            output = F.leaky_relu(layer(output), negative_slope=0.2)
        output = self.output(output)
        return output
