import os
from pathlib import Path
from types import MethodType
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet, resnet50

from f2f.utils.onnx_ops import OnnxExport

CHECKPOINT_PATH = (
    Path(__file__).parents[1] / "assets" / "deca" / "deca_model.tar"
)


class _FLAME_FEATURE_INDEX(NamedTuple):
    identity: Tuple[int, int] = (0, 100)
    texture: Tuple[int, int] = (100, 150)
    expression: Tuple[int, int] = (150, 200)
    global_rotation: Tuple[int, int] = (200, 203)
    jaw_rotation: Tuple[int, int] = (203, 206)
    scale: Tuple[int, int] = (206, 207)
    xy_translation: Tuple[int, int] = (207, 209)
    light: Tuple[int, int] = (209, 236)


@OnnxExport()
def onnx_export(use_detail: bool = False) -> None:
    # export flame encoder only
    model = DECAEncoder(
        use_detail=use_detail, checkpoint=str(CHECKPOINT_PATH)
    ).eval()
    input = torch.randn(1, 3, model.resolution, model.resolution)
    print(f"Exporting {model._get_name()} ONNX...")
    print(f"Use Input: {input.size()}")
    output_names = list(model.feature_index._fields)
    if use_detail:
        file_name = "deca_E_use_detail.onnx"
        output_names += ["detail"]
    else:
        file_name = "deca_E.onnx"
    torch.onnx.export(
        model,
        input,
        str(CHECKPOINT_PATH.parent / file_name),
        opset_version=13,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            **{k: {0: "batch_size"} for k in output_names},
        },
    )


class DECAEncoder(nn.Module):
    resolution: int = 224
    E_detail: Optional["ResNetEncoder"] = None

    def __init__(
        self,
        use_detail: bool = False,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.use_detail = use_detail
        self.feature_index = _FLAME_FEATURE_INDEX()
        self.E_flame = ResNetEncoder(max(self.feature_index)[-1])
        if self.use_detail:
            self.E_detail = ResNetEncoder(128)
        if checkpoint is not None:
            self.load(checkpoint)
            self.eval()

    def load(self, checkpoint: str) -> None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{checkpoint} does not exist.")
        state_dict: Dict[str, Any] = torch.load(checkpoint, map_location="cpu")
        state_dict = state_dict.get("state_dict", state_dict)
        self.E_flame.load_state_dict(state_dict["E_flame"])
        if self.E_detail is not None:
            self.E_detail.load_state_dict(state_dict["E_detail"])
        print(f"{self._get_name()} Loaded weights from {checkpoint}")

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        outputs = self.encode_flame(input)
        if self.E_detail is not None:
            detail_coeffs = self.E_detail(input)
            outputs["detail"] = detail_coeffs
        return outputs

    def encode_flame(self, input: Tensor) -> Dict[str, Tensor]:
        coeffs = self.E_flame(input)
        return self.decompose_flame_coeffs(coeffs)

    def decompose_flame_coeffs(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Decompose the input coefficients into the flame parameters.
        """
        parameters = {}
        start = 0
        for name in self.feature_index._fields:
            start, end = getattr(self.feature_index, name)
            coeff = input[:, start:end]
            if name == "light":
                coeff = coeff.view(-1, 9, 3)
            parameters[name] = coeff
            start = end
        return parameters


def resnet50_backbone() -> ResNet:
    model = resnet50()
    del model.fc

    def _forward_impl(self: ResNet, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    model._forward_impl = MethodType(_forward_impl, model)
    return model


class ResNetEncoder(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.encoder = resnet50_backbone().eval()
        self.resnet_feature_size = 2048
        self.layers = nn.Sequential(
            nn.Linear(self.resnet_feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_features),
        )

    def features(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def forward(self, input: Tensor) -> Tensor:
        features = self.features(input)
        return self.layers(features)


class Reshape(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        return input.view(-1, *self.shape).contiguous()


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, out_channels: int = 1) -> None:
        super().__init__()
        self.initial_resolution = 8
        sample_mode = "bilinear"
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.initial_resolution**2),  # 8192
            Reshape(128, self.initial_resolution, self.initial_resolution),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
