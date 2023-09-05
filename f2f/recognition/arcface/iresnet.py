"""
Arcface taken from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
from torch import Tensor

from f2f.utils import url_to_local_path


def iresnet_from_weights(weights: str, fp16: bool = True) -> "IResNet":
    """
    Automatically choose the correct IResNet model based on the weights URL.
    """
    weights = url_to_local_path(weights)
    state_dict = torch.load(weights, map_location="cpu")
    layers = layers_from_state_dict(state_dict)
    model = IResNet(IBasicBlock, layers, fp16=fp16)
    model.load_state_dict(state_dict)
    return model


def layers_from_state_dict(
    state_dict: Dict[str, Tensor]
) -> Tuple[int, int, int, int]:
    layers = [0, 0, 0, 0]
    for key in state_dict:
        if not key.startswith("layer"):
            continue
        layer_name, n_blocks = key.split(".")[:2]
        layer_idx = int(layer_name[-1]) - 1
        layers[layer_idx] = max(layers[layer_idx], int(n_blocks) + 1)
    return cast(Tuple[int, int, int, int], tuple(layers))


# compatible with torchvision.models.resnet._resnet
def _iresnet(
    block: Type["IBasicBlock"],
    layers: Sequence[int],
    weights: Optional[str] = None,
    **kwargs: Any
) -> "IResNet":
    model = IResNet(block, layers, **kwargs)
    if weights is not None:
        pretrained_path = url_to_local_path(weights)
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    return model


def iresnet18(weights: Optional[str] = None, **kwargs: Any) -> "IResNet":
    return _iresnet(IBasicBlock, [2, 2, 2, 2], weights, **kwargs)


def iresnet34(weights: Optional[str] = None, **kwargs: Any) -> "IResNet":
    return _iresnet(IBasicBlock, [3, 4, 6, 3], weights, **kwargs)


def iresnet50(weights: Optional[str] = None, **kwargs: Any) -> "IResNet":
    return _iresnet(IBasicBlock, [3, 4, 14, 3], weights, **kwargs)


def iresnet100(weights: Optional[str] = None, **kwargs: Any) -> "IResNet":
    return _iresnet(IBasicBlock, [3, 13, 30, 3], weights, **kwargs)


def iresnet200(weights: Optional[str] = None, **kwargs: Any) -> "IResNet":
    return _iresnet(IBasicBlock, [6, 26, 60, 6], weights, **kwargs)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        self.bn1 = nn.BatchNorm2d(
            inplanes,
            eps=1e-05,
        )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self,
        block: Type[IBasicBlock],
        layers: Sequence[int],
        dropout: int = 0,
        num_features: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.block = block
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[IBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Module:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x
