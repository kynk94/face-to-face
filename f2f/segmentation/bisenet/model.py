"""
BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
https://arxiv.org/abs/1808.00897

Reference:
    https://github.com/zllrunning/face-parsing.PyTorch
"""
import os
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18

from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)
from f2f.utils.onnx_ops import OnnxExport

BISENET_CELEBA = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/bisenet_celeba-468e13ca.pth"


@OnnxExport()
def onnx_export() -> None:
    model = FaceBiSeNet()
    input = torch.randn(1, 3, model.resolution, model.resolution)
    print(f"Exporting {model._get_name()} ONNX...")
    print(f"Use Input: {input.size()}")
    file_name = os.path.splitext(os.path.basename(BISENET_CELEBA))[0] + ".onnx"
    onnx_path = os.path.join(get_onnx_cache_dir(), file_name)
    torch.onnx.export(
        model,
        input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["segmentation"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "segmentation": {0: "batch_size"},
        },
    )
    exported_path = rename_file_with_hash(onnx_path)
    print(f"Exported to {exported_path}")


CELEBA_PARTS = {
    "background": 0,
    "skin": 1,
    "l_brow": 2,
    "r_brow": 3,
    "l_eye": 4,
    "r_eye": 5,
    "eye_glass": 6,
    "l_ear": 7,
    "r_ear": 8,
    "ear_ring": 9,
    "nose": 10,
    "mouth": 11,
    "u_lip": 12,
    "l_lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}


class FaceBiSeNet(nn.Module):
    resolution: int = 512
    """trained on 448x448 random crops from 512x512 resized CelebAMask-HQ"""
    mean: Tensor
    std: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.model = BiSeNet(n_classes=19, checkpoint=BISENET_CELEBA).eval()
        self.model.requires_grad_(False)

        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FaceBiSeNet":
        return super().train(False)

    @property
    def parts(self) -> Dict[str, int]:
        copy_parts = CELEBA_PARTS.copy()
        return copy_parts

    def normalize_input(self, input: Tensor) -> Tensor:
        """
        input: (N, 3, H, W) RGB image in range [-1, 1]
        """
        return (input * 0.5 + 0.5).sub(self.mean).div(self.std)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: (N, 3, 512, 512) RGB image in range [-1, 1]
        Returns:
            segmentation: (N, 1, 512, 512)
        """
        H, W = input.shape[-2:]
        normalized_input = self.normalize_input(input)
        feat_out = self.model.calc_largest_feature(normalized_input)
        resized_feat = F.interpolate(
            feat_out, (H, W), mode="bilinear", align_corners=True
        )
        segmentation = torch.argmax(resized_feat, dim=1, keepdim=True)
        return segmentation.byte()

    def calc_features(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            input: (N, 3, 512, 512) RGB image in range [-1, 1]
        Returns:
            feature: (N, 19, 64, 64)
            feature16: (N, 19, 64, 64)
            feature32: (N, 19, 32, 32)
        """
        normalized_input = self.normalize_input(input)
        return self.model.calc_features(normalized_input)

    def segmentation_to_mask(
        self, segmentation: Tensor, parts: Sequence[Union[str, int]]
    ) -> Tensor:
        """
        Args:
            segmentation: (N, 1, H, W) tensor
            parts: list of part names or indices to include in the mask
        Returns:
            mask: (N, 1, H, W) 0, 1 binary mask
        """
        mask = torch.zeros_like(segmentation, dtype=torch.float32)
        for part in parts:
            if isinstance(part, str):
                part = self.parts[part]
            mask[segmentation == part] = 1
        return mask

    def feature_to_mask(
        self, feature: Tensor, parts: Sequence[Union[str, int]]
    ) -> Tensor:
        """
        Args:
            feature: (N, 19, H, W) tensor
            parts: list of part names or indices to include in the mask
        Returns:
            mask: (N, 1, H, W) 0, 1 binary mask
        """
        segmentation = torch.argmax(feature, dim=1, keepdim=True)
        return self.segmentation_to_mask(segmentation, parts)


class BiSeNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 19,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()
        if checkpoint is not None:
            self.load(url_to_local_path(checkpoint))

    def calc_largest_feature(self, x: Tensor) -> Tensor:
        feat_res8, feat_cp8, feat_cp16 = self.cp(
            x
        )  # here return res3b1 feature
        feat_sp = (
            feat_res8  # use res3b1 feature to replace spatial path feature
        )
        feat_fuse = self.ffm(feat_sp, feat_cp8)
        return self.conv_out(feat_fuse)

    def calc_features(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feat_res8, feat_cp8, feat_cp16 = self.cp(
            x
        )  # here return res3b1 feature
        feat_sp = (
            feat_res8  # use res3b1 feature to replace spatial path feature
        )
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)
        return feat_out, feat_out16, feat_out32

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        H, W = x.shape[2:]
        feat_out, feat_out16, feat_out32 = self.calc_features(x)

        feat_out = F.interpolate(
            feat_out, (H, W), mode="bilinear", align_corners=True
        )
        feat_out16 = F.interpolate(
            feat_out16, (H, W), mode="bilinear", align_corners=True
        )
        feat_out32 = F.interpolate(
            feat_out32, (H, W), mode="bilinear", align_corners=True
        )
        return feat_out, feat_out16, feat_out32

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def load(self, checkpoint: str) -> None:
        """
        Load pretrained weights
        """
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{checkpoint} not found")
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)
        print(f"{self._get_name()} Loaded weights from {checkpoint}")


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, n_classes: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(
            mid_chan, n_classes, kernel_size=1, bias=False
        )
        self.init_weight()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(
            out_chan, out_chan, kernel_size=1, bias=False
        )
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x: Tensor) -> Tensor:
        feat: Tensor = self.conv(x)
        atten = feat.mean((2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet18()
        del self.resnet.avgpool
        del self.resnet.fc
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def resnet_forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(x)  # 1/8
        feat16 = self.resnet.layer3(feat8)  # 1/16
        feat32 = self.resnet.layer4(feat16)  # 1/32
        return feat8, feat16, feat32

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feat8, feat16, feat32 = self.resnet_forward(x)
        H8, W8 = feat8.shape[2:]
        H16, W16 = feat16.shape[2:]
        H32, W32 = feat32.shape[2:]

        avg = feat32.mean((2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(
            out_chan,
            out_chan // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_chan // 4,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp: Tensor, fcp: Tensor) -> Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat: Tensor = self.convblk(fcat)
        atten = feat.mean((2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)
