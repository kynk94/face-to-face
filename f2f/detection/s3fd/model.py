"""
S3FD from face_alignment https://github.com/1adrianb/face-alignment
face-alignment is licensed under the BSD 3-Clause License.
"""
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

from f2f.detection.utils import cal_order_by_area, nms, s3fd_predictions
from f2f.utils import (
    get_onnx_cache_dir,
    rename_file_with_hash,
    url_to_local_path,
)

FAN_S3FD_URL = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
)


def export_onnx() -> None:
    model = S3FD().model
    if model.training is False:
        raise RuntimeError("Model must be in eval mode")
    input = torch.randn(1, 3, 640, 640)
    print(f"Exporting {model._get_name()} ONNX...")
    print(f"Use Input: {input.size()}")
    output_names = []
    for i in range(1, 7):
        output_names.append(f"cls_{i}")
        output_names.append(f"reg_{i}")
    file_name = os.path.splitext(os.path.basename(FAN_S3FD_URL))[0] + ".onnx"
    onnx_path = os.path.join(get_onnx_cache_dir(), file_name)
    torch.onnx.export(
        model,
        input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            **{
                k: {0: "batch_size", 2: "height", 3: "width"}
                for k in output_names
            },
        },
    )
    exported_path = rename_file_with_hash(onnx_path)
    print(f"Exported to {exported_path}")


class S3FD(nn.Module):
    nms_threshold: float = 0.3

    def __init__(
        self,
        threshold: float = 0.5,
        pretrained_path: str = FAN_S3FD_URL,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.model = S3FDModel()

        local_path = url_to_local_path(pretrained_path)
        self.model.load_state_dict(
            torch.load(local_path, map_location="cpu"), strict=False
        )
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "S3FD":
        return super().train(False)

    @torch.no_grad()
    def forward(
        self, images: Tensor, center_weight: float = 0.3
    ) -> List[Optional[ndarray]]:
        """
        Args:
            images: (N, C, H, W), range [-1, 1]
        Returns:
            bboxes: list of ndarray (F, 5), [x1, y1, x2, y2, score] or None
        """
        # [cls1, reg1, cls2, reg2, ...], list of 12 tensors
        s3fd_outputs: List[Tensor] = self.model(images)

        np_outputs = [o.cpu().numpy() for o in s3fd_outputs]
        batch_bboxes = s3fd_predictions(np_outputs)

        filtered_bboxes: List[Optional[ndarray]] = []
        for i in range(batch_bboxes.shape[0]):
            bboxes = batch_bboxes[i]
            if bboxes.size == 0:
                filtered_bboxes.append(None)
                continue

            keep = nms(bboxes, self.nms_threshold)
            bboxes = bboxes[keep]

            # thresholding by score
            bboxes = bboxes[bboxes[:, -1] > self.threshold]
            if bboxes.size == 0:
                filtered_bboxes.append(None)
                continue

            # sort by area and center
            H, W = images[i].shape[-2:]
            area_order = cal_order_by_area(
                height=H,
                width=W,
                bboxes=bboxes,
                center_weight=center_weight,
            )
            bboxes = bboxes[area_order]
            filtered_bboxes.append(bboxes)
        return filtered_bboxes


class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale: float = 1.0) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(
            torch.empty(self.n_channels).fill_(self.scale)
        )

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class S3FDModel(nn.Module):
    mean: Tensor
    """tensor([ 104.0, 117.0, 123.0 ]) - 127.5, shape (1, 3, 1, 1)"""

    def __init__(self) -> None:
        super().__init__()
        mean = (
            torch.tensor([104.0, 117.0, 123.0], dtype=torch.float32)
            .view(1, 3, 1, 1)
            .sub(127.5)
        )
        self.register_buffer("mean", mean)

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(
            256, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv3_3_norm_mbox_loc = nn.Conv2d(
            256, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv4_3_norm_mbox_conf = nn.Conv2d(
            512, 2, kernel_size=3, stride=1, padding=1
        )
        self.conv4_3_norm_mbox_loc = nn.Conv2d(
            512, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv5_3_norm_mbox_conf = nn.Conv2d(
            512, 2, kernel_size=3, stride=1, padding=1
        )
        self.conv5_3_norm_mbox_loc = nn.Conv2d(
            512, 4, kernel_size=3, stride=1, padding=1
        )

        self.fc7_mbox_conf = nn.Conv2d(
            1024, 2, kernel_size=3, stride=1, padding=1
        )
        self.fc7_mbox_loc = nn.Conv2d(
            1024, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv6_2_mbox_conf = nn.Conv2d(
            512, 2, kernel_size=3, stride=1, padding=1
        )
        self.conv6_2_mbox_loc = nn.Conv2d(
            512, 4, kernel_size=3, stride=1, padding=1
        )
        self.conv7_2_mbox_conf = nn.Conv2d(
            256, 2, kernel_size=3, stride=1, padding=1
        )
        self.conv7_2_mbox_loc = nn.Conv2d(
            256, 4, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> List[Tensor]:  # return list of 12 tensors
        """
        Args:
            x: (N, 3, H, W), range [-1, 1]
                trained on (N, 3, 640, 640)
        """
        # RGB -> BGR, normalize input
        x = x.flip(1).mul(127.5).sub(self.mean)

        h = F.relu(self.conv1_1(x), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h), inplace=True)
        h = F.relu(self.fc7(h), inplace=True)
        ffc7 = h
        h = F.relu(self.conv6_1(h), inplace=True)
        h = F.relu(self.conv6_2(h), inplace=True)
        f6_2 = h
        h = F.relu(self.conv7_1(h), inplace=True)
        h = F.relu(self.conv7_2(h), inplace=True)
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        bmax = torch.max(cls1[:, :-1], dim=1, keepdim=True)[0]
        cls1 = torch.cat((bmax, cls1[:, -1:]), dim=1)

        # include softmax for onnx export
        if not self.training:
            cls1 = F.softmax(cls1, dim=1)
            cls2 = F.softmax(cls2, dim=1)
            cls3 = F.softmax(cls3, dim=1)
            cls4 = F.softmax(cls4, dim=1)
            cls5 = F.softmax(cls5, dim=1)
            cls6 = F.softmax(cls6, dim=1)
        return [
            cls1,
            reg1,
            cls2,
            reg2,
            cls3,
            reg3,
            cls4,
            reg4,
            cls5,
            reg5,
            cls6,
            reg6,
        ]
