from enum import Enum
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from f2f.recognition.arcface.iresnet import iresnet_from_weights


class ARCFACE_MODELS(Enum):
    R50 = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/arcface_ms1mv3_r50_fp16-ef005ba5.pth"
    R100 = "https://github.com/kynk94/face-to-face/releases/download/weights-v0.1/arcface_ms1mv3_r100_fp16-a566a623.pth"


class Arcface(nn.Module):
    def __init__(
        self,
        weights: Union[ARCFACE_MODELS, str] = ARCFACE_MODELS.R50,
        fp16: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(weights, ARCFACE_MODELS):
            weights = weights.value
        self.backbone = iresnet_from_weights(weights, fp16=fp16)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: shape (B, 3, H, W) representing the input image.
        Returns:
            features: shape (B, 512) representing the embedding.
        """
        return self.backbone(input)

    def cosine_distance(self, features_1: Tensor, features_2: Tensor) -> Tensor:
        """
        Args:
            features_1: shape (B, 512) representing embedding of first image.
            features_2: shape (B, 512) representing embedding of second image.
        Returns:
            distance: shape (B) representing the cosine distance between the \
                two embeddings.
        """
        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)
        return 1 - torch.cosine_similarity(features_1, features_2)
