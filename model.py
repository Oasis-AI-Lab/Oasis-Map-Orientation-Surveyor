import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)


class RelativeRotationModel(nn.Module):
    """
    相对旋转方向分类模型。
    支持 EfficientNet-B0（默认）和 MobileNetV3-Small 两种 backbone。
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = False, backbone: str = "efficientnet_b0"):
        super().__init__()

        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        elif backbone == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int = 8, pretrained: bool = False, backbone: str = "efficientnet_b0") -> RelativeRotationModel:
    return RelativeRotationModel(num_classes=num_classes, pretrained=pretrained, backbone=backbone)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
