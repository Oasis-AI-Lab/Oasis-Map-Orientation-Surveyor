import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class RelativeRotationModel(nn.Module):
    """
    基于 MobileNetV3_Small_100 的相对旋转方向分类模型
    输出 8 维向量，对应 8 个相对旋转区间 (0-7)
    """

    def __init__(self, num_classes: int = 8, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int = 8, pretrained: bool = False) -> RelativeRotationModel:
    return RelativeRotationModel(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
