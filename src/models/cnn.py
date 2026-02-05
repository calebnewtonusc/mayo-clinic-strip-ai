"""CNN model architectures."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class SimpleCNN(nn.Module):
    """Simple baseline CNN for binary classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetClassifier(nn.Module):
    """ResNet-based classifier with transfer learning support."""

    def __init__(
        self,
        arch: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3
    ):
        super(ResNetClassifier, self).__init__()

        # Load pretrained ResNet
        if arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif arch == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif arch == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify first conv layer if not 3 channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer
        self.backbone.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier."""

    def __init__(
        self,
        arch: str = 'efficientnet_b0',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(EfficientNetClassifier, self).__init__()

        # Load pretrained EfficientNet
        if arch == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif arch == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        elif arch == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            feature_dim = 1408
        elif arch == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = 1536
        elif arch == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            feature_dim = 1792
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# TODO: Add more architectures
# - DenseNet
# - Vision Transformer (ViT)
# - Swin Transformer
# - Custom medical imaging architectures
