"""CNN model architectures."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class SimpleCNN(nn.Module):
    """Simple baseline CNN for binary classification.

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Example:
        >>> model = SimpleCNN(in_channels=3, num_classes=2)
        >>> x = torch.randn(8, 3, 224, 224)  # batch of 8 RGB images
        >>> y = model(x)  # Output: (8, 2)
    """

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
    """ResNet-based classifier with transfer learning support.

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Recommended input size: 224x224 for ImageNet pretrained weights

    Example:
        >>> model = ResNetClassifier(arch='resnet50', num_classes=2, pretrained=True)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = model(x)  # Output: (8, 2)
    """

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
    """EfficientNet-based classifier.

    Input shape: (batch_size, 3, height, width)
    Output shape: (batch_size, num_classes)

    Recommended input sizes:
    - EfficientNet-B0: 224x224
    - EfficientNet-B1: 240x240
    - EfficientNet-B2: 260x260
    - EfficientNet-B3: 300x300
    - EfficientNet-B4: 380x380

    Example:
        >>> model = EfficientNetClassifier(arch='efficientnet_b0', num_classes=2)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = model(x)  # Output: (8, 2)
    """

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

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class DenseNetClassifier(nn.Module):
    """DenseNet-based classifier with transfer learning support.

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Recommended input size: 224x224 for ImageNet pretrained weights

    Example:
        >>> model = DenseNetClassifier(arch='densenet121', num_classes=2, in_channels=3)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = model(x)  # Output: (8, 2)
    """

    def __init__(
        self,
        arch: str = 'densenet121',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3
    ):
        super(DenseNetClassifier, self).__init__()

        # Load pretrained DenseNet
        if arch == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = 1024
        elif arch == 'densenet161':
            self.backbone = models.densenet161(pretrained=pretrained)
            feature_dim = 2208
        elif arch == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            feature_dim = 1664
        elif arch == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            feature_dim = 1920
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify first conv layer if not 3 channels
        if in_channels != 3:
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier (always trainable)
        self.backbone.classifier = nn.Linear(feature_dim, num_classes)
        if freeze_backbone:
            # Ensure classifier is trainable even when backbone is frozen
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class VisionTransformerClassifier(nn.Module):
    """Vision Transformer (ViT) classifier.

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Recommended input sizes:
    - ViT-B/16: 224x224 (patch size 16x16)
    - ViT-B/32: 224x224 (patch size 32x32)
    - ViT-L/16: 224x224 (patch size 16x16)
    - ViT-L/32: 224x224 (patch size 32x32)

    Example:
        >>> model = VisionTransformerClassifier(arch='vit_b_16', num_classes=2, in_channels=3)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = model(x)  # Output: (8, 2)
    """

    def __init__(
        self,
        arch: str = 'vit_b_16',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        image_size: int = 224,
        in_channels: int = 3
    ):
        super(VisionTransformerClassifier, self).__init__()

        # Load pretrained ViT
        if arch == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            feature_dim = 768
            patch_size = 16
        elif arch == 'vit_b_32':
            self.backbone = models.vit_b_32(pretrained=pretrained)
            feature_dim = 768
            patch_size = 32
        elif arch == 'vit_l_16':
            self.backbone = models.vit_l_16(pretrained=pretrained)
            feature_dim = 1024
            patch_size = 16
        elif arch == 'vit_l_32':
            self.backbone = models.vit_l_32(pretrained=pretrained)
            feature_dim = 1024
            patch_size = 32
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify patch embedding if not 3 channels
        if in_channels != 3:
            self.backbone.conv_proj = nn.Conv2d(
                in_channels, feature_dim, kernel_size=patch_size, stride=patch_size
            )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        self.backbone.heads = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class SwinTransformerClassifier(nn.Module):
    """Swin Transformer classifier.

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Recommended input size: 224x224 for ImageNet pretrained weights

    Example:
        >>> model = SwinTransformerClassifier(arch='swin_t', num_classes=2, in_channels=3)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = model(x)  # Output: (8, 2)
    """

    def __init__(
        self,
        arch: str = 'swin_t',
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3
    ):
        super(SwinTransformerClassifier, self).__init__()

        # Load pretrained Swin Transformer
        if arch == 'swin_t':
            self.backbone = models.swin_t(pretrained=pretrained)
            feature_dim = 768
            embed_dim = 96
        elif arch == 'swin_s':
            self.backbone = models.swin_s(pretrained=pretrained)
            feature_dim = 768
            embed_dim = 96
        elif arch == 'swin_b':
            self.backbone = models.swin_b(pretrained=pretrained)
            feature_dim = 1024
            embed_dim = 128
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify patch embedding if not 3 channels
        if in_channels != 3:
            # Swin uses a patch embedding in features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, embed_dim, kernel_size=4, stride=4
            )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        self.backbone.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MedicalCNN(nn.Module):
    """Custom CNN designed for medical image analysis.

    Features:
    - Deeper architecture than SimpleCNN
    - Squeeze-and-Excitation (SE) blocks for channel attention
    - Residual connections for better gradient flow
    - Suitable for medical imaging tasks

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Note: Input images should be at least 32x32. Larger images (224x224+) recommended.

    Example:
        >>> model = MedicalCNN(in_channels=3, num_classes=2)
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = model(x)  # Output: (8, 2)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(MedicalCNN, self).__init__()

        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with SE attention."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(batch, channels)
        # Excitation
        y = self.excitation(y).view(batch, channels, 1, 1)
        # Scale
        return x * y.expand_as(x)


def get_model(
    arch: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function to create model by architecture name.

    Args:
        arch: Architecture name
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Model instance

    Example:
        >>> model = get_model('resnet50', num_classes=2, pretrained=True)
        >>> model = get_model('vit_b_16', num_classes=3, pretrained=True)
    """
    arch_lower = arch.lower()

    # ResNet family
    if 'resnet' in arch_lower:
        return ResNetClassifier(
            arch=arch_lower,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    # EfficientNet family
    elif 'efficientnet' in arch_lower:
        return EfficientNetClassifier(
            arch=arch_lower,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    # DenseNet family
    elif 'densenet' in arch_lower:
        return DenseNetClassifier(
            arch=arch_lower,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    # Vision Transformer family
    elif 'vit' in arch_lower:
        return VisionTransformerClassifier(
            arch=arch_lower,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    # Swin Transformer family
    elif 'swin' in arch_lower:
        return SwinTransformerClassifier(
            arch=arch_lower,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    # Simple CNN
    elif arch_lower == 'simplecnn':
        return SimpleCNN(num_classes=num_classes, **kwargs)

    # Medical CNN
    elif arch_lower == 'medicalcnn':
        return MedicalCNN(num_classes=num_classes, **kwargs)

    else:
        raise ValueError(
            f"Unknown architecture: {arch}. "
            f"Supported: resnet*, efficientnet*, densenet*, vit*, swin*, simplecnn, medicalcnn"
        )
