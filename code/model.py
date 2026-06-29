from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNNEncoder(nn.Module):
    """CPU-friendly CIFAR-10 encoder with global average pooling."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            ConvBlock(3, 64, stride=1),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, feature_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, feature_dim: int = 256, projection_dim: int = 128, projection_hidden_dim: int = 512):
        super().__init__()
        self.encoder = SmallCNNEncoder(feature_dim=feature_dim)
        self.projector = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(x)
        projection = F.normalize(self.projector(representation), dim=1)
        return representation, projection


class LinearProbe(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)
