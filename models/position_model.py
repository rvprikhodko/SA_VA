import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class ArteriesLocalizationModel(nn.Module):

    def __init__(self, pretrained=True):
        super(ArteriesLocalizationModel, self).__init__()

        weights = ResNet101_Weights.DEFAULT if pretrained else None
        self.backbone = resnet101(weights=weights)

        # Заменяем последний полносвязный слой для предсказания 4 координат (x1, y1, x2, y2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Для нормализации координат в диапазоне [0, 1]
        )

    def forward(self, x):
        return self.backbone(x)