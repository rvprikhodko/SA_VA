import torch
import torch.nn as nn
from torchvision import models

class VertebralArteryRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super(VertebralArteryRegressor, self).__init__()
        # Загрузка предобученной ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Заменяем последний слой для регрессии двух значений
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # 2 выхода: диаметр левой и правой артерий
        )

    def forward(self, x):
        return self.resnet(x)