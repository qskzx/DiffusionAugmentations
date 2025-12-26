import torch.nn as nn
import timm

class Classifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=int(num_classes))

    def forward(self, x):
        return self.backbone(x)
