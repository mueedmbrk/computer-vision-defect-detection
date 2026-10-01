import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
