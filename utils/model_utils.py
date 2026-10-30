import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_model(name='resnet18', pretrained=True, num_classes=4, **kwargs):
    if name != 'resnet18':
        raise NotImplementedError(name)
    model = resnet18(weights='DEFAULT') if pretrained else resnet18()
    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=num_classes),
    )
    return model

def load_weight(model, path, device):
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return model
