import torch
from torch import nn
from torchvision.models import efficientnet_b0


def create_model(num_classes: int = 7):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def load_model(path: str, device: torch.device, num_classes: int = 7):
    model = create_model(num_classes)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
