"""ResNet-50 binary classifier."""

def build_resnet50(pretrained: bool = False):
    import torch.nn as nn
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
