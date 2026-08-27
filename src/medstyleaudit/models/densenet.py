"""DenseNet-121 binary classifier."""

def build_densenet121(pretrained: bool = False):
    import torch.nn as nn
    from torchvision.models import DenseNet121_Weights, densenet121

    model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model
