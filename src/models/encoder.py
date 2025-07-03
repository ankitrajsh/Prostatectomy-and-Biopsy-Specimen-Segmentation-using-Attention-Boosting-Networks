import torch
import torch.nn as nn
import torchvision.models as models

class EncoderWithDbN(nn.Module):
    def __init__(self, base_model='resnet50', pretrained=True):
        super(EncoderWithDbN, self).__init__()

        # Load the base ResNet model
        if base_model == 'resnet50':
            resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Encoder model '{base_model}' is not supported.")

        # Use layers up to the last convolutional layer (remove FC layer)
        self.encoder_layers = nn.Sequential(*list(resnet.children())[:-2])

        # Set up the number of output channels
        self.out_channels = 2048  # For resnet50, this is typically 2048 in the last conv layer

        # Optionally, you can add DbN layers here for feature enhancement

    def forward(self, x):
        return self.encoder_layers(x)