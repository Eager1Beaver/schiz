import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    A feature extraction module based on a pre-trained model (ResNet18).
    This module removes the classification layer and modifies the first convolutional 
    layer to support a custom number of input channels.

    Attributes:
        feature_extractor (nn.Sequential): The modified backbone model without the classification head.
    """

    def __init__(self,
                 base_model_name: str = 'resnet18',
                 weights=models.ResNet18_Weights.IMAGENET1K_V1,
                 input_channels: int = 1) -> None:
        """
        Initializes the FeatureExtractor.

        Parameters:
            base_model_name (str): The name of the base model to use (default: 'resnet18').
            weights (torchvision.models.Weights): Pretrained weights for the base model.
            input_channels (int): Number of input channels for the first convolutional layer.
        
        Returns:
            None
        """
        super(FeatureExtractor, self).__init__()
        base_model = getattr(models, base_model_name)(weights=weights)
        # Remove the classification layer (fully connected layer)
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1])

        # Modify the first convolutional layer to match the desired number of input channels
        conv1 = self.feature_extractor[0]
        self.feature_extractor[0] = nn.Conv2d(in_channels=input_channels,
                                              out_channels=conv1.out_channels,
                                              kernel_size=conv1.kernel_size,
                                              stride=conv1.stride,
                                              padding=conv1.padding,
                                              bias=conv1.bias)
        # Copy the weights from the first channel of the original conv1 layer
        self.feature_extractor[0].weight.data = conv1.weight.data.mean(
            dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Extracted features in shape (batch_size, feature_dim).
        """
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, feature_dim)
