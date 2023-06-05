"""
Custom modification of the InceptionV3 network
"""
import torchvision
from torchvision.models.inception import Inception3
import torch
from torchvision import transforms
from typing import Optional
from torchvision.transforms import Resize

# --- Classes --- #

class FeatureExtractor(torch.nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ISPLInceptionV3(FeatureExtractor):
    """
    Custom implementation of the Inception3 allowing to process an arbitrary number of input channels images
    """
    def __init__(self, num_classes: int=1000, in_channels: int=3, pretrained: bool=True):
        """
        Constructor method
        :param num_classes: number of output classes to predict
        :param in_channels: number of input channels in the image to analyze
        :param pretrained: bool, whether to use the pretrained model or not
        :return: nn.Module, an
        """
        super(ISPLInceptionV3, self).__init__()
        self.inception = Inception3(num_classes=num_classes, init_weights=pretrained)
        if in_channels != 3:
            self.inception.transform_input = False
            self.inception.Conv2d_1a_3x3 = torchvision.models.inception.BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.resize = Resize(size=(299, 299), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != 299 or x.shape[1] != 299:
            x = self.resize(x)
        # N x in_channels x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[torch.Tensor] = None
        if self.inception.AuxLogits is not None:
            if self.training:
                aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # N x num_classes
        x = self.inception.fc(x)
        return x






