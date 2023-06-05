import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)


class FeatureExtractor(nn.Module):
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


class ISPLEfficientNet(EfficientNet):
    """
    Custom implementation from ISPL waiting for the pull-request from the original repo
    It allows for a custom number of input channels
    """

    @classmethod
    def _from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        block_args, global_params = get_model_params(model_name, override_params)
        return cls(block_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls._from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def from_name(cls, model_name, num_classes=1000, in_channels=3):
        model = cls._from_name(model_name, override_params={'num_classes': num_classes})
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetGen, self).__init__()

        if pretrained:
            self.efficientnet = ISPLEfficientNet.from_pretrained(model, in_channels=in_channels, num_classes=n_classes)
        else:
            self.efficientnet = ISPLEfficientNet.from_name(model, in_channels=in_channels)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, n_classes)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        # x = F.softmax(x, dim=-1)
        return x


class EfficientNetGen1vsRest(FeatureExtractor):
    def __init__(self, model: str, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetGen1vsRest, self).__init__()

        if pretrained:
            self.efficientnet = ISPLEfficientNet.from_pretrained(model)
        else:
            self.efficientnet = ISPLEfficientNet.from_name(model, in_channels=in_channels)
        self.classifiers = nn.ModuleList([nn.Linear(self.efficientnet._conv_head.out_channels, 1) for _ in
                                          range(n_classes)])
        del self.efficientnet._fc
        self.n_classes = n_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x, out):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        for i in range(self.n_classes):
            out[:, i] = self.classifiers[i](x)
        # x = F.softmax(x, dim=-1)
        return out


class EfficientNetGenOpenMax(FeatureExtractor):
    def __init__(self, model: str, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetGenOpenMax, self).__init__()

        if pretrained:
            self.efficientnet = ISPLEfficientNet.from_pretrained(model)
        else:
            self.efficientnet = ISPLEfficientNet.from_name(model, in_channels=in_channels)
        self.pu_classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 10*n_classes)
        self.final_classifier = nn.Linear(10*n_classes, n_classes)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.pu_classifier(x)
        x = self.final_classifier(x)
        # x = F.softmax(x, dim=-1)
        return x


class EfficientNetB0(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetB0, self).__init__(model='efficientnet-b0', n_classes=n_classes, pretrained=pretrained,
                                             in_channels=in_channels)


class EfficientNetB4(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4', n_classes=n_classes, pretrained=pretrained,
                                             in_channels=in_channels)


class EfficientNetB01vsRest(EfficientNetGen1vsRest):
    def __init__(self, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetB01vsRest, self).__init__(model='efficientnet-b0', n_classes=n_classes, pretrained=pretrained,
                                             in_channels=in_channels)


class EfficientNetB41vsRest(EfficientNetGen1vsRest):
    def __init__(self, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetB41vsRest, self).__init__(model='efficientnet-b4', n_classes=n_classes, pretrained=pretrained,
                                             in_channels=in_channels)


class EfficientNetB0OpenMax(EfficientNetGenOpenMax):
    def __init__(self, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetB0OpenMax, self).__init__(model='efficientnet-b0', n_classes=n_classes, pretrained=pretrained,
                                             in_channels=in_channels)


class EfficientNetB4OpenMax(EfficientNetGenOpenMax):
    def __init__(self, n_classes: int, pretrained: bool, in_channels: int):
        super(EfficientNetB4OpenMax, self).__init__(model='efficientnet-b4', n_classes=n_classes, pretrained=pretrained,
                                             in_channels=in_channels)