import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models
from torchvision.models.densenet import _DenseBlock, _Transition
from typing import Tuple
from collections import OrderedDict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class DenseNet121(nn.Module):
    def __init__(self, n_inputs=12, numCls=17):
        super().__init__()

        #        densenet = models.densenet121(pretrained=False, memory_efficient=True)
        densenet = models.densenet121(pretrained=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            *densenet.features[1:])

        # classifier
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(65536, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


class DenseNetFullDropout(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNetFullDropout, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.features.add_module('dropout0', nn.Dropout())

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            self.features.add_module('dropout%d' % (i + 1), nn.Dropout())
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.features.add_module('dropout5', nn.Dropout())
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class DenseNet121FullDropout(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

#        densenet = models.densenet121(pretrained=False, memory_efficient=True)
        densenet = DenseNetFullDropout(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *densenet.features[1:])
        
        # classifier
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(65536, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
    
    
class DenseNet161(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        densenet = models.densenet161(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(141312, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 
    
    
    
class DenseNet169(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        densenet = models.densenet169(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(106496, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 
    
    
class DenseNet201(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        densenet = models.densenet201(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(122880, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits    
    
    
    
    

if __name__ == "__main__":
    
    inputs = torch.randn((2, 12, 256, 256)) # (how many images, spectral channels, pxl, pxl)
    
    #
    import time
    start_time = time.time()
    #
    
    net = DenseNet121()


    outputs = net(inputs)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    print(outputs)
    print(outputs.shape)

    numParams = count_parameters(net)

    print(f"{numParams:.2E}")
 


  