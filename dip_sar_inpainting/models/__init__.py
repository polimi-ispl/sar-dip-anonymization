from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet, MulResUnet

import torch.nn as nn
import torch


# Old DIP original repo implementation
def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU',
            skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5,
            downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        net = ResNet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels,
                   num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d,
                                                                            int) else skip_n33d,
                   num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u,
                                                                          int) else skip_n33u,
                   num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11,
                                                                           int) else skip_n11,
                   upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios=[32, 16, 8, 4, 2, 1],
                               fill_noise=False, pad=pad)

    elif NET_TYPE == 'UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=3,
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d,
                   need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()
    else:
        assert False

    return net


def create_network(net_name: str, input_depth: int, pad: str, upsample: str, activation: str, need_sigmoid: bool,
                   dtype: torch.TensorType, num_channels: int):
    if 'skip' in net_name:

        depth = int(net_name[-1])
        net = skip(input_depth, num_channels, #img_np.shape[0]
                   num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                   num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                   num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                   filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                   upsample_mode=upsample,  # downsample_mode='avg',
                   need1x1_up=False,
                   need_sigmoid=need_sigmoid, need_bias=True, pad=pad, act_fun=activation).type(dtype)

    elif net_name == 'Unet':

        net = UNet(num_input_channels=input_depth, num_output_channels=num_channels,
                   feature_scale=8, more_layers=1,
                   concat_x=False, upsample_mode=upsample,
                   pad=pad, norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=need_sigmoid, need_bias=True).type(dtype)

    elif net_name == 'ResNet':

        net = ResNet(input_depth, num_channels, 8, 32, need_sigmoid=need_sigmoid, act_fun=activation).type(dtype)

    elif net_name == 'MultiResUnet':
        # Found by Fantong, don't know the exact source
        net = MulResUnet(num_input_channels=input_depth,
                         num_output_channels=num_channels,
                         num_channels_down=[16, 32, 64, 128, 256],
                         num_channels_up=[16, 32, 64, 128, 256],
                         num_channels_skip=[16, 32, 64, 128],
                         upsample_mode=upsample,  # default is bilinear
                         need_sigmoid=need_sigmoid,
                         need_bias=True,
                         pad=pad,  # default is reflection, but Fantong uses zero
                         act_fun=activation  # default is LeakyReLU).type(self.dtype)
                         ).type(dtype)
    else:
        assert False

    return net