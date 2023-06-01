import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import * 


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class UNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True):
        super(UNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x


        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid: 
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs):

        # Downsample 
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out,  downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_= l(up_, prevs[self.more - idx - 2])
        else:
            up_= down4

        up4= self.up4(up_, down3)
        up3= self.up3(up4, down2)
        up2= self.up2(up3, down1)
        up1= self.up1(up2, in64)

        return self.final(up1)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        print(pad)
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down= nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs= self.down(inputs)
        outputs= self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2 
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2 
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))

        return output


# MultiResUNet
def _conv2dbn(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride', act_fun='LeakyReLU'):
    block = conv(in_f, out_f, kernel_size, stride=stride, bias=bias, pad=pad, downsample_mode=downsample_mode)
    block.add(bn(out_f))
    block.add(act(act_fun))
    return block


class _Add(nn.Module):
    def __init__(self, *args):
        super(_Add, self).__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        return torch.stack(inputs, dim=0).sum(dim=0)


def _MultiResBlockFunc(U, f_in, alpha=1.67, pad='zero', act_fun='LeakyReLU', bias=True):
    W = alpha * U
    out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    model = nn.Sequential()
    deep = nn.Sequential()
    conv3x3 = _conv2dbn(f_in, int(W * 0.167), 3, 1,
                        bias=bias, pad=pad, act_fun=act_fun)
    conv5x5 = conv3x3.add(_conv2dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                    pad=pad, act_fun=act_fun))
    conv7x7 = conv5x5.add(_conv2dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                    pad=pad, act_fun=act_fun))
    shortcut = _conv2dbn(f_in, out_dim, 1, 1,
                         bias=bias, pad=pad, act_fun=act_fun)
    deep.add(Concat(1, conv3x3, conv5x5, conv7x7))
    deep.add(bn(out_dim))
    model.add(_Add(deep, shortcut))
    model.add(act(act_fun))
    model.add(bn(out_dim))
    return model


class _MultiResBlock(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, pad='zero', act_fun='LeakyReLU', bias=True):
        super(_MultiResBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = _conv2dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                  bias=bias, pad=pad, act_fun=act_fun)
        self.conv3x3 = _conv2dbn(f_in, int(W * 0.167), 3, 1, bias=bias,
                                 pad=pad, act_fun=act_fun)
        self.conv5x5 = _conv2dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                 pad=pad, act_fun=act_fun)
        self.conv7x7 = _conv2dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                 pad=pad, act_fun=act_fun)
        self.bn1 = bn(self.out_dim)
        self.bn2 = bn(self.out_dim)
        self.accfun = act(act_fun)

    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], dim=1))
        out = torch.add(self.shortcut(input), out)
        out = self.bn2(self.accfun(out))
        return out


class _PathRes(nn.Module):
    def __init__(self, f_in, f_out, length, pad='zero', act_fun='LeakyReLU', bias=True):
        super(_PathRes, self).__init__()
        self.network = []
        self.network.append(_conv2dbn(f_in, f_out, 3, 1, bias=bias, pad=pad, act_fun=act_fun))
        self.network.append(_conv2dbn(f_in, f_out, 1, 1, bias=bias, pad=pad, act_fun=act_fun))
        self.network.append(bn(f_out))
        for i in range(length - 1):
            self.network.append(_conv2dbn(f_out, f_out, 3, 1, bias=bias, pad=pad, act_fun=act_fun))
            self.network.append(_conv2dbn(f_out, f_out, 1, 1, bias=bias, pad=pad, act_fun=act_fun))
            self.network.append(bn(f_out))
        self.accfun = act(act_fun)
        self.length = length
        self.network = nn.Sequential(*self.network)

    def forward(self, input):
        out = self.network[2](self.accfun(torch.add(self.network[0](input),
                                                    self.network[1](input))))
        for i in range(1, self.length):
            out = self.network[i * 3 + 2](self.accfun(torch.add(self.network[i * 3](out),
                                                                self.network[i * 3 + 1](out))))

        return out


def MulResUnet(num_input_channels=2, num_output_channels=3,
               num_channels_down=(16, 32, 64, 128, 256),
               num_channels_up=(16, 32, 64, 128, 256),
               num_channels_skip=(16, 32, 64, 128),
               filter_size_down=3, filter_size_up=3, filter_skip_size=1, alpha=1.67, need_sigmoid=True, need_bias=True,
               pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):

    assert len(num_channels_down) == len(
        num_channels_up) == (len(num_channels_skip) + 1)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model
    multires = _MultiResBlock(num_channels_down[0], num_input_channels,
                              alpha=alpha, pad=pad, act_fun=act_fun, bias=need_bias)

    model_tmp.add(multires)
    input_depth = multires.out_dim

    for i in range(1, len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        multires = _MultiResBlock(num_channels_down[i], input_depth,
                                  alpha=alpha, pad=pad, act_fun=act_fun, bias=need_bias)

        deeper.add(conv(input_depth, input_depth, 3, stride=2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(input_depth))
        deeper.add(act(act_fun))
        deeper.add(multires)

        if num_channels_skip[i - 1] != 0:
            skip.add(_PathRes(input_depth, num_channels_skip[i - 1], 1, pad=pad, act_fun=act_fun, bias=need_bias))
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        deeper_main = nn.Sequential()

        if i != len(num_channels_down) - 1:
            # not the deepest
            deeper.add(deeper_main)

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(_MultiResBlock(num_channels_up[i - 1], multires.out_dim + num_channels_skip[i - 1],
                                     alpha=alpha, pad=pad, act_fun=act_fun, bias=need_bias))

        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = num_channels_up[0] * alpha
    last_kernel = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)

    model.add(
        conv(last_kernel, num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

