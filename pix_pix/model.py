import functools

import torch
import torch.nn as nn


class UnetBlock(nn.Module):
    def __init__(self, config, out_channels, middle_channels, in_channels=None, submodule=None, is_last=False):
        super(UnetBlock, self).__init__()
        self.is_last = is_last

        in_channels = in_channels or out_channels
        downconv = nn.Conv2d(
            in_channels,
            middle_channels,
            kernel_size=config["kernel_size"],
            stride=config['stride'],
            padding=config['padding'],
            bias=False)
        downnorm = nn.BatchNorm2d(middle_channels)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(out_channels)

        if is_last:
            channels_in = 2 * middle_channels
            channels_out = out_channels
            bias = True
            upconv = self._get_upconv(config, channels_in, channels_out, bias)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif not submodule:  # First layer
            channels_in = middle_channels
            channels_out = out_channels
            upconv = self._get_upconv(config, channels_in, channels_out)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            channels_in = 2 * middle_channels
            channels_out = out_channels
            upconv = self._get_upconv(config, channels_in, channels_out)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]

        self.model = nn.Sequential(*model)

    @staticmethod
    def _get_upconv(config, channels_in, channels_out, bias=False):
        return nn.ConvTranspose2d(
            channels_in,
            channels_out,
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['padding'],
            bias=bias)

    def forward(self, x):
        if not self.is_last:
            result = torch.cat([x, self.model(x)], 1)
        else:
            result = self.model(x)
        return result


class UnetGenerator(nn.Module):
    def __init__(self, config):
        super(UnetGenerator, self).__init__()
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        num_downs = config['num_downs']
        filters_num = config["filters_num"]

        unet_block = UnetBlock(config, filters_num * 8, filters_num * 8)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(config, filters_num * 8, filters_num * 8, submodule=unet_block)
        unet_block = UnetBlock(config, filters_num * 4, filters_num * 8, submodule=unet_block)
        unet_block = UnetBlock(config, filters_num * 2, filters_num * 4, submodule=unet_block)
        unet_block = UnetBlock(config, filters_num, filters_num * 2, submodule=unet_block)
        self.model = UnetBlock(
            config, out_channels, filters_num, in_channels=in_channels, submodule=unet_block, is_last=True)

    def forward(self, input):
        result = self.model(input)
        return result + input


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        in_channels = config["out_channels"]
        filters_num = config["filters_num"]
        kernel_size = config["kernel_size"]
        d_depth = config["d_depth"]
        padw = config["padding"]
        sequence = [
            nn.Conv2d(in_channels, filters_num, kernel_size=kernel_size, stride=2, padding=config["padding"]),
            nn.LeakyReLU(0.2, True)
        ]
        mult = 1
        mult_prev = 1

        for i in range(1, d_depth):
            mult_prev = mult
            mult = 2 ** i
            sequence += [
                nn.Conv2d(
                    filters_num * mult_prev,
                    filters_num * mult,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=config["padding"],
                    bias=False),
                nn.BatchNorm2d(filters_num * mult),
                nn.LeakyReLU(0.2, True)
            ]

        mult_prev = mult
        mult = 2 ** d_depth
        sequence += [
            nn.Conv2d(
                filters_num * mult_prev,
                filters_num * mult,
                kernel_size=kernel_size,
                stride=1,
                padding=padw,
                bias=False),
            nn.BatchNorm2d(filters_num * mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(filters_num * mult, 1, kernel_size=kernel_size, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
