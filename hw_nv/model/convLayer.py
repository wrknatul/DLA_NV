import torch.nn as nn
from torch.nn.utils import weight_norm


def conv_2d_layer(in_channels, out_channels, kernel_size, stride=1, norm=weight_norm):
    assert kernel_size[1] == 1, f'{kernel_size} expected to have second dimension = 1'
    padding = ((kernel_size[0] - 1) // 2, 0)
    return norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    )


def conv_1d_layer(in_channels,
                  out_channels,
                  kernel_size,
                  padding=None,
                  stride=1,
                  groups=1,
                  dilation=1,
                  norm=weight_norm):
    if padding is None:
        padding = (kernel_size * dilation - dilation) // 2
    return norm(
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
    )


def conv_transposed(in_channels, out_channels, kernel_size, stride=1, norm=weight_norm):
    padding = (kernel_size - stride) // 2
    return norm(
        nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    )