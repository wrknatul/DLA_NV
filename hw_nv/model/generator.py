import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from hw_nv.base.base_model import BaseModel
from hw_nv.model.convLayer import conv_transposed
from hw_nv.model.convLayer import conv_1d_layer


class ResidualStack(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int):
        super().__init__()
        self.first_layers = nn.ModuleList([
            conv_1d_layer(n_channels, n_channels, kernel_size, dilation=1),
            conv_1d_layer(n_channels, n_channels, kernel_size, dilation=3),
            conv_1d_layer(n_channels, n_channels, kernel_size, dilation=5)
        ])

        self.second_layers = nn.ModuleList([
            conv_1d_layer(n_channels, n_channels, kernel_size, dilation=1),
            conv_1d_layer(n_channels, n_channels, kernel_size, dilation=1),
            conv_1d_layer(n_channels, n_channels, kernel_size, dilation=1)
        ])

    def forward(self, x):
        for first_l, second_l in zip(self.first_layers, self.second_layers):
            out = first_l(F.leaky_relu(x))
            out = second_l(F.leaky_relu(out))
            x = x + out
        return x


class Generator(nn.Module):
    def __init__(self,
                 in_channels: int = 80,
                 hid_channels: int = 512,
                 strides: Sequence[int] = (8, 8, 2, 2),
                 kernels: Sequence[int] = (16, 16, 4, 4),
                 res_block_kernels: Sequence[int] = (3, 7, 11)):
        super().__init__()
        self.n_blocks = len(strides)
        self.n_res_blocks = len(res_block_kernels)

        assert len(strides) == len(kernels), f'Expected len({strides}) == len({kernels})'
        self.first_conv = conv_1d_layer(in_channels, hid_channels, 7)

        self.tr_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        n_channels = hid_channels
        for stride, kernel_size in zip(strides, kernels):
            self.tr_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    conv_transposed(n_channels, n_channels // 2, kernel_size, stride)
                )
            )
            n_channels //= 2
            for res_kernel in res_block_kernels:
                self.res_blocks.append(ResidualStack(n_channels, res_kernel))

        self.last_conv = conv_1d_layer(n_channels, 1, 7)

    def forward(self, mel, **kwargs):
        out = self.first_conv(mel)
        print("SIZ1", out.size())
        for block_idx, tr_block in enumerate(self.tr_blocks):
            out = tr_block(out)
            print("SIZ2", out.size())
            res_out = self.res_blocks[block_idx * self.n_res_blocks](out)
            for res_idx in range(1, self.n_res_blocks):
                res_out = res_out + self.res_blocks[block_idx * self.n_res_blocks + res_idx](out)
            out = res_out / self.n_res_blocks
            print("SIZ3", out.size())

        out = self.last_conv(F.leaky_relu(out))           
        print("SIZ4", out.size())
        return torch.tanh(out)