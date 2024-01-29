import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from typing import Tuple, Iterable, Dict, Union

from hw_nv.base.base_model import BaseModel
from hw_nv.model.convLayer import conv_1d_layer


class SubMSD(nn.Module):
    def __init__(self, num_channels: int = 128):
        super().__init__()
        assert num_channels % 4 == 0
        self.conv_layers = nn.ModuleList([
            conv_1d_layer(1, num_channels, kernel_size=15, stride=1, padding=7),
            conv_1d_layer(num_channels, num_channels * 2, kernel_size=41, stride=4, groups=4),
            conv_1d_layer(num_channels * 2, num_channels * 4, kernel_size=41, stride=4, groups=16),
            conv_1d_layer(num_channels * 4, num_channels * 8, kernel_size=41, stride=4, groups=64),
            conv_1d_layer(num_channels * 8, num_channels * 8, kernel_size=41, stride=4, groups=256),
            conv_1d_layer(num_channels * 8, num_channels * 8, kernel_size=5, stride=1)
        ])

        self.head = conv_1d_layer(num_channels * 8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, wave):
        assert len(wave.shape) == 3, f'Expected len({wave.shape}) == 3'
        out = wave
        out_layers = []
        for layer in self.conv_layers:
            out = F.leaky_relu(layer(out))
            out_layers.append(out)

        out = self.head(out)
        out_layers.append(out)

        return out, out_layers


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, sub_params: dict, n_pooling: int = 2):
        super().__init__()
        self.sub_discs = nn.ModuleList([SubMSD(**sub_params) for _ in range(n_pooling + 1)])
        self.pooling_layers = nn.ModuleList([nn.AvgPool1d(kernel_size=2**i) for i in range(1, n_pooling + 1)])

    def forward(self, wave, wave_gen):
        real, out_real = self.sub_discs[0](wave)
        gen, out_gen = self.sub_discs[0](wave_gen)
        res_real, res_gen, outputs_real, outputs_gen = [real], [gen], [out_real], [out_gen]
        for idx, pooling in enumerate(self.pooling_layers):
            cur_real = pooling(wave)
            real, out_real = self.sub_discs[idx + 1](cur_real)
            res_real.append(real)
            outputs_real.append(out_real)

            cur_gen = pooling(wave_gen)
            gen, out_gen = self.sub_discs[idx + 1](cur_gen)
            res_gen.append(gen)
            outputs_gen.append(out_gen)

        assert len(res_real) == len(res_gen) == len(outputs_real) == len(outputs_gen)

        return res_real, res_gen, outputs_real, outputs_gen