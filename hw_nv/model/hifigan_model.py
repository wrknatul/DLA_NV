import torch
from torch import nn
from torch.nn import functional as F

from hw_nv.base.base_model import BaseModel
from hw_nv.model.ScaleDiscriminator import MultiScaleDiscriminator
from hw_nv.model.PeriodDiscriminator import MultiPeriodDiscriminator
from hw_nv.model.generator import Generator


class HiFiGAN(nn.Module):
    def __init__(self, mpd_params, msd_params, generator_params):
        super().__init__()

        self.mpd_discriminator = MultiPeriodDiscriminator(**mpd_params)
        self.msd_discriminator = MultiScaleDiscriminator(**msd_params)
        self.generator = Generator(**generator_params)

    def __str__(self):
        return self.mpd_discriminator.__str__() + "\n\n\n" + self.msd_discriminator.__str__() + "\n\n\n" + \
            self.generator.__str__()

    def forward(self):
        raise NotImplementedError("Cannot forward, please, use MSD/MPD/Generator components directly instead")