import torch
import torch.nn as nn

from hw_nv.logger import logger
from .feature_loss import FeatureLoss
from .generator_loss import GeneratorLoss
from .discriminator_loss import DiscriminatorLoss
from .mel_loss import MelLoss


class HiFiGANLoss(nn.Module):
    def __init__(self, lam_feat: float = 2, lam_mel: float = 45):
        """
        Construct loss for HiFi-GAN model
        """
        super().__init__()
        self.lam_feat = lam_feat
        self.lam_mel = lam_mel

        self.feature_loss = FeatureLoss()
        self.gen_loss = GeneratorLoss()
        self.mel_loss = MelLoss()
        self.disc_loss = DiscriminatorLoss()

    def forward(self, **kwargs):
        raise NotImplementedError("Cannot use forward directly, because GAN model has different losses")