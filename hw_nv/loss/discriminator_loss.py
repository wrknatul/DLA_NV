import torch
import torch.nn as nn

from hw_nv.logger import logger


class DiscriminatorLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Compute discriminator loss
        L = (D(x) - 1)^2 + D(G(s))^2
        """
        super().__init__(**kwargs)

    def __call__(self, d_real, d_gen):
        disc_loss = 0
        for dr, dg in zip(d_real, d_gen):
            disc_loss += torch.mean((dr - 1)**2) + torch.mean(dg**2)
        return disc_loss