import torch
import torch.nn as nn

from hw_nv.logger import logger


class MelLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l1 = nn.L1Loss()

    def __call__(self, real_spec, gen_spec):
        return self.l1(gen_spec, real_spec)