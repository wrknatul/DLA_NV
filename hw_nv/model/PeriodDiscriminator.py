import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, *args, **kwargs):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))
                ),
                weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, sub_params):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, audio, pred_audio):
        audio_mel_prob = []
        pred_mel_prob = []
        history_audio = []
        history_prob = []
        for i, distance in enumerate(self.discriminators):
            a_m_p, audio_hist = distance(audio)
            p_m_p, pred_hist = distance(pred_audio)
            audio_mel_prob.append(a_m_p)
            history_audio.append(audio_hist)
            pred_mel_prob.append(p_m_p)
            history_prob.append(pred_hist)

        return audio_mel_prob, pred_mel_prob, history_audio, history_prob