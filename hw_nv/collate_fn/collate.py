import logging
import numpy as np
from typing import List
import torch


from hw_nv.datasets import MelSpectrogram


class Collator(object):
    def __init__(self, mel_spec: dict):
        self.mel_spec = MelSpectrogram(mel_spec)

    def __call__(self, batch: List[dict]):
        length_wave = torch.tensor([len(el["wave"]) for el in batch])

        waves = torch.zeros([len(batch), max(length_wave)])
        for idx, (length, el) in enumerate(zip(length_wave, batch)):
            waves[idx, :length] = torch.tensor(el["wave"])

        mels = self.mel_spec(waves)
        assert len(mels.shape) == 3 and len(mels) == len(waves), f'Unxpected {mels.shape} for {waves.shape}'

        return {
            "wave": waves.unsqueeze(1),
            "length_wave": length_wave,
            "mel": mels
        }