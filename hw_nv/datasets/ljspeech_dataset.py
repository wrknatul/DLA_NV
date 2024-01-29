import numpy as np
import random
import os
import soundfile as sf
import time
from tqdm import tqdm

from hw_nv.utils import ROOT_PATH
from hw_nv.logger import logger


class LJspeechDataset(object):
    def __init__(self, cut_audio: int, limit=None):
        self._data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
        self.index = self._load_index()
        self.cut_audio = cut_audio
        if limit is not None:
            random.seed(42)
            random.shuffle(self.index)
            self.index = self.index[:limit]

    def _load_index(self):
        filenames = [filename for filename in (self._data_dir / "wavs").iterdir() if filename.suffix == ".wav"]
        start = time.perf_counter()
        index = []
        for f_name in tqdm(sorted(filenames), desc="Loading wav files"):
            wav, _ = sf.read(os.path.join('', f_name))
            index.append({"wave": wav})

        logger.info(f"Cost {time.perf_counter() - start:.2f}s to load all data.")

        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        d = self.index[item]
        start_pos = np.random.randint(0, max(0, len(d["wave"]) - self.cut_audio) + 1)
        d["wave"] = d["wave"][start_pos:start_pos + self.cut_audio]
        return d