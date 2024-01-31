import argparse
import multiprocessing
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from hw_nv.datasets import MelSpectrogram
from hw_nv.logger import WanDBWriter
import hw_nv.model as module_model
from hw_nv.utils import ROOT_PATH
from hw_nv.utils.parse_config import ConfigParser
from train import SEED

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "best_model" / "checkpoint.pth"
torch.manual_seed(SEED)


def main(args, config):
    logger = config.get_logger("test")
    writer: WanDBWriter = WanDBWriter(config, logger) if args.log_wandb else None

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # prepare mel-spec
    mel_spec = MelSpectrogram(config["mel_spec"])
    mel_spec = mel_spec.to(device)

    # read audio filenames and texts
    audio_names = [filename for filename in sorted(Path(args.input_dir).iterdir()) if filename.suffix == ".wav"]
    text_path = Path(args.input_dir) / "text.txt"
    if text_path.exists():
        with open(text_path) as file:
            texts = [line.strip() for line in file.readlines()]

    sr = config["preprocessing"]["sr"]
    with torch.no_grad():
        for idx, audio_name in tqdm(enumerate(audio_names), total=len(audio_names), desc="Infer"):
            name = audio_name.stem
            audio, _ = sf.read(os.path.join('', audio_name))
            audio = torch.tensor(audio).to(torch.float32).unsqueeze(0).to(device)
            spec = mel_spec(audio)
            pred_audio = model.generator(spec).squeeze(1)
            torchaudio.save(Path(args.output) / f"{name}.wav", pred_audio, sample_rate=sr)
            if writer is not None:
                writer.set_step(step=idx, mode="test")
                if text_path.exists():
                    writer.add_text("text", texts[idx])
                writer.add_audio("pred_audio", pred_audio, sample_rate=sr)
                writer.add_audio("target_audio", audio, sample_rate=sr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH.parents[1] / "test_data"),
        help="Path to file with texts and audios to test"
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="Dir to write result audio",
    )
    args.add_argument(
        "-l",
        "--log-wandb",
        default=False,
        type=bool,
        help="Save results in wandb or not (wand params are in config file)"
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # prepare output dir
    Path(args.output).mkdir(exist_ok=True, parents=True)

    main(args, config)