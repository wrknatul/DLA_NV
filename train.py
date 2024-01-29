import argparse
import collections
import itertools
import warnings

import numpy as np
import torch

from hw_nv.datasets import MelSpectrogram
import hw_nv.loss as module_loss
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for disabling scheduler
    # discriminator
    discriminator_params = itertools.chain(
        filter(lambda p: p.requires_grad, model.mpd_discriminator.parameters()),
        filter(lambda p: p.requires_grad, model.msd_discriminator.parameters())
    )
    optimizer_disc = config.init_obj(config["optimizer_disc"], torch.optim, discriminator_params)
    lr_scheduler_disc = config.init_obj(config["lr_scheduler_disc"], torch.optim.lr_scheduler, optimizer_disc)
    # generator
    generator_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    optimizer_gen = config.init_obj(config["optimizer_gen"], torch.optim, generator_params)
    lr_scheduler_gen = config.init_obj(config["lr_scheduler_gen"], torch.optim.lr_scheduler, optimizer_gen)

    # mel-spectrogram converter
    mel_spec = MelSpectrogram(config["mel_spec"])
    mel_spec = mel_spec.to(device)

    trainer = Trainer(
        model,
        loss_module,
        optimizer_disc,
        optimizer_gen,
        config=config,
        device=device,
        dataloaders=dataloaders,
        mel_spec=mel_spec,
        lr_scheduler_disc=lr_scheduler_disc,
        lr_scheduler_gen=lr_scheduler_gen,
        len_epoch=config["trainer"].get("len_epoch", None),
        log_step=config["trainer"].get("log_step", 50)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)