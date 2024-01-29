from itertools import chain
import logging
import random
from pathlib import Path
from random import shuffle

import PIL.Image
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torchaudio
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_nv.base import BaseTrainer
from hw_nv.logger.logger import logger
from hw_nv.logger.utils import plot_spectrogram_to_buf
from hw_nv.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer_disc,
            optimizer_gen,
            config,
            device,
            dataloaders,
            mel_spec,
            batch_accum=1,
            lr_scheduler_disc=None,
            lr_scheduler_gen=None,
            len_epoch=None,
            log_step=50,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer_disc, optimizer_gen, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.mel_spec = mel_spec
        self.batch_accum = batch_accum
        self.len_loader = len(self.train_dataloader)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler_disc = lr_scheduler_disc
        self.lr_scheduler_gen = lr_scheduler_gen
        self.log_step = log_step

        self.metrics = [
            "gen_loss", "disc_loss", "msd_loss", "mpd_loss", "adv_gen_loss",
            "feature_loss", "mel_loss"
        ]
        self.train_metrics = MetricTracker(
            "grad norm disc", "grad norm gen", *self.metrics, writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["wave", "mel"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, parameters=None):
        if parameters is None:
            parameters = self.model.parameters()
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                parameters, self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    idx=batch_idx,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Disc Loss: {:.6f} Gen Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["disc_loss"].item(), batch["gen_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate disc", self.optimizer_disc.param_groups[0]['lr']
                )
                self.writer.add_scalar(
                    "learning rate gen", self.optimizer_gen.param_groups[0]['lr']
                )
                self._log_spectrogram_and_audio(**batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
        log = last_train_metrics

        return log

    def process_batch(self, batch, is_train: bool, idx: int, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        # applying generator
        batch["wave_gen"] = self.model.generator(batch["mel"])
        batch["pred_mel"] = self.mel_spec(batch["wave_gen"].squeeze(1))

        # calculating MPD/MSD losses

        # MPD
        print(batch["wave"].size, batch["wave_gen"].detach().size)
        mpd_real_det, mpd_gen_det, _, _ = self.model.mpd_discriminator(batch["wave"], batch["wave_gen"].detach())
        batch["mpd_loss"] = self.criterion.disc_loss(mpd_real_det, mpd_gen_det)

        # MSD
        msd_real_det, msd_gen_det, _, _ = self.model.msd_discriminator(batch["wave"], batch["wave_gen"].detach())
        batch["msd_loss"] = self.criterion.disc_loss(msd_real_det, msd_gen_det)

        batch["disc_loss"] = batch["mpd_loss"] + batch["msd_loss"]

        if is_train:
            # backward for discriminators
            self.optimizer_disc.zero_grad()
            batch["disc_loss"].backward()
            self._clip_grad_norm(
                chain(self.model.mpd_discriminator.parameters(), self.model.msd_discriminator.parameters())
            )
            self.optimizer_disc.step()
            metrics.update(
                "grad norm disc",
                self.get_grad_norm(
                    chain(self.model.mpd_discriminator.parameters(), self.model.msd_discriminator.parameters())
                )
            )
            if self.lr_scheduler_disc is not None:
                self.lr_scheduler_disc.step()

        # calculating Generator loss = Adversarial loss + lam_ml * Mel-Spec loss + lam_ft * Feature loss
        mpd_real, mpd_gen, mpd_feat_real, mpd_feat_gen = self.model.mpd_discriminator(batch["wave"], batch["wave_gen"])
        mpd_gen_loss = self.criterion.gen_loss(mpd_gen)
        mpd_feat_loss = self.criterion.feature_loss(mpd_feat_real, mpd_feat_gen)

        msd_real, msd_gen, msd_feat_real, msd_feat_gen = self.model.msd_discriminator(batch["wave"], batch["wave_gen"])
        msd_gen_loss = self.criterion.gen_loss(msd_gen)
        msd_feat_loss = self.criterion.feature_loss(msd_feat_real, msd_feat_gen)

        mel_loss = self.criterion.mel_loss(real_spec=batch["mel"], gen_spec=batch["pred_mel"])

        # gathering together
        batch["adv_gen_loss"] = mpd_gen_loss + msd_gen_loss
        batch["feature_loss"] = self.criterion.lam_feat * (mpd_feat_loss + msd_feat_loss)
        batch["mel_loss"] = self.criterion.lam_mel * mel_loss

        batch["gen_loss"] = batch["adv_gen_loss"] + batch["feature_loss"] + batch["mel_loss"]

        if is_train:
            self.optimizer_gen.zero_grad()
            batch["gen_loss"].backward()
            self._clip_grad_norm(self.model.generator.parameters())
            self.optimizer_gen.step()
            metrics.update("grad norm gen", self.get_grad_norm(self.model.generator.parameters()))
            if self.lr_scheduler_gen is not None:
                self.lr_scheduler_gen.step()

        for met in self.metrics:
            try:
                metrics.update(met, batch[met].item())
            except Exception as err:
                self.logger.warning(f'Caught {err}')
                metrics.update(met, np.nan)
        return batch

    def _log_spectrogram_and_audio(self, mel, pred_mel, wave, wave_gen, **batch):
        idx = np.random.choice(np.arange(len(mel)))
        for mels, name in zip([mel, pred_mel], ["target_spec", "pred_spec"]):
            # logging specs
            img = PIL.Image.open(plot_spectrogram_to_buf(mels[idx].detach().cpu().numpy()))
            self.writer.add_image(name, ToTensor()(img))
        for audios, name in zip([wave, wave_gen], ["target_wave", "pred_wave"]):
            # logging audios
            self._log_audio(audios[idx], name)

    def _log_audio(self, audio, name):
        self.writer.add_audio(name, audio, sample_rate=self.config["preprocessing"]["sr"])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, parameters=None, norm_type=2):
        if parameters is None:
            parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))