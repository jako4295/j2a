import time
from contextlib import contextmanager
from typing import Any, Generator, List, Tuple
import numpy as np  # type: ignore

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch import optim
from torch.utils.data import DataLoader  # type: ignore
from torch.amp import autocast  # type: ignore
from functools import partial
from transformers.modeling_outputs import CausalLMOutputWithPast  # type: ignore

from j2a.dataset import MusicDataset
from j2a.trainer.save_cfg import SaveCfg
from j2a.trainer.train_cfg import TrainerCfg
from j2a.model import Model
from j2a.dataset import Batch


class Trainer:
    def __init__(
        self,
        cfg: TrainerCfg,
        model: Model,
        dataset_train: MusicDataset,
        dataset_eval: MusicDataset | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        # TODO look at paged_adamw_8bit
        # if optimizer and schedular should be inputs we can do like huggingface:
        # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adama_beta2),
            eps=cfg.adam_eps,
        )
        self.lr_scheduler = None
        # TODO add lr scheduler (figure out which one to use)
        # self.schedular = optim.lr_scheduler
        self.model = model
        self.model.to(self.device)
        self.train_data_loader: DataLoader[MusicDataset] = DataLoader(
            dataset=dataset_train, batch_size=1, shuffle=True, num_workers=4
        )

        self.eval_data_loader: DataLoader[MusicDataset] | None
        if dataset_eval:
            self.eval_data_loader = DataLoader(
                dataset=dataset_eval, batch_size=1, shuffle=True, num_workers=4
            )
        else:
            self.eval_data_loader = None

    def train(self) -> None:
        start_time = time.time()
        mean_train_losses = []
        mean_eval_losses = []
        all_train_losses = []
        all_eval_losses = []
        for epoch in range(self.cfg.epoch):
            time_fwd: List[float] = []
            time_backprop: List[float] = []
            train_losses = []
            # <train>
            for i, local_batch in enumerate(self.train_data_loader):
                # TODO: Consider constructing a Batch typeddict instead of dict
                # Can be done by _batch ={for...}; batch = _batch(**_batch)
                _batch = {
                    k: v.to(self.device)
                    for k, v in local_batch.items()
                    if not k.startswith("_")
                }
                batch = Batch(**_batch)  # type: ignore

                with timing_context(time_fwd):
                    with autocast(device_type="cuda", dtype=torch.float16):
                        mout, audio_seq = self.model.forward(batch)

                prompt_ids_seq = local_batch["prompt_ids"].shape[1]
                end_prompt_ids_seq = local_batch["end_prompt_ids"].shape[1]
                logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq

                # Why we use logits: https://reneelin2019.medium.com/neural-network-concepts-explained-what-are-logits-c39dd96b2e91

                # remove the last output
                logits = mout.logits
                # remove the prompt and audio seq from logits
                # calculation; additionally, remove the final item
                logits = logits[:, logits_start:-1, :].contiguous()

                # calculate target using only `cap_ids`
                targets = batch["label_ids"][:]
                targets = targets[:, 1:]

                # print("logits", logits.view(-1, logits.shape[-1]).mean(dim=1), logits.view(-1, logits.shape[-1]).std(dim=1))

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), targets.view(-1)
                )

                # ce_loss = nn.CrossEntropyLoss()
                # loss = ce_loss(logits.view(-1, logits.shape[-1]), targets.view(-1))

                with timing_context(time_backprop):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                trn_loss = loss.detach().cpu()
                train_losses.append(trn_loss)
                all_train_losses.append(trn_loss)

                batch_avg_time_sec = np.mean(time_backprop[-100:]) + np.mean(
                    time_fwd[-100:]
                )

                # <house keeping>
                time_backprop = time_backprop[-100:]
                time_fwd = time_fwd[-100:]

                print_status(
                    mode="tr",
                    remaining_time_sec=(len(self.train_data_loader) - (i + 1))
                    * batch_avg_time_sec,
                    running_loss=train_losses,
                )
                # </house keeping>

            # </train>

            # <eval>
            if self.eval_data_loader:
                eval_losses = []
                with torch.no_grad():
                    for local_batch in self.eval_data_loader:
                        # Transfer to GPU
                        _batch = {
                            k: v.to(self.device)
                            for k, v in local_batch.items()
                            if not k.startswith("_")
                        }
                        batch = Batch(**_batch)  # type: ignore
                    with timing_context(time_fwd):
                        with autocast(device_type="cuda", dtype=torch.float16):
                            mout, audio_seq = self.model.forward(batch)

                    prompt_ids_seq = local_batch["prompt_ids"].shape[1]
                    end_prompt_ids_seq = local_batch["end_prompt_ids"].shape[1]
                    logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq

                    logits = mout.logits
                    # remove the prompt and audio seq from logits
                    # calculation; additionally, remove the final item
                    logits = logits[:, logits_start:-1, :].contiguous()

                    # calculate target using only `cap_ids`
                    targets = batch["label_ids"][:]
                    targets = targets[:, 1:]

                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), targets.view(-1)
                    )

                    evl_loss = loss.detach().cpu()
                    eval_losses.append(evl_loss)
                    all_eval_losses.append(evl_loss)

                    print_status(
                        mode="ev",
                        remaining_time_sec=0,
                        running_loss=eval_losses,
                    )
            # </eval>
            # Adjust the learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

            mean_train_losses.append(np.mean(train_losses[-1000:]))
            if self.eval_data_loader:
                mean_eval_losses.append(np.mean(eval_losses[-1000:]))

                print(
                    f"{epoch}:tloss{mean_train_losses[-1]:.4f},eloss{mean_eval_losses[-1]:.4f}"
                )

                if self.cfg.model_out_dir:
                    if epoch != 0 and (epoch % self.cfg.model_save_freq == 0):
                        eval_loss_4f = f"{mean_eval_losses[-1]:.4f}"
                        self.model.save(
                            SaveCfg(
                                epoch=epoch,
                                out_dir=self.cfg.model_out_dir,
                                eval_loss_4f=eval_loss_4f,
                                loss_train=torch.tensor(all_train_losses),
                                loss_eval=torch.tensor(all_eval_losses),
                                time_in_sec=time.time() - start_time,
                            ),
                        )

            else:
                print(f"\n{epoch}: tloss {mean_train_losses[-1]:.4f}\n")

                if self.cfg.model_out_dir:
                    if epoch != 0 and (epoch % self.cfg.model_save_freq == 0):
                        eval_loss_4f = "no_eval"
                        self.model.save(
                            SaveCfg(
                                epoch=epoch,
                                out_dir=self.cfg.model_out_dir,
                                eval_loss_4f=eval_loss_4f,
                                loss_train=torch.tensor(all_train_losses),
                                loss_eval=torch.tensor(all_eval_losses),
                                time_in_sec=time.time() - start_time,
                            ),
                        )


def status_update_line(status: str) -> str:
    return "\x1b[2K%s" % status


def print_status(
    *, mode: str, remaining_time_sec: np.floating[Any] | int, running_loss: list
) -> None:
    print(
        status_update_line(
            "[{}] eta={} loss={:.4f}".format(
                mode,
                seconds_to_human_readable(remaining_time_sec),
                np.mean(running_loss[-100:]),
                # extra_info,
            )
        ),
        end="\r",
    )


def seconds_to_human_readable(elapsed: np.floating[Any] | int) -> str:
    # Calculate days, hours, minutes, and seconds
    days, remainder = divmod(float(elapsed), 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)

    # Format the result as a string
    result = ""
    if days > 0:
        result += f"{int(days)}d"

    if hours > 0:
        if result:
            result += ", "
        result += f"{int(hours)}h"

    # If no days or hours, show minutes
    if not result:
        minutes, seconds = divmod(remainder, 60)
        result += f"{int(minutes)}m"

    return result


@contextmanager
def timing_context(sample: list) -> Generator:
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    sample.append(elapsed_time)
