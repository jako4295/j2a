import glob
import os
import pathlib as pl
from dataclasses import dataclass
from typing import Optional

import torch  # type: ignore


@dataclass
class SaveCfg:
    epoch: int
    out_dir: str
    eval_loss_4f: str

    time_in_sec: Optional[float] = None
    loss_train: Optional[torch.Tensor] = None
    loss_eval: Optional[torch.Tensor] = None
    out_dir_path: Optional[pl.Path] = None

    def __post_init__(self):
        self.out_dir_path = pl.Path(self.out_dir)  # / "model_info"
        if not os.path.exists(self.out_dir_path.__str__()):
            os.makedirs(self.out_dir_path.__str__(), exist_ok=True)

    def output_filename(self):
        if self.out_dir_path is None:
            raise ValueError()

        return f"model_e{self.epoch}_ev{self.eval_loss_4f}.pth"

    def state_from_epoch(self):
        if self.out_dir_path is None:
            raise ValueError()

        return f"{self.out_dir_path}/model_e{self.epoch}_ev{self.eval_loss_4f}.pth"
        # for f in glob.glob(self.out_dir_path / f"model_e{self.epoch}_*"):
        #     return f
