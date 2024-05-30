from dataclasses import dataclass


@dataclass
class TrainerCfg:
    model_out_dir: str = "model"
    device: str = "cuda"
    epoch: int = 1
    model_save_freq: int = 1
    lr: float = 1.5e-3
    adam_beta1: float = 0.9
    adama_beta2: float = 0.999
    adam_eps: float = 1e-8
