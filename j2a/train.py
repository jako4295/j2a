from argparse import ArgumentParser

import torch  # type: ignore

torch.backends.cuda.matmul.allow_tf32 = True

from j2a.dataset import MusicDataset
from j2a.model import AudioProjector, AudioProjectorNoPool, Model, load_llm
from j2a.trainer.train_cfg import TrainerCfg
from j2a.trainer.trainer import Trainer

parser = ArgumentParser("Driver code.")

parser.add_argument(
    "--load_projector_path",
    type=str,
    required=False,
    default=None,
    help=f"Path to pretrained projector weights (.pth file).",
)

parser.add_argument(
    "--load_llm_path",
    type=str,
    required=False,
    default=None,
    help=f"Path to local pretrained llm weights (folder)",
)

parser.add_argument(
    "--train_path",
    type=str,
    required=True,
    help=f"Path to training data (a csv file).",
)

parser.add_argument(
    "--eval_path",
    type=str,
    required=False,
    default=None,
    help=f"Path to evaluation data (a csv file).",
)

parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help=f"Path to where the model should be saved.",
)

parser.add_argument(
    "--model_name",
    type=str,
    required=False,
    default="j2a-2.1",
    help="Model name to be used for predictions",
)

parser.add_argument(
    "--update_llm",
    type=bool,
    required=False,
    default=False,
    help=f"If true the weights of the llm will be updated and saved",
)

args = parser.parse_args()

# Access the values of the arguments
tr_cfg = TrainerCfg(
    epoch=1000,
    model_save_freq=5,
    device="cuda",
    model_out_dir=args.save_path,
    lr=1.5e-3,
    adam_beta1=0.9,
    adama_beta2=0.999,
    adam_eps=1e-8,
)

model_id = "Open-Orca/Mistral-7B-OpenOrca"  # "Open-Orca/Mistral-7B-OpenOrca"  # "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer, llm = load_llm(model_id=model_id)

model_name = args.model_name
audio_projector: AudioProjector | AudioProjectorNoPool
if model_name == "j2a-2.1":
    audio_projector = AudioProjector()
elif model_name == "j2a-2.0":
    audio_projector = AudioProjectorNoPool()
else:
    raise ValueError("Invalid model name")

audio_projector.to(tr_cfg.device)
model = Model(audio_projector.to(torch.bfloat16), llm, args.update_llm)

load_prjector_path = args.load_projector_path
load_llm_path = args.load_llm_path
if load_prjector_path:
    model.load_projector_from_path(load_prjector_path)

if load_llm_path:
    model.load_llm_from_path(load_llm_path)


train_path = args.train_path
eval_path = args.eval_path

train_ds = MusicDataset(train_path, tokenizer)
eval_ds: MusicDataset | None
if eval_path is not None:
    eval_ds = MusicDataset(eval_path, tokenizer)
else:
    eval_ds = None

trainer = Trainer(tr_cfg, model, train_ds, eval_ds)

trainer.train()
