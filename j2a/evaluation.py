from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from j2a.model_interactor import ModelInteractor
from j2a.model import Model, load_llm, AudioProjector

from evaluate import load
import torch


@dataclass
class EvaluationArgs:
    model_path: Path
    wav_file: Path
    prompt_file: Path | None
    device: Optional[str] = "cuda"
    prompt: str | None


def get_args() -> EvaluationArgs:
    parser = ArgumentParser("Driver code.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=f"Path to model (file ending with .ph).",
    )

    parser.add_argument(
        "--wav_file",
        type=str,
        required=True,
        help=f"Path to wav file used as input to the model.",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help=f"Path to where the model should be saved.",
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        default=None,
        help=f"Path to txt file containing prompt.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default=None,
        help=f"The prompt past to the model.",
    )
    args = parser.parse_args()

    if not (model_path := Path(args.model_path)).exists():
        raise FileNotFoundError(
            f"Could not find {model_path}, instantiated by {args.model_path}"
        )

    if not (wav_file := Path(args.wav_file)).exists():
        raise FileNotFoundError(
            f"Could not find {wav_file.absolute()}, instantiated by {args.wav_file}"
        )

    device = args.device
    prompt = args.prompt
    pp = args.prompt_file

    if prompt is None and args.prompt_file is None:
        raise ValueError("Either prompt or prompt file must be specified")

    if pp is not None:
        if not (pp := Path(pp)).exists():
            raise FileNotFoundError(
                f"Could not find {pp.absolute()}, instantiated by {args.prompt_file}"
            )

    eval_args = EvaluationArgs(
        model_path=model_path,
        wav_file=wav_file,
        prompt=prompt,
        prompt_file=pp,
        device=device,
    )

    return eval_args


def load_interactor(
    eval_args: EvaluationArgs, model_id: str = "Open-Orca/Mistral-7B-OpenOrca"
) -> ModelInteractor:
    _, llm = load_llm(model_id=model_id)

    audio_projector = AudioProjector()
    audio_projector.to(eval_args.device)

    model = Model(audio_projector.to(torch.bfloat16), llm, False)

    model_interactor = ModelInteractor(
        model, model_path=eval_args.model_path, device=eval_args.device
    )

    return model_interactor


if __name__ == "__main__":
    eval_args = get_args()
    model_interactor = load_interactor(eval_args)
