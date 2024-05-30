from contextlib import contextmanager
from typing import Any, List
import numpy as np  # type: ignore
from pathlib import Path
from argparse import ArgumentParser

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader  # type: ignore
from torch.amp import autocast  # type: ignore

from j2a.dataset import MusicDataset
from j2a.model import Model
from j2a.dataset import Batch
from j2a.model import Model, AudioProjector, AudioProjectorNoPool, load_llm


def status_update_line(status: str) -> str:
    return "\x1b[2K%s" % status


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


def eval(model: Model, eval_data_loader: DataLoader, device: str) -> Tensor:
    eval_losses = []
    with torch.no_grad():
        for local_batch in eval_data_loader:
            # Transfer to GPU
            _batch = {
                k: v.to(device) for k, v in local_batch.items() if not k.startswith("_")
            }
            batch = Batch(**_batch)  # type: ignore
        with autocast(device_type="cuda", dtype=torch.float16):
            mout, audio_seq = model.forward(batch)

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

        print_status(
            mode="ev",
            remaining_time_sec=0,
            running_loss=eval_losses,
        )

        return evl_loss


def main(
    model: Model,
    model_list: List[Path],
    eval_data_loader: DataLoader,
    device: str,
    save_path: Path | str,
) -> List[Tensor]:
    if type(save_path) == str:
        save_path = Path(save_path)
    all_eval_losses = []
    epoch = 5
    for model_path in model_list:
        file = list(model_path.glob("*.pth"))[0]
        model.load(str(file))
        evl_loss = eval(model, eval_data_loader, device)
        all_eval_losses.append(evl_loss)
        torch.save(evl_loss, save_path / f"e{epoch}_loss.pt")  # type: ignore
        epoch += 5
    torch.save(all_eval_losses, save_path / "all_loss.pt")  # type: ignore
    return all_eval_losses


def get_list_of_models(model_info_path: Path | str) -> List[Path]:
    if type(model_info_path) == str:
        model_info_path = Path(model_info_path)
    return [p for p in list(model_info_path.iterdir()) if p.is_dir()]  # type: ignore


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")

    parser.add_argument(
        "--model_info_path",
        type=str,
        required=True,
        help=f""""Path to the model_info folder,
        which contains the model epoch folders. 
        Path should include 'model_info'""",
    )

    parser.add_argument(
        "--eval_path",
        type=str,
        required=True,
        help=f"Path to the csv file used for the evaluation",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default="Open-Orca/Mistral-7B-OpenOrca",
        help=f"""Model id to be used for predictions. 
        Default is Open-Orca/Mistral-7B-OpenOrca""",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to be used for predictions. Default is cuda",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="j2a-2.1",
        help=f"""Model name to be used for predictions. 
        Default is j2a-2.1. 
        Options are [j2a-2.1, j2a-2.0]""",
    )

    args = parser.parse_args()

    model_info_path = args.model_info_path
    model_id = (
        args.model_id
    )  # "Open-Orca/Mistral-7B-OpenOrca"  # "mistralai/Mistral-7B-Instruct-v0.2"
    eval_path = args.eval_path
    device = args.device
    model_name = args.model_name

    save_path_start = model_info_path.replace("model_info", "")
    save_path = Path(save_path_start) / "eval_results"
    save_path.mkdir(parents=False, exist_ok=True)

    model_list = get_list_of_models(model_info_path)

    tokenizer, llm = load_llm(model_id=model_id)

    eval_ds = MusicDataset(eval_path, tokenizer)
    eval_data_loader = DataLoader(
        dataset=eval_ds, batch_size=1, shuffle=True, num_workers=4
    )

    audio_projector: AudioProjector | AudioProjectorNoPool
    if model_name == "j2a-2.1":
        audio_projector = AudioProjector()
    elif model_name == "j2a-2.0":
        audio_projector = AudioProjectorNoPool()
    else:
        raise ValueError("Invalid model name")
    audio_projector.to(device)

    model = Model(audio_projector.to(torch.bfloat16), llm, False)

    main(model, model_list, eval_data_loader, device, save_path)
