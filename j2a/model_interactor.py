from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch  # type: ignore
from torch import Tensor
from tqdm import tqdm  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,  # type: ignore
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from j2a.dataset import end_template, text_2_ids_and_attention_mask
from j2a.encoder.encode import Encoder
from j2a.model import AudioProjector, AudioProjectorNoPool, Model, load_llm
from j2a.trainer.save_cfg import SaveCfg
from j2a.trainer.train_cfg import TrainerCfg


class ModelInteractor:
    def __init__(
        self,
        model: Model,
        projector_path: str | Path,
        llm_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.model = model
        self.model.load_projector_from_path(projector_path)
        if llm_path:
            self.model.load_llm_from_path(llm_path)

        self.tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")

    def encode(self, wav_file: str):
        encoding_obj = Encoder()
        encoding = encoding_obj.encode_to_tensor(wav_file)
        encoding_shape = encoding.shape
        return encoding.reshape(1, encoding_shape[0], -1)

    @torch.no_grad()
    def sample_with_audio(
        self, prompt: str, wav_file: str, iteration: int = 50
    ) -> Tensor:
        if wav_file.endswith(".wav"):
            audio_encoding = self.encode(wav_file)
        elif wav_file.endswith(".pt"):
            audio_encoding = torch.load(wav_file)

        else:
            raise ValueError(
                "wav_file needs to be an encoding (.pt file) or a .wav file"
            )
        end_prompt_ids, end_prompt_attention_mask = text_2_ids_and_attention_mask(
            self.tokenizer,
            end_template(),
            truncate=True,
        )
        prompt_ids, prompt_attention_mask = text_2_ids_and_attention_mask(
            self.tokenizer,
            prompt,
        )

        prompt_ids = prompt_ids.to(self.device)
        prompt_attention_mask = prompt_attention_mask.to(self.device)
        end_prompt_attention_mask = end_prompt_attention_mask.to(self.device)
        end_prompt_ids = end_prompt_ids.to(self.device)
        sampled_ids = None

        prompt_embeds = None
        end_prompt_embeds = None
        audio_embeds = None

        with torch.cuda.amp.autocast(
            dtype=torch.float16
        ):  # use float16 to reduce GPU memory
            if audio_embeds is None:
                audio_embeds = self.model.audio_projector(
                    audio_encoding.to(self.device)
                )
            if len(audio_embeds.shape) != 3:
                audio_embeds = audio_embeds.reshape(1, *audio_embeds.shape)

            bs, audio_seq = audio_embeds.shape[:2]
            mask_concat_args = [
                prompt_attention_mask.to("cuda"),
                torch.ones(bs, audio_seq).to(audio_embeds.device),
                end_prompt_attention_mask.to("cuda"),
            ]

            for _ in range(iteration):
                if sampled_ids is not None:
                    mask_concat_args.append(
                        torch.ones(bs, sampled_ids.shape[1]).to(audio_embeds.device)
                    )

                attention_mask = torch.concat(
                    tuple(mask_concat_args),
                    dim=1,
                )

                if prompt_embeds is None:
                    prompt_embeds = self.model.llm.model.embed_tokens(prompt_ids)
                if end_prompt_embeds is None:
                    end_prompt_embeds = self.model.llm.model.embed_tokens(
                        end_prompt_ids
                    )

                sampled_ids_embeds = None
                if sampled_ids is not None:
                    sampled_ids_embeds = self.model.llm.model.embed_tokens(sampled_ids)

                embeds_concat_args = [
                    prompt_embeds,
                    audio_embeds.to(prompt_embeds.dtype),
                    end_prompt_embeds,
                ]
                if sampled_ids_embeds is not None:
                    embeds_concat_args.append(sampled_ids_embeds)

                inputs_embeds = torch.concat(
                    tuple(embeds_concat_args),
                    dim=1,
                )

                mout = self.model.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )

                logits = mout.logits
                sampled = torch.multinomial(logits[:, -1, :].softmax(dim=-1), 1)

                if sampled_ids is None:
                    sampled_ids = sampled
                else:
                    sampled_ids = torch.cat((sampled_ids, sampled), dim=-1).to(
                        self.device
                    )

        return torch.concat(
            (
                prompt_ids,
                end_prompt_ids,
                sampled_ids,
            ),  # type: ignore
            dim=-1,
        )

    def get_sample(self, wav_file: str, prompt: str) -> str:
        sample = self.tokenizer.decode(
            self.sample_with_audio(wav_file=wav_file, prompt=prompt)[0]
        )
        remove = f"<s>{prompt} {end_template()}"

        return sample.replace(remove, "")

    def get_samples_from_list(self, wav_files: List[str], prompt: str) -> List[str]:
        samples = []
        for file in tqdm(wav_files):
            sample = self.get_sample(wav_file=file, prompt=prompt)
            samples.append(sample)
        return samples


if __name__ == "__main__":
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

    model_path = args.model_path
    device = args.device
    wav_file = args.wav_file
    prompt_file = args.prompt_file
    prompt = args.prompt
    if not prompt:
        f = open(prompt_file, "r")
        prompt = f.read()

    tr_cfg = TrainerCfg(
        epoch=1,
        model_save_freq=1,
        device="cuda",
        model_out_dir="folder",
        lr=1.5e-3,
        adam_beta1=0.9,
        adama_beta2=0.999,
        adam_eps=1e-8,
    )
    model_id = "Open-Orca/Mistral-7B-OpenOrca"  # "Open-Orca/Mistral-7B-OpenOrca"  # "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer, llm = load_llm(model_id=model_id)
    audio_projections = AudioProjector()

    audio_projections.to(tr_cfg.device)
    model = Model(audio_projections.to(torch.bfloat16), llm, False)

    model_interactor = ModelInteractor(model, projector_path=model_path, device=device)

    sample = model_interactor.get_sample(wav_file=wav_file, prompt=prompt)

    print(sample)
