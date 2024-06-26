import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,  # type: ignore
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast  # type: ignore
from transformers.models.mistral.modeling_mistral import MistralForCausalLM  # type: ignore

from j2a.dataset import Batch
from j2a.trainer.save_cfg import SaveCfg

logger = logging.getLogger(__name__)


class AudioProjector(nn.Module):
    def __init__(self, output_embedding_size: int = 4096):
        """
        args
            output_embedding_size: int = 4096 / mistral default embedding size
        """
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(250)
        self.proj = nn.Linear(3840, output_embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(3840)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        res = self.pool(encodings.transpose(-2, -1))
        res = self.ln1(res.transpose(-2, -1))
        res = self.proj(res)
        return res


class AudioProjectorNoPool(nn.Module):
    def __init__(self, output_embedding_size: int = 4096):
        """
        args
            output_embedding_size: int = 4096 / mistral default embedding size
        """
        super().__init__()

        self.proj = nn.Linear(3840, output_embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(3840)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        res = self.ln1(encodings)
        res = self.proj(res)
        return res


class Model(nn.Module):
    def __init__(
        self,
        audio_projector: AudioProjector | AudioProjectorNoPool,
        llm: MistralForCausalLM,
        update_llm: bool = True,
    ):
        super().__init__()

        self.llm = llm
        self.audio_projector = audio_projector
        self.update_llm = update_llm

        if not self.update_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def save(self, save_cfg: SaveCfg) -> None:
        if save_cfg.out_dir_path is None:
            raise ValueError("out_dir_path is None.")
        if save_cfg.time_in_sec is None:
            raise ValueError("time_in_sec is None.")
        if save_cfg.loss_eval is None:
            raise ValueError("loss_eval is None.")
        if save_cfg.loss_train is None:
            raise ValueError("loss_train is None.")
        model_name = save_cfg.output_filename()
        logger.info("saving out to", model_name)

        _now = datetime.now()
        stamp = _now.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_cfg.out_dir_path.__str__(), f"model_{stamp}")
        os.makedirs(save_dir)

        if self.update_llm:
            path = os.path.join(save_dir, "llm")
            self.llm.save_pretrained(path)

        torch.save(
            self.audio_projector.state_dict(),
            os.path.join(save_dir, model_name),
        )
        torch.save(
            save_cfg.loss_train,
            os.path.join(save_dir, "loss_train.pt"),
        )
        torch.save(
            save_cfg.loss_eval,
            os.path.join(save_dir, "loss_eval.pt"),
        )
        torch.save(
            torch.tensor(save_cfg.time_in_sec),
            os.path.join(save_dir, "time_in_sec.pt"),
        )

    def load_projector_from_path(self, model_path: str | Path) -> None:
        self.audio_projector.load_state_dict(torch.load(model_path))

    def load_llm_from_path(self, model_path: str | Path) -> None:
        self.llm.from_pretrained(model_path)

    def forward(self, batch: Batch) -> Tuple[CausalLMOutputWithPast, int]:
        audio_encoding = batch["audio_encoding"]
        label_ids = batch["label_ids"]
        label_ids_attention_mask = batch["label_attention_mask"]
        prompt_ids = batch["prompt_ids"]
        prompt_ids_attention_mask = batch["prompt_attention_mask"]
        end_prompt_ids = batch["end_prompt_ids"]
        end_prompt_ids_attention_mask = batch["end_prompt_attention_mask"]

        audio_embeds = self.audio_projector(audio_encoding)
        # print('audio_embeds', audio_embeds.mean(dim=1), audio_embeds.std(dim=1))
        bs, audio_seq = audio_embeds.shape[:2]

        attention_mask = torch.concat(
            (
                prompt_ids_attention_mask,
                torch.ones(bs, audio_seq).to(label_ids.device),
                end_prompt_ids_attention_mask,
                label_ids_attention_mask,
            ),
            dim=1,
        )
        # label_ids = nn.functional.normalize(label_ids)
        # prompt_ids = nn.functional.normalize(prompt_ids)
        # end_prompt_ids = nn.functional.normalize(end_prompt_ids)

        # emb = nn.Embedding(32000, 4096).to("cuda")
        label_embeds = self.llm.model.embed_tokens(label_ids)
        # label_embeds = emb(label_ids.to("cuda"))
        prompt_embeds = self.llm.model.embed_tokens(prompt_ids)
        # prompt_embeds = emb(prompt_ids.to("cuda"))
        end_prompt_embeds = self.llm.model.embed_tokens(end_prompt_ids)
        # end_prompt_embeds = emb(end_prompt_ids.to("cuda"))
        inputs_embeds = torch.concat(
            (
                prompt_embeds,
                audio_embeds.to(label_embeds.dtype),
                end_prompt_embeds,
                label_embeds,
            ),
            dim=1,
        )

        # print('label_embeds', label_embeds.mean(dim=1), label_embeds.std(dim=1))
        mout = self.llm(
            inputs_embeds=inputs_embeds,
            # output_attentions=True,
            # output_hidden_states=True,
            attention_mask=attention_mask,
            # use_cache=False,
        )

        return mout, audio_embeds.shape[1]


def load_llm(
    model_id: str,
) -> tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, MistralForCausalLM]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        # load_in_8bit=True,
    )

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # {"": 0},
        trust_remote_code=False,
        use_safetensors=True,
        quantization_config=bnb_config,
    )

    return tokenizer, model
