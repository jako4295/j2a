from typing import Callable
from typing_extensions import NotRequired
import pandas as pd  # type: ignore
import torch  # type: ignore
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore
from typing import TypedDict, Tuple
from torch import Tensor


class Batch(TypedDict):
    audio_encoding: Tensor
    label_ids: Tensor
    label_attention_mask: Tensor
    prompt_ids: Tensor
    prompt_attention_mask: Tensor
    end_prompt_ids: Tensor
    end_prompt_attention_mask: Tensor
    _name: NotRequired[str]


class MusicDataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        audio_encoder: Callable | None = None,
    ):
        """
        A torch Dataset for data on the format from the
        :class:`j2a.data_handler.train_test_split.TrainTestSplit().split()`
        class. The csv file must have the following columns:
        - label: The label of the data.
        - prompt: The prompt for the data.
        - name: The name of the data.
        - encoding: The encoding of the data. This can be a path to a .pt file
            or a .wav file. If it is a .wav file, then it is either encoded
            with a specified audio_encoder or the default audio_encoder is
            used. The default audio_encoder is None, which means that the
            encoding is done with the Encoder class from j2a.encoder.encode.

        Parameters
        ----------
        path : str
            Path to csv file. Should have the columns: label, prompt, name, encoding.
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
            Tokenizer for the text data.
        audio_encoder : Callable | None, optional
            Optional audio encoder if the column "encoding" in the csv file
            ends with .wav, then this encoder will be used. Default None.
        """
        # Load train_test_split data
        self.csv_file = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.audio_encoder = audio_encoder

    def __len__(self) -> int:
        """

        Returns
        -------
        int
            Length method for the dataset.
        """
        return len(self.csv_file)

    def __getitem__(self, index: int) -> Batch:
        """Method for getting an item from the dataset class.

        Parameters
        ----------
        index : int
            Index in the dataset.

        Returns
        -------
        Batch
            Returns a TypeDict with the following keys:
            - name: The name of the data.
            - label_ids: The tokenized label ids.
            - label_attention_mask: The attention mask for the label.
            - prompt_ids: The tokenized prompt ids.
            - prompt_attention_mask: The attention mask for the prompt.
            - end_prompt_ids: The tokenized end prompt ids.
            - end_prompt_attention_mask: The attention mask for the end prompt.
            - encoding: The encoding of the data.

        """
        entry = self.csv_file.iloc[index]
        entry_checklist = ["label", "prompt", "name", "encoding"]
        if not all([item in entry.keys() for item in entry_checklist]):
            raise ValueError(
                f"The CSV file does not have the required entries. Must have {entry_checklist}."
            )

        if entry["encoding"].endswith(".pt"):
            encoding = torch.load(entry["encoding"])

        elif entry["encoding"].endswith(".wav"):
            if self.audio_encoder is None:
                from j2a.encoder.encode import Encoder

                encoding_obj = Encoder()
                encoding = encoding_obj.encode_to_tensor(entry["encoding"])
            else:
                encoding = self.audio_encoder(entry["encoding"])
        else:
            raise ValueError(
                "No encoding found in entry of the csv file. encoding has"
                " to be a path ending with .pt (and able to load with "
                "torch.load()) or end with .wav."
            )

        label_ids, label_attention_mask = text_2_ids_and_attention_mask(
            self.tokenizer, entry["label"], truncate=True
        )
        prompt_ids, prompt_attention_mask = text_2_ids_and_attention_mask(
            self.tokenizer, entry["prompt"]
        )
        end_prompt_ids, end_prompt_attention_mask = text_2_ids_and_attention_mask(
            self.tokenizer, end_template(), truncate=True
        )

        batch = Batch(
            _name=entry["name"],
            label_ids=label_ids.squeeze(0),
            label_attention_mask=label_attention_mask.squeeze(0),
            prompt_ids=prompt_ids.squeeze(0),
            prompt_attention_mask=prompt_attention_mask.squeeze(0),
            end_prompt_ids=end_prompt_ids.squeeze(0),
            end_prompt_attention_mask=end_prompt_attention_mask.squeeze(0),
            audio_encoding=encoding,
        )

        return batch


def end_template() -> str:
    return """ <|im_end|><|im_start|> assistant
    """


def text_2_ids_and_attention_mask(
    tokenizer, input_text: str, truncate: bool = False
) -> Tuple[Tensor, Tensor]:
    """Tokenize text and return input_ids and attention_mask.

    Parameters
    ----------
    input_text : str
        text to be tokenized.
    truncate : bool, optional
        If true then the <s> start will be ignored, by default False

    Returns
    -------
    tuple
        Tuple with input_ids and attention_mask from the tokenizer.
    """
    res = tokenizer(input_text, return_tensors="pt")

    if truncate:
        return res.input_ids[:, 1:], res.attention_mask[:, 1:]

    return res.input_ids, res.attention_mask


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = MusicDataset(
        "j2a/data/train-test_2024-04-05_11-13-56/train.csv",
        tokenizer,
    )
    print(dataset[0])
    print(dataset.__len__())
