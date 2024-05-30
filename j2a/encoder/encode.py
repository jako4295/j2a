import contextlib
import os.path

import torchaudio  # type: ignore
import gdown  # type: ignore
import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from importlib import import_module  # type: ignore
from argparse import ArgumentParser

from j2a.data_handler.json_handler import JsonInteractor  # type: ignore
from j2a.encoder.mwmae_jax_official.hear_api import RuntimeMAE  # type: ignore


class Encoder:
    model_path: None | str
    weight_bias_path: None | str

    def __init__(self, model_path: None | str = None):
        self.model_path = model_path

    def _save_weights_and_biases(
        self, weight_bias_path: str | None = None, url: None | str = None
    ) -> None:
        if weight_bias_path is None:
            self.weight_bias_path = os.path.dirname(__file__) + "/.weights_biases"
        elif isinstance(weight_bias_path, str):
            self.weight_bias_path = weight_bias_path
        else:
            raise ValueError("weight_bias_path must be a str or None")

        if not os.path.exists(self.weight_bias_path):
            os.makedirs(self.weight_bias_path)
            self._get_weights_and_biases(self.weight_bias_path, url)
            print(f"Saved weights and biases to {self.weight_bias_path}")

    @staticmethod
    def _get_weights_and_biases(output_path: str, url: None | str = None) -> None:
        if url is None:
            url = "https://drive.google.com/drive/folders/1EE82eMsFc0i7qz8MNpDk3u4Ree5v58yd"

        gdown.download_folder(
            url=url, output=output_path, quiet=True, use_cookies=False
        )

    @staticmethod
    def _encode_folder(
        mae: RuntimeMAE,
        audio_folder: str,
        output_path: str,
        save_to_json: bool = True,
        mean_dim: int = 0,
    ) -> None:
        if not os.path.exists(audio_folder):
            raise FileNotFoundError(audio_folder)
        if not os.path.exists(output_path):
            raise FileNotFoundError(output_path)

        if audio_folder[-1] != "/":
            audio_folder += "/"
        if output_path[-1] != "/":
            output_path += "/"

        audio = os.listdir(audio_folder)
        audio = [aud for aud in audio if aud.endswith(".wav")]

        existing_encodings = [
            enc for enc in os.listdir(output_path) if enc.endswith(".pt")
        ]
        interactor = JsonInteractor()
        for aud in tqdm(audio):
            if (aud[:-4] + ".pt") in existing_encodings:
                continue

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                _torch_aud, _ = torchaudio.load(audio_folder + aud)
                _encoded = mae.audio2feats(_torch_aud).mean(dim=mean_dim)

            torch.save(_encoded, output_path + aud[:-4] + ".pt")
            if save_to_json:
                _json_dir_path = os.path.realpath(output_path) + "/" + aud[:-4] + ".pt"
                interactor.json_dict[aud[:-4]]["encoded_path"] = _json_dir_path
                interactor.save(ask=False, silent=True)

    def encode_to_tensor(
        self, audio_file: str, weights_biases_path: None | str = None, mean_dim: int = 0
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_file : str
            Path to the audio file that will be encoded.
        weights_biases_path : None | str, optional, by default None
            Should only be specified if it is not used in our framework.
            If None then it is saved in ./.weights_biases.
        mean_dim : int, optional, by default 0
            If 0 the output dim is (690, 3840)
            If 1 the output dim is (2, 3840)

        Returns
        -------
        torch.Tensor
            Encodings. See the mwmae_jax_official.hear_api.RuntimeMAE.get_scene_embeddings method.
        """
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            mae = self._load_model(weights_biases_path)
            _torch_aud, _ = torchaudio.load(audio_file)
            encoding = mae.audio2feats(_torch_aud).mean(dim=mean_dim)
        return encoding

    def encode_to_file(
        self,
        audio_folder: str,
        output_path: str,
        weights_biases_path: None | str = None,
        save_to_json: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        audio_folder str
            The path to the audio files. Encodes all .wav files in the folder.
        output_path str
            Path to the encoded audio files. Encodes all .wav files in the
            audio_folder.
        weights_biases_path None | str, default None
            Should only be specified if it is not used in our framework.
            If None then it is saved in ./.weights_biases.

        Returns
        -------
        None
        """
        if not os.path.exists(output_path):
            raise FileNotFoundError(output_path)

        mae = self._load_model(weights_biases_path)

        print("Encoding audio files")
        self._encode_folder(mae, audio_folder, output_path, save_to_json)

    def _load_model(self, weights_biases_path: None | str = None) -> RuntimeMAE:
        if self.model_path is None:
            self.model_path = "j2a.encoder.mwmae_jax_official.configs.pretraining.mwmae_base_200_4x16_precomputed"
            print("saving weights and biases for encoder")
            self._save_weights_and_biases(weights_biases_path)

            config = import_module(
                "j2a.encoder.mwmae_jax_official.configs.pretraining.mwmae_base_200_4x16_precomputed"
            ).get_config()

            mae = RuntimeMAE(config, self.weight_bias_path)
        else:
            mae = import_module("hear_api.mwmae_base_200_4x16_384d-8h-4l").load_model()
        return mae


def main():
    parser = ArgumentParser("Driver code.")

    parser.add_argument(
        "--audio_folder",
        type=str,
        required=True,
        help=f"Path to folder with .wav files that will be encoded.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help=f"Path to the encoded .wav files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="If a model is provided this will be used, otherwise the "
        "default model is used.",
    )
    parser.add_argument(
        "--save_to_json",
        type=bool,
        required=False,
        default=True,
        help="In the j2a/data_handler/.json_files/data_summary.json the path to "
        "the encoded .wav files is saved if true",
    )

    args = parser.parse_args()

    if args.model is None:
        encoder = Encoder(args.model)
    else:
        encoder = Encoder()

    encoder.encode_to_file(args.audio_folder, args.output_folder)


if __name__ == "__main__":
    main()
