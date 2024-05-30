from j2a.data_handler import json_handler
from pydub import AudioSegment  # type: ignore
from pathlib import Path
from argparse import ArgumentParser
import os
from tqdm import tqdm  # type: ignore


class convertor:
    def __init__(self) -> None:
        self.interactor = json_handler.JsonInteractor()
        self.json_dict = self.interactor.json_dict

    def convert_json(self, wav_folder):
        Path(wav_folder).mkdir(parents=True, exist_ok=True)
        for key, value in tqdm(self.json_dict.items()):
            if os.path.exists(value["sound_path"]):
                wav_path = f"{wav_folder}/{key}.wav"
                self.conver_mp3(value["sound_path"], wav_path)
                self.interactor.add_sound_path_wav(key, wav_path)

        self.interactor.save(ask=False, silent=True)

    def conver_mp3(self, mp3, wav):
        sound = AudioSegment.from_mp3(mp3)
        sound.export(wav, format="wav")


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")

    parser.add_argument(
        "--wav_path",
        type=str,
        required=False,
        default=None,
        help="Path to the folder where the wav files is saved",
    )

    args = parser.parse_args()

    # wav_path = "/home/jacob/uniprojects/data_p10/music_data/wav_files"
    wav_path = args.wav_path
    conv = convertor()
    conv.convert_json(wav_path)
