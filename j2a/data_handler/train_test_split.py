from enum import Enum
import json
import os
from datetime import datetime
from typing_extensions import NotRequired, Unpack
from numpy.typing import ArrayLike  # type: ignore
from typing import List, TypedDict
from argparse import ArgumentParser

import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from j2a.data_handler.json_handler import JsonInteractor  # type: ignore


class TTSTypedDict(TypedDict):
    """
    TypedDict for typing on kwargs that are passed to
    sklearn.model_selection.train_test_split
    """

    test_size: NotRequired[float | int | None]
    train_size: NotRequired[float | int | None]
    random_state: NotRequired[int | None]
    shuffle: NotRequired[bool]
    stratify: NotRequired[ArrayLike | None]


class FilterJsonFileBy(Enum):
    """
    Enum for filtering the json dictionary.
    """

    sound_path = ".mp3"
    features = "features"
    metadata = "metadata"
    sound_path_wav = ".wav"
    encoded_path = ".pt"


class TrainTestSplit:
    output_path: str
    dict_copy: dict

    def __init__(self, prompt_path: str) -> None:
        """
        Constructor for TrainTestSplit. Imports data from json file
        and filters dictionary keys that do not have a filepath that
        exists and ends with .mp3. It also removes keys that does
        not contain any features.

        Parameters
        ----------
        prompt_path : str
            Path to a .txt file containing a prompt.
        """
        json_obj: JsonInteractor = JsonInteractor()
        self.dict_copy: dict = json_obj.json_dict.copy()

        self.get_prompt(prompt_path)

    def __filter_dict(self, filterby: List[FilterJsonFileBy]) -> None:
        """
        Updates the self.dict_copy and removes keys that do not
        have a filepath that exists and ends with .mp3. It also
        removes keys that does not contain any features.

        Returns
        -------
        None
        """
        filtered_dict = dict()
        for key, value in self.dict_copy.items():
            should_continue = False
            for f in filterby:  # Guarding filterby
                if f is FilterJsonFileBy.features:
                    if not self.dict_copy[key][f.name]:
                        should_continue = True
                        break
                elif f is FilterJsonFileBy.metadata:
                    if self.dict_copy[key][f.name].get("aspect_list") is None:
                        should_continue = True
                        break
                else:
                    if self.dict_copy[key].get(f.name) is None:
                        should_continue = True
                        break
                    if not self.dict_copy[key][f.name].endswith(f.value):
                        should_continue = True
                        break
                    if not os.path.exists(self.dict_copy[key][f.name]):
                        should_continue = True
                        break

            if should_continue:
                continue

            filtered_dict[key] = value

        self.dict_copy = filtered_dict

    def split(
        self,
        output_path: str,
        from_info_json: str | None = None,
        filterby: List[FilterJsonFileBy] = [],
        validation_size: float | None = None,
        **kwargs: Unpack[TTSTypedDict],
    ) -> None:
        """
        A wrapping function of sklearn.model_selection.train_test_split. If
        from_info_json is provided then it will load this configuration and
        create the train and test data equivalent to the ones created with
        that configuration.

        Parameters
        ----------
        output_path : str
            Path to the output directory. A folder will be created at this
            location with an info.json file, a test.csv file, and train.csv file.
        from_info_json : str, optional
            If provided then train and test data will be created based on this.
            The file is the one created by this method and is named info.json.
        filterby : List[FilterJsonFileBy], optional
            List of enums to filter the json dictionary by.
        kwargs : dict, optional
            Arguments from sklearn.model_selection.train_test_split

        Returns
        -------
        None
        """
        if isinstance(from_info_json, str):
            if from_info_json.endswith(".json"):
                with open(from_info_json) as f:
                    _json_kwargs = json.load(f)

                if isinstance(_json_kwargs.get("filterby"), list):
                    filterby = [
                        getattr(FilterJsonFileBy, filt_nam)
                        for filt_nam in _json_kwargs["filterby"]
                    ]

                    del _json_kwargs["filterby"]

                if _json_kwargs.get("validation_size") is not None:
                    validation_size = _json_kwargs["validation_size"]
                    del _json_kwargs["validation_size"]

                kwargs.update(_json_kwargs)

            else:
                raise ValueError("from_info_json must end with .json")

        if not output_path.endswith("/"):
            output_path += "/"
        self.output_path = output_path
        if kwargs.get("random_state") is None:
            kwargs["random_state"] = 69

        info_dict = self.__kwargs_to_info(kwargs)

        # Filtering dict and removing based on filterlist
        self.__filter_dict(filterby=filterby)

        info_dict["filterby"] = [f.name for f in filterby]

        if isinstance(validation_size, float) and isinstance(
            kwargs["test_size"], float
        ):
            kwargs["test_size"] = kwargs["test_size"] + validation_size

        train_list, test_list = train_test_split(list(self.dict_copy.keys()), **kwargs)

        if isinstance(validation_size, float) and isinstance(
            kwargs["test_size"], float
        ):
            kwargs["test_size"] = kwargs["test_size"] - validation_size
            sum_data_size = (
                info_dict["train_size"] + info_dict["test_size"] + validation_size
            )
            eps = 1e-2
            assert not (
                sum_data_size >= 1.0 + eps or sum_data_size <= 1.0 - eps
            ), "Train size, test size, and validation size does not sum to one"

            _val_size = round(
                validation_size / (info_dict["test_size"] + validation_size), 2
            )
            _test_size = round(1 - _val_size, 2)

            test_list, validation_list = train_test_split(
                test_list,
                train_size=_test_size,
                test_size=_val_size,
                random_state=info_dict["random_state"],
            )
            print(f"val_size: {len(validation_list)}\n")

        info_dict["validation_size"] = validation_size

        if info_dict.get("test_size") is None:
            info_dict["test_size"] = len(test_list)
        if info_dict.get("train_size") is None:
            info_dict["train_size"] = len(train_list)

        print(f"test_size: {len(test_list)}\n")
        print(f"train_size: {len(train_list)}")

        # Save path:
        _now = datetime.now()
        stamp = _now.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = self.output_path + "train-test_" + stamp
        os.mkdir(save_path)

        # Save info_dict to file:
        info_json = json.dumps(info_dict)
        info_json_path = os.path.join(save_path, "info.json")
        with open(info_json_path, "w") as f:
            f.write(info_json)
            print(f"Info saved to {info_json_path}")

        self.__write_train_test_files(train_list, save_path + "/train.csv")
        self.__write_train_test_files(test_list, save_path + "/test.csv")
        if validation_size is not None:
            self.__write_train_test_files(
                validation_list, save_path + "/validation.csv"
            )

    def __write_train_test_files(self, keys_list: list, data_path: str) -> None:
        """
        Write train and test data to files

        Parameters
        ----------
        keys_list : list
            List of names from the json database.
        data_path : str
            path to save

        Returns
        -------
        None
        """
        df = pd.DataFrame(
            columns=["name", "prompt", "encoding", "metadata", "features", "label"]
        )
        for key in keys_list:
            _prompt = self.prompt
            _encoding = self.dict_copy[key].get("encoded_path")
            _metadata = self.dict_copy[key]["metadata"].get("aspect_list")
            _features = self.dict_copy[key].get("features")
            _lab = self.dict_copy[key]["metadata"]
            label = (  # depending on database the metadata is different.
                _lab.get("caption")
                if _lab.get("caption") is not None
                else _lab.get("text")
            )
            df = pd.concat(
                [
                    pd.DataFrame(
                        [[key, _prompt, _encoding, _metadata, _features, label]],
                        columns=df.columns,
                    ),
                    df,
                ],
                ignore_index=True,
            )

        df = df.set_index("name")
        df.to_csv(data_path)

    @staticmethod
    def __kwargs_to_info(split_kwargs: TTSTypedDict) -> dict:
        """
        Takes kwargs that are passed to sklearn.model_selection.train_test_split.

        Parameters
        ----------
        split_kwargs : dict
            Arguments passed to sklearn.model_selection.train_test_split.

        Returns
        -------
        Not none valued dictionary items.
        """
        info_dict = dict()
        for key, value in split_kwargs.items():
            if value is not None:
                info_dict[key] = value

        # If nothing is specified, then we don't save this
        return info_dict

    def get_prompt(self, path: str) -> None:
        """
        Load .txt file and set it as prompt

        Parameters
        ----------
        path : str
            Path to .txt file.
        """
        with open(path) as f:
            self.prompt = f.read()


if __name__ == "__main__":

    parser = ArgumentParser("Driver code.")

    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to a .txt file containing a prompt.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the folder where a folder with train.csv, test.csv, and info.json is created.",
    )

    args = parser.parse_args()
    filterby = [FilterJsonFileBy.encoded_path, FilterJsonFileBy.sound_path_wav]
    tts = TrainTestSplit(args.prompt_path)
    # tts.split(
    #     output_path=args.output_dir,
    #     from_info_json="/home/jam/private/uni/P10/data/train/info.json",
    # )
    tts.split(
        args.output_dir,
        filterby=filterby,
        random_state=69,
        train_size=0.8,
        test_size=0.10,
        validation_size=0.10,
    )
