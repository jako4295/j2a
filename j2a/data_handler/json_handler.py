import os
from enum import Enum
from typing import Type, Any, List

import beautifultable  # type: ignore
import pandas as pd  # type: ignore
import dictdiffer  # type: ignore
import json


class MetaDataEnum(Enum):
    musiccaps = "MusicCaps"
    # musicnet = "MusicNet"
    youtube_8m = "YouTube8M-MusicTextClips"

    @classmethod
    def enum_attr(cls):
        """
        Returns
        -------
        Gives all available enum items
        """
        return cls.__members__.keys()


class MetaData:
    """
    This class handles metadata related to the databases specified by
    the MetaDataEnum class. Metadata is imported in the constructor
    and the method ensures that all metadata is imported correctly.

    MetaData.full_json_file saves the metadata from the sound files
    in a json file located at j2a/data_handler/.json_files/data_summary.json.
    """

    def __init__(self, path: str) -> None:
        """
        In the folder there should be metadata from the databases defined in
        MetaDataEnum.

        Parameters
        ----------
        path : str
            Path to the metadata folder.
        """
        if not path.endswith("/"):
            path = path + "/"

        self._get_metadata(path)

    def _get_metadata(self, path: str) -> None:
        """
        Imports all metadata from the path specified.
        This data is accessed as attributes for this class.

        Parameters
        ----------
        path : str
            Path to the metadata folder.

        Returns
        -------
        None
        """
        metadata: Type[MetaDataEnum] = MetaDataEnum

        for database in metadata.enum_attr():
            match getattr(metadata, database).value:
                case "MusicCaps":
                    _filename: str | list = "musiccaps-public.csv"
                case "MusicNet":
                    _filename = "musicnet_metadata.csv"
                case "YouTube8M-MusicTextClips":
                    _filename = ["test.csv", "train.csv"]
                case _:
                    raise ValueError("The enums did not match the expected values")

            if isinstance(_filename, str):
                setattr(self, database + "_metadata", pd.read_csv(path + _filename))
            elif isinstance(_filename, list):
                df = [pd.read_csv(path + fil_nam) for fil_nam in _filename]
                setattr(self, database + "_metadata", pd.concat(df, ignore_index=True))

            _df = getattr(self, database + "_metadata")
            _df["_name"] = _df.index.to_series().map(
                lambda x: f"{getattr(metadata, database).value}_{x}"
            )

    def full_json_file(self) -> None:
        """
        Saves the full json file of the metadata. It is located at
        j2a/data_handler/.json_files/data_summary.json.

        NOTE: This method should only be used to initialize the
              json file because it overwrites the existing file.

        To interact with the json file please use the
        :class:`j2a.data_handler.json_handler.JsonInteractor`

        Returns
        -------
        None
        """
        _output_path = os.path.dirname(__file__).split("/")
        output_path = "/".join(_output_path) + "/.json_files"

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            pass

        if os.path.exists(f"{output_path}/data_summary.json"):
            print(
                "Are you sure you want to overwrite the already "
                "existing data_summary.json? [y/n]"
            )
            answer = input().lower()
            if answer != "y":
                return None

        database_enum: Type[MetaDataEnum] = MetaDataEnum
        full_dict = dict()
        for database in database_enum.enum_attr():
            json_str: str = self.metadata_to_json(getattr(MetaDataEnum, database))
            database_dict = json.loads(json_str)
            full_dict.update(database_dict)

        full_json = json.dumps(full_dict)

        with open(output_path + "/data_summary.json", "w") as f:
            f.write(full_json)
            print(f"file saved in  \n {output_path}/data_summary.json")

    def metadata_to_json(self, database: MetaDataEnum) -> str:
        """
        Converts from metadata to Json str.

        Parameters
        ----------
        database : MetaDataEnum
            Name of the database in the MetaDataEnum class

        Returns
        -------
        The json as a string.
        """

        if not hasattr(self, database.name + "_metadata"):
            raise ValueError("The metadata is not instantiated yet.")

        df = getattr(self, database.name + "_metadata")
        df = df.set_index("_name")

        df_to_dict: dict[Any, dict[Any, Any]] = dict()
        for index, row in df.iterrows():
            df_to_dict[index] = dict()
            df_to_dict[index]["sound_path"] = ""
            df_to_dict[index]["metadata"] = row.to_dict()
            df_to_dict[index]["features"] = dict()

        json_string = json.dumps(df_to_dict)

        return json_string

    def _get_fma_csv(self, path: str) -> pd.DataFrame:
        """
        For now we ignore this database, but in the future
        this method might be relevant

        Database can be found here:
        https://github.com/mdeff/fma#MIT-1-ov-file
        """
        raise NotImplementedError("Not implemented yet")

    def _get_mtg_jamendo_csv(self, path: str) -> pd.DataFrame:
        """
        For now we ignore this database, but in the future
        this method might be relevant.

        Database can be found here:
        https://github.com/MTG/mtg-jamendo-dataset
        """
        raise NotImplementedError("Not implemented yet")


class JsonInteractor:
    """
    This class is used to interact with json file located
    at j2a/data_handler/.json_files/data_summary.json.

    If this file is not yet created please use the
    :class:`j2a.data_handler.json_handler.MetaData`.
    """

    json_path: str
    json_dict: dict

    def __init__(self):
        current_file = os.path.dirname(__file__)
        self.json_path = current_file + "/.json_files/data_summary.json"
        if not os.path.exists(self.json_path):
            raise ValueError(
                "Please download the data_summary.json. This is done "
                "by the following lines of code: \n"
                ">>> metadata: MetaData = MetaData('path/to/metadata/') \n"
                ">>> metadata.full_json_file()"
            )
        with open(self.json_path) as f:
            self.json_dict = json.load(f)

    def clean_json_param(self, param: str) -> None:
        """
        This method cleans a parameter from the json file.
        For instance if param='sound_path' then all sound_paths
        will be replaced with "".

        Parameters
        ----------
        param : str
            The name of the parameter to be cleaned

        Returns
        -------
        None
        """
        all_keys = self.json_dict.keys()
        for key in all_keys:
            if self.json_dict[key].get(param) is not None:
                self.json_dict[key][param] = ""

    def update_json_paths(self, data_path: str, param: str) -> None:
        """
        This method updates the paths in the json file to
        the data given in data_path. It updates the paramter
        given by param (has to be "sound_path", "sound_path_wav",
        or "encoded_path")
        Parameters
        ----------
        data_path : Path to folder with data
        param : Name of the parameter to be updated

        Returns
        -------
        None
        """
        endswith: str
        match param:
            case "sound_path":
                endswith = ".mp3"
            case "sound_path_wav":
                endswith = ".wav"
            case "encoded_path":
                endswith = ".pt"
            case _:
                raise ValueError(f"Unexpected param: {param}")
        if not os.path.exists(data_path):
            raise ValueError(f"Filepath does not exist")
        if data_path[0] != "/":
            data_path = os.path.realpath(data_path)

        data_in_path = [dat for dat in os.listdir(data_path) if dat.endswith(endswith)]
        for dat in data_in_path:
            file_nam = dat[: -len(endswith)]
            self.json_dict[file_nam][param] = os.path.join(data_path, dat)

    def add_feature(self, name: List[str] | str, feature: List[Any]) -> None:
        """
        Adds a feature to the given names. Note that the names has
        to be in the self.json_dict.

        This method does NOT write to file, but updates self.json_dict.
        To update the file with changes then use self.save().

        Parameters
        ----------
        name : List[str] | str
            If list of string is provided, the names has to be in the
            data_summary.json. Otherwise, the string options are "all",
            "musiccaps", and "youtube_8m".
        feature : List[Any]
            Feature to add to the data_summary.json.

        Returns
        -------
        None
        """
        if len(name) != len(feature):
            raise ValueError("len(feature) does not match len(name)")
        if isinstance(name, str):
            match name.lower():
                case "all":
                    name = list(self.json_dict.keys())
                case "musiccaps":
                    name = []
                    for _key in self.json_dict.keys():
                        if "musiccaps" in _key.lower():
                            name.append(_key)
                        else:
                            continue
                case "youtube_8m":
                    name = []
                    for _key in self.json_dict.keys():
                        if "youtube8m" in _key.lower():
                            name.append(_key)
                        else:
                            continue
                case _:
                    raise ValueError("The string input is not recognized.")

        for nam, feat in zip(name, feature):
            self.json_dict[nam]["features"] = feat

    def add_sound_path(self, name: str, sound_path: str) -> None:
        """
        Add a sound path to the self.json_dict.

        This method does NOT write to file, but updates self.json_dict.
        To update the file with changes then use self.save().

        Parameters
        ----------
        name : str
            The name has to be in self.json_dict and is the name where
            the sound path is added.
        sound_path : str
            Please provide the absolute path to the sound file

        Returns
        -------
        None
        """
        if not os.path.exists(sound_path):
            raise ValueError(
                "The sound path does not exist and "
                "can therefore not be added to the data_summary.json"
            )
        # If the path is not absolute then we reformat it
        if sound_path[0] != "/":
            sound_path = os.path.realpath(sound_path)
        if name not in self.json_dict.keys():
            raise ValueError("The name does not appear in data_summary.json")

        self.json_dict[name]["sound_path"] = sound_path

    def add_sound_path_wav(self, name: str, sound_path: str) -> None:
        """
        Add a sound path to the self.json_dict. The path will be added under sound_path_wav

        This method does NOT write to file, but updates self.json_dict.
        To update the file with changes then use self.save().

        Parameters
        ----------
        name : str
            The name has to be in self.json_dict and is the name where
            the sound path is added.
        sound_path : str
            Please provide the absolute path to the wav sound file

        Returns
        -------
        None
        """
        if not os.path.exists(sound_path):
            raise ValueError(
                "The sound path does not exist and "
                "can therefore not be added to the data_summary.json"
            )
        # If the path is not absolute then we reformat it
        if sound_path[0] != "/":
            sound_path = os.path.realpath(sound_path)
        if name not in self.json_dict.keys():
            raise ValueError("The name does not appear in data_summary.json")

        self.json_dict[name]["sound_path_wav"] = sound_path

    def get_info(self, name: str | List[str]) -> None:
        """
        Get info about one or multiple elements in self.json_dict.

        Parameters
        ----------
        name : str | List[str]
            If str then it is the name of the element in self.json_dict.
            If list then it is a list of the elements in self.json_dict.

        Returns
        -------
        None
        """
        table = beautifultable.BeautifulTable()
        if isinstance(name, str):
            table = self.__get_info(name, table)
        elif isinstance(name, list):
            for i, name in enumerate(name):
                table = self.__get_info(name, table)
        else:
            raise ValueError("Unexpected name type")

        print(table)

    def __get_info(
        self, name: str, table: beautifultable.BeautifulTable
    ) -> beautifultable.BeautifulTable:
        """
        Putting info from name in self.json_dict into a
        beautifultable.BeautifulTable object.

        Parameters
        ----------
        name : str
            It is the name of the element in self.json_dict.
        table : beautifultable.BeautifulTable
            Appends info about element to the given table.
            the table can be empty.

        Returns
        -------
        beautifultable.BeautifulTable
            Updated the table from the input with info about
            the element specified by name.
        """
        if name not in self.json_dict.keys():
            raise ValueError("The name does not appear in data_summary.json")
        dict_entrance = self.json_dict[name]

        rows_header_meta = list(dict_entrance["metadata"].keys())
        rows_header_feat = list(dict_entrance["features"].keys())

        table.rows.append(["Info:", name])
        table.rows.append(["Sound file", dict_entrance["sound_path"]])

        if rows_header_meta:
            _table_meta = beautifultable.BeautifulTable()
            for _key, _val in dict_entrance["metadata"].items():
                _table_meta.rows.append([_key, _val])

            table.rows.append(["Metadata", _table_meta])

        if rows_header_feat:
            _table_feat = beautifultable.BeautifulTable()
            for _key, _val in dict_entrance["features"].items():
                _table_feat.rows.append([_key, _val])

            table.rows.append(["Features", _table_feat])

        return table

    def save(self, ask: bool = True, silent: bool = False) -> None:
        """
        Saves changes in self.json_dict to the file in
        j2a/data_handler/.json_files/data_summary.json

        Parameters
        ----------
        ask : bool, optional
            If True, asks the user to save the changes
            if False, any changes will be made without asking.
        silent: bool, optional
            If silent mode is on there will only be printed if ask is True

        Returns
        -------
        None
        """
        with open(self.json_path) as f:
            old_json = json.load(f)

        dictdiffer_list = list(dictdiffer.diff(old_json, self.json_dict))

        if not dictdiffer_list and not silent:
            print("No changes are made to data_summary.json")
            return None

        if not silent:
            for diff in dictdiffer_list:
                print(diff)

        if not ask:
            json_str = json.dumps(self.json_dict)
            with open(self.json_path, "w") as f:
                f.write(json_str)

        else:
            print(
                "Are you sure you want to make the changes to the data_summary.json? [y/n]"
            )
            answer = input()
            if answer.lower() == "y":
                json_str = json.dumps(self.json_dict)
                with open(self.json_path, "w") as f:
                    f.write(json_str)
                print(f"Saved changes to \n{self.json_path}")
            else:
                print("No changes are made to file data_summary.json")


if __name__ == "__main__":
    metadata: MetaData = MetaData("metadata/")
    metadata.full_json_file()

    json_obj: JsonInteractor = JsonInteractor()
    json_obj.add_sound_path("MusicCaps_0", "data/MusicCaps_0.mp3")
    json_obj.save()
