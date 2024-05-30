from argparse import ArgumentParser

import pandas as pd  # type: ignore
from pytube import YouTube  # type: ignore
from pydub import AudioSegment  # type: ignore
import os
from tqdm import tqdm  # type: ignore

from j2a.data_handler.json_handler import JsonInteractor, MetaData, MetaDataEnum  # type: ignore


class YoutubeToMp3:
    """
    Converts youtube video files to mp3 and updating metadata in
    j2a/data_handler/.json_files/data_summary.json.
    """

    base_url: str = "https://www.youtube.com/watch?v="
    data_csv_file: pd.DataFrame

    def __init__(self, data_enum: MetaDataEnum, metadata_path: str | None = None):
        """
        Parameters
        ----------
        data_enum : MetaDataEnum
            The database you want to import from.
        metadata_path : str | None = None
            The path to the metadata file. Defaults to
            location in the package, but can be changed.
        """
        if metadata_path is None:
            metadata_path = os.path.dirname(__file__) + "/metadata"
        meta_data_obj: MetaData = MetaData(metadata_path)
        data_csv_file = getattr(meta_data_obj, f"{data_enum.name}_metadata")

        self.data_csv_file = data_csv_file

        # If .json_files/data_summary.json does not exist it is initialized
        if not os.path.isfile(
            os.path.dirname(__file__) + "/.json_files/data_summary.json"
        ):
            print("data_summary.json is being initialized.")
            meta_data_obj.full_json_file()

    def download_all(self, output_folder: str, file_type: str = "wav") -> None:
        """
        Downloads all videos from the given database specified
        by the construct. The path to the data is also saved
        in j2a/data_handler/.json_files/data_summary.json.

        Parameters
        ----------
        output_folder : str
            Folder to save the downloaded videos.

        Returns
        -------
        None
        """

        if file_type.lower() not in ["wav", "mp3"]:
            raise ValueError(
                f"Filetype not recognised: Must be 'wav' or 'mp3'. Got {file_type}"
            )

        if not isinstance(output_folder, str):
            raise ValueError("output_folder must be a string")

        if output_folder[0] != "/":
            output_folder = os.path.realpath(output_folder)
        if output_folder[-1] == "/":
            output_folder = output_folder[:-1]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Output path did not exist, but is created at \n{output_folder}")
        else:
            pass

        csv_file = self.data_csv_file

        if csv_file.columns.isin(["ytid"]).any():
            url_ext = csv_file["ytid"]
            start_t = csv_file["start_s"]
            end_t = csv_file["end_s"]
            name = "MusicCaps"
        elif csv_file.columns.isin(["video_id"]).any():
            url_ext = csv_file["video_id"]
            start_t = csv_file["start"]
            end_t = csv_file["end"]
            name = "YouTube8M-MusicTextClips"
        else:
            raise ValueError("The video id was not recognized")

        already_downloaded = os.listdir(output_folder)
        already_downloaded = [
            song for song in already_downloaded if song.endswith(file_type)
        ]

        json_interactor = JsonInteractor()
        for i, (yt_id, start, end) in enumerate(
            tqdm(zip(url_ext, start_t, end_t), position=0, leave=True)
        ):
            _file_nam = name + "_" + str(i)
            if f"{_file_nam}.{file_type}" in already_downloaded:
                continue
            try:
                self.__download(yt_id, output_folder, mp3_name="tmp.mp3")
                self.__cut_video(
                    output_path=output_folder,
                    start_time=start,
                    end_time=end,
                    file_name=_file_nam,
                    save_as=file_type,
                )

                if file_type == "wav":
                    json_interactor.add_sound_path_wav(
                        name=_file_nam,
                        sound_path=f"{output_folder}/{_file_nam}.{file_type}",
                    )
                else:
                    json_interactor.add_sound_path(
                        name=_file_nam,
                        sound_path=f"{output_folder}/{_file_nam}.{file_type}",
                    )

                json_interactor.save(ask=False, silent=True)

            except Exception as e:
                print(e)

        print(f"All data is downloaded.")

    @staticmethod
    def __cut_video(
        output_path: str,
        start_time: int,
        end_time: int,
        file_name: str,
        save_as: str = "wav",
    ) -> None:
        """
        Cutting YouTube video according to start and end time
        specified in the database.

        Parameters
        ----------
        output_path : str
            Output path of the video that will be cut in length
        start_time : int
            New start time of the YouTube video relative to the
            original length.
        end_time : int
            New end time of the YouTube video relative to the
            original length.
        file_name : str
            New name of the file.
        save_as : str, optional
            Saves the file as either 'wav' or 'mp3'
        Returns
        -------
        None
        """
        sound = AudioSegment.from_file(output_path + "/tmp.mp3")

        trim = sound[start_time * 1000 : end_time * 1000]  # convert to milliseconds
        trim.export(os.path.join(output_path, f"{file_name}.{save_as}"), format=save_as)

        os.remove(output_path + "/tmp.mp3")

    def __download(
        self,
        url_extended: str,
        output_path: str | None = None,
        mp3_name: str | None = None,
    ) -> None:
        """
        Download YouTube video to mp3 file.

        Parameters
        ----------
        url_extended : str
            The url extension to self.base_url to specify the video to
            be downloaded.
        output_path : str | None = None
            output path for the mp3 file. Default is the directory of this file.
        mp3_name : str | None = None
            The name of the mp3 file.

        Returns
        -------
        None
        """
        url = self.base_url + url_extended
        yt = YouTube(url)

        if yt.age_restricted:
            yt.bypass_age_gate()

        # extract only audio
        video = yt.streams.filter(only_audio=True).first()

        if output_path is None:
            output_path = os.path.dirname(__file__)

        # download the file
        out_file = video.download(output_path=output_path)

        if mp3_name is None:
            mp3_name = url_extended + ".mp3"
        else:
            if not mp3_name.endswith(".mp3"):
                mp3_name = mp3_name + ".mp3"

        outfile_dir = os.path.dirname(out_file)
        # save the file
        os.rename(out_file, outfile_dir + "/" + mp3_name)


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")

    parser.add_argument(
        "--data_enum",
        type=str,
        required=True,
        help=f"MetaDataEnum to download. Options are: {MetaDataEnum.enum_attr()}",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Folder to save the mp3 files"
    )

    args = parser.parse_args()
    enum = getattr(MetaDataEnum, args.data_enum)
    ytmp3 = YoutubeToMp3(data_enum=enum)
    ytmp3.download_all(output_folder=args.output)
