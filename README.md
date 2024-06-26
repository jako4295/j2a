# j2a

![J2A](https://img.shields.io/badge/J2A-Music%20Captioning-blue)

This repository is used for creating a multimodal LLM-based model used for generating music captions given an audio file (currently only supported by .wav files). Note that the code itself is under the Apache 2.0 license, but any models trained using the YouTube8M-MusicTextClips dataset are under the [Research-only, non-commercial Adobe Research License](./LICENSE2). However, the MusicCaps dataset is under the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) and can be used for commercial purposes.

**Documentation page is found [here](https://jako4295.github.io/docs/j2a/docs/build/html/index.html).**

<!--
| Model name | Model link    |
| ---------- | ------------- |
| J2A-2.0    | Link to model |
| J2A-2.1    | Link to model |
-->

## <a name='TableofContents'></a>Table of Contents

<!-- vscode-markdown-toc -->

- [Installation](#Installation)
- [Usage](#Usage)
  - [Getting Started](#GettingStarted)
  - [Data Extractor](#DataExtractor)
    - [`JsonInteractor`](#JsonInteractor)
  - [Feature Extractor](#FeatureExtractor)
  - [Encoder](#Encoder)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=false
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Installation'></a>Installation

For developer installation, run the following command in the terminal:

```bash
git clone https://github.com/jako4295/j2a.git
pip install -e j2a/
```

## <a name='Usage'></a>Usage

### <a name='GettingStarted'></a>Getting Started

To train (and then evaluate) a model you can follow the following steps:

1. Extract data from the databases using the data_handler module described in [Data Extractor](#DataExtractor). This is split into two commands, one for each dataset:

```bash
python j2a/data_handler/youtube_download.py --data_enum musiccaps --output <path/to/output/folder>
```

```bash
python j2a/data_handler/youtube_download.py --data_enum youtube_8m --output <path/to/output/folder>
```

2. Encode the data using the encoder module described in [Encoder](#Encoder). This is done with the following command:

```bash
python j2a/encoder/encode.py --audio_folder <path/to/data_folder> --output_folder <path/to/output/folder>
```

> [!NOTE]  
> Both the path to the data and the path to the encoded data is stored in the j2a/data_handler/.json_files/data_summary.json. This file is used for parsing all information to the train test split script. Hence, if the data is moved the paths needs to be updated. The procedure to change the paths is the same as when changing the path to a different encoding.
>
> If you implement a different encoder and want to use this in the training process, you need to update the `j2a/data_handler/.json_files/data_summary.json` with the path to the new encoded data. This can be done as follows:
>
> ```python
> from j2a.data_handler.json_handler import JsonInteractor
>
> inter = JsonInteractor() # loads the data_summary.json
> inter.clean_json_param("encoded_path")
> inter.update_json_paths("path/to/new/encoding", "encoded_path")
> inter.save(ask=False, silent=True)
> ```

3. Create train and test splits using the `train_test_split.py`. This script splits the data into 80% training data, 10% test data, and 10% validation data. The script can be run with the following command:

```bash
python j2a/data_handler/train_test_split.py --prompt_path <path/to/prompt_txt_file> --output_dir <path/to/output_dir>
```

This will create a train/test/validation split in a separate folder inside the output_dir.

4. A new model can be trained using the `j2a/train.py`. This file is run with the following arguments:

```bash
python j2a/train.py --train_path <path/to/train.csv> --eval_path <path/to/validation.csv> --save_path <path/to/save_location>
```

Note that the eval_path argument is optional. If it is desired to change more hyperparameters we refer to the `j2a/train.py` file. In the `save_path` a `model_info` will be created with relevant files.

Additional optional arguments are `--load_projector_path` (path to a pretrained projection path) and `--update_llm` (boolean - when true then the llm weights and biases are also updated).

5. Evaluating the data is currently set to handle J2A-2.0 and J2A-2.1 and the type should be specified since they have a different audio projection layer. The command for evaluating a J2A-2.0 type model is seen below

```bash
python j2a/evaluation/j2a_2x.py --projector_path <path/to/model.pth> --test_csv_path <path/to/test.csv> --model_name "j2a-2.0"
```

This will create a `pred.csv` with responses for all data in the `test.csv` and will be located in the same folder as the `projector_path`. If you have a trained llm then the argument `--llm_path` can be used to point to the folder with the language model.

### <a name='DataExtractor'></a>Data Extractor

There are currently two databases used in this project, [MusicCaps](https://google-research.github.io/seanet/musiclm/examples/) and [YouTube8M-MusicTextClips](https://zenodo.org/records/8040754). In the `data_handler` folder, there is a method to extract data from both databases.

The data is stored in the specified folder and an additional `.json` file is created to store metadata about the data. This metadata is stored in `j2a/data_handler/.json_files/data_summary.json`.

To extract the data create a Python environment and install the following packages:

```bash
pip install -r requirements_datagen.txt
```

> [!NOTE]  
> If you can't import j2a you need to run the `pip install -e j2a/`

Go to the `llm_tolls/data_handler` folder. To extract the data run

```bash
python <path/to/youtube_download.py> --data_enum <database> --output <path/to/output/folder>
```

where the `database` specify the database you want to extract data from (options are `musiccaps` and `youtube_8m`). For the output folder please use absolute path. An example of this is shown below:

```bash
python j2a/data_handler/youtube_download.py --data_enum musiccaps --output <path/to/output_folder>
```

This will create `j2a/data_handler/.json_files/data_summary.json` and store the data in the specified folder. Note that some files are age-restricted and will not be downloaded.

Once the data is extracted the `data_summary.json` file can be interacted with through the `JsonInteractor` class.

> [!WARNING]  
> If the sound files are moved the path in `j2a/data_handler/.json_files/data_summary.json` will be wrong and errors can occur if the paths are not updated in the file. How to update the path is show below.

#### <a name='JsonInteractor'></a>`JsonInteractor`

Base functionalities are:

- `add_feature`: Add a feature to elements in the `data_summary.json` file.
- `add_sound_path`: Add a sound path to elements in the `data_summary.json` file. This is automatically done when extracting data through the [youtube_download.py](j2a/data_handler/youtube_download.py).
- `get_info`: Get information about elements in the `data_summary.json` file.
- `save`: Save the changes to the `data_summary.json` file.

To get the names of the elements in the `data_summary.json` file use the `MetaDataEnum` class. An example of how to use the `MetaDataEnum` class and the `JsonInteractor` class is shown below:

```python
from j2a.data_handler.json_handler import JsonInteractor, MetaDataEnum

database_name = MetaDataEnum.musiccaps.value + "_0"

json_interactor = JsonInteractor()
json_interactor.get_info(database_name)
```

This will print the following information about "MusicCaps_0" from the `data_summary.json` file:

```text
+------------+-----------------------------------------------------------------+
|   Info:    |                           MusicCaps_0                           |
+------------+-----------------------------------------------------------------+
| Sound file |            /usr/local/path/to/data/MusicCaps_0.mp3              |
+------------+-----------------------------------------------------------------+
|  Metadata  | +--------------------------+----------------------------------+ |
|            | |           ytid           |           -0Gj8-vB1q4            | |
|            | +--------------------------+----------------------------------+ |
|            | |         start_s          |                30                | |
|            | +--------------------------+----------------------------------+ |
|            | |          end_s           |                40                | |
|            | +--------------------------+----------------------------------+ |
|            | | audioset_positive_labels |   /m/0140xf,/m/02cjck,/m/04rlf   | |
|            | +--------------------------+----------------------------------+ |
|            | |       aspect_list        | ['low quality', 'sustained strin | |
|            | |                          | gs melody', 'soft female vocal', | |
|            | |                          |  'mellow piano melody', 'sad', ' | |
|            | |                          |       soulful', 'ballad']        | |
|            | +--------------------------+----------------------------------+ |
|            | |         caption          | The low quality recording featur | |
|            | |                          | es a ballad song that contains s | |
|            | |                          | ustained strings, mellow piano m | |
|            | |                          | elody and soft female vocal sing | |
|            | |                          | ing over it. It sounds sad and s | |
|            | |                          | oulful, like something you would | |
|            | |                          |     hear at Sunday services.     | |
|            | +--------------------------+----------------------------------+ |
|            | |        author_id         |                4                 | |
|            | +--------------------------+----------------------------------+ |
|            | |    is_balanced_subset    |                0                 | |
|            | +--------------------------+----------------------------------+ |
|            | |     is_audioset_eval     |                1                 | |
|            | +--------------------------+----------------------------------+ |
+------------+-----------------------------------------------------------------+
```

If we move the data to a new folder and also want to add other features to the `data_summary.json` file we can use the `add_feature` method. An example of this is shown below:

```python
json_interactor.add_sound_path(
    database_name, "/usr/local/new_path/to/data"
)
json_interactor.add_feature([database_name], [{"bpm": 60, "duration": 10}])
json_interactor.get_info(database_name)
```

The `get_info` method will now print the following information about "MusicCaps_0" from the `json_interactor` object file:

```text
+------------+-----------------------------------------------------------------+
|   Info:    |                           MusicCaps_0                           |
+------------+-----------------------------------------------------------------+
| Sound file |           /usr/local/new_path/to/data/MusicCaps_0.mp3           |
+------------+-----------------------------------------------------------------+
|  Metadata  | +--------------------------+----------------------------------+ |
|            | |           ytid           |           -0Gj8-vB1q4            | |
|            | +--------------------------+----------------------------------+ |
|            | |         start_s          |                30                | |
|            | +--------------------------+----------------------------------+ |
|            | |          end_s           |                40                | |
|            | +--------------------------+----------------------------------+ |
|            | | audioset_positive_labels |   /m/0140xf,/m/02cjck,/m/04rlf   | |
|            | +--------------------------+----------------------------------+ |
|            | |       aspect_list        | ['low quality', 'sustained strin | |
|            | |                          | gs melody', 'soft female vocal', | |
|            | |                          |  'mellow piano melody', 'sad', ' | |
|            | |                          |       soulful', 'ballad']        | |
|            | +--------------------------+----------------------------------+ |
|            | |         caption          | The low quality recording featur | |
|            | |                          | es a ballad song that contains s | |
|            | |                          | ustained strings, mellow piano m | |
|            | |                          | elody and soft female vocal sing | |
|            | |                          | ing over it. It sounds sad and s | |
|            | |                          | oulful, like something you would | |
|            | |                          |     hear at Sunday services.     | |
|            | +--------------------------+----------------------------------+ |
|            | |        author_id         |                4                 | |
|            | +--------------------------+----------------------------------+ |
|            | |    is_balanced_subset    |                0                 | |
|            | +--------------------------+----------------------------------+ |
|            | |     is_audioset_eval     |                1                 | |
|            | +--------------------------+----------------------------------+ |
+------------+-----------------------------------------------------------------+
|  Features  |                        +----------+----+                        |
|            |                        |   bpm    | 60 |                        |
|            |                        +----------+----+                        |
|            |                        | duration | 10 |                        |
|            |                        +----------+----+                        |
+------------+-----------------------------------------------------------------+
```

> [!IMPORTANT]  
> If you want to save the changes made to `j2a/data_handler/.json_files/data_summary.json` you have to use the `save` method.

To save the changes to the `data_summary.json` file use the `save` method. If there are any changes to the files this will be printed to the console. An example of this is shown below:

```python
json_interactor.save()
```

This will print the following to the console:

```bash
('change', 'MusicCaps_0.sound_path', ('/usr/local/path/to/data/MusicCaps_0.mp3', '/usr/local/new_path/to/data/MusicCaps_0.mp3'))
('add', 'MusicCaps_0.features', [('bpm', 60), ('duration', 10)])
Are you sure you want to make the changes to the data_summary.json? [y/n]
```

If you are sure you want to make the changes to the `data_summary.json` file type `y` and press enter. If you are not sure type `n` and press enter - in this case, the changes will still be on the `json_interactor` object, but will not be saved to the file. To discard all changes to the `json_interactor` just create a new `JsonInteractor` object.

### <a name='FeatureExtractor'></a>Feature Extractor

Features, such as key and tempo, can be extracted from a song by using this module. To extract every supported feature you should access the `extract_all` function found in the [extract_all.py](j2a/feature_extractor/extract_all.py) file. Below is an example of how this is done.

```Python
from j2a.feature_extractor.extract_all import extract_all

path = "data/youtube/MusicCaps-8.mp3"
features = extract_all(path)
```

This returns a typed dictionary, making it easy to see which features will be extracted.

Arguments can be passed to the `extract_all` function to enable/disable `cuda` functionality, or change the model used for extracting the genre of a song. Make sure you disable `cuda` if this is not available on your system, since it is enabled by default. All models available to the genre extractor are HuggingFace models, and an `Enum` class that lists these is defined in [this file](j2a/feature_extractor/genre_classifier.py).

You could use the individual extractors, but this is not advised since the extractors using [madmom](https://github.com/CPJKU/madmom) do not produce a predictable output.

### <a name='Encoder'></a>Encoder

The encoder used is based on [Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners](https://arxiv.org/abs/2306.00561). To encode all data within a folder (only works with `.wav` files) the `j2a.encoder.encode.py` can be run with the following arguments:

```bash
python j2a/encoder/encode.py --audio_folder <path/to/data_folder> --output_folder <path/to/output_folder>
```

This will encode all `.wav` files in the `audio_path` folder and store the encoded data in the `output_path` folder. The encoded data is stored in `.pt` files (this is the format used for torch tensors, so it can be loaded with `torch.load()`). By default, the encoding will create encoding to the `.json` file described in [Data Extractor](#DataExtractor). However, if you want to encode the data without the `.json` file being updated then `--save_to_json` can be set to `False`.

If the weights and biases from the `mwmae-jax-official` encoder are not already downloaded, then this is done automatically. The weights and biases are stored in `j2a/encoder/.weights_biases`.

> [!IMPORTANT]  
> If there are issues with the submodule `mwmae-jax-official` remember to use the `git pull --recurse-submodules` command when pulling the repository. If the submodule is not downloaded, then use the `git submodule update --init --recursive` command.
