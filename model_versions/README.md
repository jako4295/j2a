# Model versions
This directory contains the model versions that are used in the project. 

## <a name='TableofContents'></a>Table of Contents
<!-- vscode-markdown-toc -->
- [Databases](#databases)
- [J2A-1.0](#j2a-10)
	- [Extract Features](#extract-features)
	- [Inputs and Outputs](#inputs-and-outputs)
- [J2A-2.x](#j2a-2x)
	- [ Tokenizing the Music for the LLM](#-tokenizing-the-music-for-the-llm)
	- [Training and Inference](#training-and-inference)
	- [J2A-2.0](#j2a-20)
	- [J2A-2.1](#j2a-21)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Databases'></a>Databases
The databases used for this model are [MusicCaps](https://google-research.github.io/seanet/musiclm/examples/) with 5520 songs and [YouTube8M-MusicTextClips](https://zenodo.org/records/8040754) with 4168 songs. 

Both of the databases have a text description of each of the sound files, while only MusicCaps also has miscellaneous metadata. Furthermore, all metadata is based on only 10 seconds of each of the YouTube videos. When importing the data about 7% is discarded due to unavailable videos, private videos, or age restrictions on the videos. 

The text descriptions from the databases are used as labels when training the model.

## <a name='J2A-1.0'></a>J2A-1.0
![j2a-1-0.png](images/j2a-1-0.png)

The J2A-LLM is a [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) model and is fine-tuned on the [MusicCaps](https://google-research.github.io/seanet/musiclm/examples/) and [YouTube8M-MusicTextClips](https://zenodo.org/records/8040754) databases with the prompt mentioned above.

### <a name='ExtractFeatures'></a>Extract Features
We generate features for each song using two different frameworks. The first is inspired by the implementation from [Llark](https://github.com/spotify-research/llark). These features are calculated using [madmom](https://github.com/CPJKU/madmom) and include:
- BPM
- Key
- Downbeats
- Chords

The other framework is a huggingface model called [genre-recognizer-finetuned-gtzan_dset](https://huggingface.co/pedromatias97/genre-recognizer-finetuned-gtzan_dset) which is used to predict the genre of the song.

### <a name='InputsandOutputs'></a>Inputs and Outputs
J2A-LLM is a text-to-text model that takes a text prompt as input and returns a text description of the sound as output. The J2A-1.0 is an audio-to-text model taking an audio file as input, calculating features and using J2A-LLM to generate a text description of the sound as output. 

## <a name='J2A-2.x'></a>J2A-2.x
The class of models which fit in the J2A-2.x architecture follow this general structure
![J2A-2.x model](images/tmp-J2A2.0.drawio.png)
What differs in these models are the audio encoder, and how the audio projector is implemented.

### <a name='Tokenizing the Music for the LLM'></a> Tokenizing the Music for the LLM
The biggest difference between the two larger versions of the models is the way the music information is passed to the LLM. In this new architecture we encode the music using the MW-MAE audio encoder. This encoding is then passed to an audio projection model, which maps this encoding to an embedding the model can understand. These embeddings are concatenated with the other embeddings generated from the prompt more traditionally, which is then passed to the model as one large tensor.

### <a name='Training and Inference'></a>Training and Inference
In all of these models we only train on the audio projector itself, letting all other parameters of the model be frozen. <br>
Training is done by tokenizing and embedding the ground truth description as well. When doing these embeddings we get the logits of the tokens for both the model generated output and the ground truth. These logits are then used to compute the loss via cross entropy.

In order to run inference one can use the tokenizer associated with the LLM to decode the output tensor, in order to get the text description of the piece of music.

### <a name='J2A-2.0'></a>J2A-2.0
- Projector
  - The audio projector consists of a linear normalization layer, and a normalization layer
- Audio encoder
  - We take the mean over dimension 1, which reduced the number of dimensions of the output of the encoder

### <a name='J2A-2.1'></a>J2A-2.1
- Projector
  - The audio projector works similarly as it did in [J2A-2.0](#j2a-20), but we add a pooling layer, as was done in the cookbook which we're closely following
- Audio Encoder
  - We do not take the mean in this version, giving a passing a much higher resolution tensor to the audio projection layer, hence the pooling.