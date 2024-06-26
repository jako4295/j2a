import os
from argparse import ArgumentParser

import pandas as pd  # type: ignore
import torch  # type: ignore
from evaluate import load as load_evaluation_model  # type: ignore
from tqdm import tqdm  # type: ignore

from j2a.model import AudioProjector, AudioProjectorNoPool, Model, load_llm
from j2a.model_interactor import ModelInteractor

# csv structure
# name      , prompt              , encoding        , metadata , features, label
# name in db, prompt used to train, path to encoding, List[str], ........, ground truth


def load_interactor(
    projector_path: str,
    llm_path: str | None,
    model_id: str = "Open-Orca/Mistral-7B-OpenOrca",
    device: str = "cuda",
    model_name: str = "j2a-2.1",
) -> ModelInteractor:
    audio_projector: AudioProjector | AudioProjectorNoPool
    if model_name == "j2a-2.1":
        audio_projector = AudioProjector()
    elif model_name == "j2a-2.0":
        audio_projector = AudioProjectorNoPool()
    else:
        raise ValueError("Invalid model name")
    audio_projector.to(device)

    _, llm = load_llm(model_id=model_id)

    model = Model(audio_projector.to(torch.bfloat16), llm, False)
    model_interactor = ModelInteractor(
        model, projector_path=projector_path, llm_path=llm_path, device=device
    )

    return model_interactor


def load_test_csv(test_csv_path: str) -> pd.DataFrame:
    test_csv = pd.read_csv(test_csv_path)
    test_csv = test_csv[["name", "prompt", "encoding", "label"]]

    return test_csv


def generate_predictions(
    model_interactor: ModelInteractor, test_csv: pd.DataFrame
) -> pd.DataFrame:
    result_dict = dict()

    for i, (idx, row) in tqdm(enumerate(test_csv.iterrows())):
        sample = model_interactor.get_sample(
            wav_file=row["encoding"], prompt=row["prompt"]
        )
        result_dict[idx] = sample

    result_series = pd.Series(result_dict)

    test_and_predictions = test_csv.copy()
    test_and_predictions["prediction"] = result_series

    return test_and_predictions


def score(tests_and_predictions: pd.DataFrame):
    bert_scorer = load_evaluation_model("bertscore")
    candidates = tests_and_predictions["prediction"].to_list()
    references = tests_and_predictions["label"].to_list()

    results = bert_scorer.compute(
        predictions=candidates,
        references=references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        rescale_with_baseline=True,
    )

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--projector_path",
        type=str,
        required=True,
        help="Path to the weights of a projector (.pth).",
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        required=False,
        default=None,
        help="Path to the llm weights (path to folder).",
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        required=True,
        help="Path to the test csv file (this refers to the test data from train test split)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        default="Open-Orca/Mistral-7B-OpenOrca",
        help="Model id to be used for predictions",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="j2a-2.1",
        help="Model name to be used for predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to be used for predictions",
    )
    args = parser.parse_args()

    model_interactor = load_interactor(
        projector_path=args.projector_path,
        llm_path=args.llm_path,
        model_id=args.model_id,
        device=args.device,
        model_name=args.model_name,
    )
    test_csv = load_test_csv(args.test_csv_path)
    test_and_predictions = generate_predictions(
        model_interactor=model_interactor, test_csv=test_csv
    )

    scores = score(tests_and_predictions=test_and_predictions)
    for key, val in scores.items():
        if key == "hashcode":
            continue
        test_and_predictions["bert_" + key] = val

    path_to_pred = os.path.dirname(args.projector_path) + "/pred.csv"
    test_and_predictions.to_csv(path_to_pred)


if __name__ == "__main__":
    main()
