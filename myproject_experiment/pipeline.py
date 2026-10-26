"""Optional experiment file orchestration, separate from the callable detector."""

from pathlib import Path

import numpy as np

from . import data
from myproject.detector import Detector


def fit(root: Path, task: str, output: Path) -> None:
    pack = data.load_pack(data.pack_path(root, task, "fit"))
    output.parent.mkdir(parents=True, exist_ok=True)
    Detector(task).fit(data.select_rows(pack, "fit")).save(output)


def score(root: Path, task: str, partition: str, model: Path, output: Path) -> None:
    path = data.pack_path(root, task, partition)
    metadata = data.read_json(path.with_suffix(".json"))
    detector = Detector.load(model)
    if detector.task != task:
        raise ValueError("Reference task differs from the requested scoring task")
    pack = data.select_rows(data.load_pack(path), partition)
    values = {"score": detector.score(pack)}
    values.update({name: pack[name] for name in ("target", "token_id", "answer_index")})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scores.npz", **values)
    data.write_json(output / "metadata.json", {
        "task": task, "partition": partition, "dataset": metadata["dataset"],
        "response_root": metadata["response_root"], "records": metadata["records"],
        "model_sha256": data.file_hash(model), "pack_sha256": data.file_hash(path),
    })
    data.write_json(output / "seal.json", {
        name: data.file_hash(output / name) for name in ("scores.npz", "metadata.json")
    })


def predict_dataset(prepared: Path, model: Path, task: str, output: Path,
                    references: Path | None = None, device: str = "cuda:0") -> None:
    """Score prepared text inputs, checkpoint each response, then seal predictions.

    The bundled R4 references are reused unless explicitly supplied. Annotation
    files are never opened here. Resuming requires identical inputs and code.
    """
    import json

    from myproject.api import HallucinationDetector
    from .datasets import read_samples

    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    samples = list(read_samples(prepared / "inputs.jsonl"))
    if not samples:
        raise ValueError("The prepared split is empty")
    for sample in samples:
        _ = sample.inputs  # Validate text-evidence support before loading a model.
        if sample.task != task:
            raise ValueError("Select one recipe/task per prediction run")
    reference = (Path(references).resolve() if references else
                 Path(__file__).parents[1] / "myproject/references" / f"{task}.npz")
    manifest = data.read_json(prepared / "manifest.json")
    if (data.file_hash(prepared / "inputs.jsonl") != manifest["inputs_sha256"]
            or len(samples) != manifest["samples"]):
        raise ValueError("Prepared inputs differ from their dataset manifest")
    code = [*Path(__file__).parents[1].joinpath("myproject").glob("*.py"),
            *(Path(__file__).with_name(name) for name in
              ("pipeline.py", "datasets.py", "data.py", "inputs.py"))]
    model_files = sorted({path for pattern in
                          ("*.json", "*.safetensors", "*.bin", "tokenizer.model", "merges.txt")
                          for path in Path(model).glob(pattern) if path.is_file()})
    config = {
        "input": str(prepared / "inputs.jsonl"),
        "input_sha256": data.file_hash(prepared / "inputs.jsonl"),
        "manifest_sha256": data.file_hash(prepared / "manifest.json"),
        "model": str(Path(model).resolve()), "task": task, "device": device,
        "model_files": {str(path.resolve()): data.file_hash(path) for path in model_files},
        "references": str(reference), "reference_sha256": data.file_hash(reference),
        "code": {str(path.resolve()): data.file_hash(path) for path in sorted(code)},
    }
    output.mkdir(parents=True, exist_ok=True)
    for relative in (".run.json.tmp", ".complete.json.tmp", "predictions/.seal.json.tmp"):
        (output / relative).unlink(missing_ok=True)
    run_path = output / "run.json"
    if run_path.exists():
        if data.read_json(run_path) != config:
            raise ValueError("Resume inputs, model, references or code differ")
    else:
        if any(output.iterdir()):
            raise ValueError("A new prediction directory must be empty")
        data.write_json(run_path, config)
    cases = output / "predictions"
    cases.mkdir(exist_ok=True)
    if (cases / "seal.json").exists():
        _verify_predictions(cases)
        expected = {data.prediction_path(cases, sample.id).name for sample in samples}
        if set(data.read_json(cases / "seal.json")) != expected:
            raise ValueError("Sealed predictions differ from prepared sample IDs")
        for sample in samples:
            _validate_prediction(data.read_json(data.prediction_path(cases, sample.id)), sample)
        _complete_predictions(output, len(samples))
        return
    detector = None
    for index, sample in enumerate(samples, 1):
        destination = data.prediction_path(cases, sample.id)
        identity = {"id": sample.id, "source_id": sample.source_id,
                    "response_sha256": data.text_hash(sample.response)}
        if destination.exists():
            _validate_prediction(data.read_json(destination), sample)
            continue
        if detector is None:
            detector = HallucinationDetector.from_pretrained(
                model, task=task, references=reference, device=device,
            )
        tokens = detector.predict(**sample.inputs)
        prediction = {**identity, "tokens": tokens}
        _validate_prediction(prediction, sample)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(prediction, stream, ensure_ascii=False,
                      allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
        if index % 25 == 0 or index == len(samples):
            print(f"predicted {index}/{len(samples)}", flush=True)
    expected = {data.prediction_path(cases, sample.id).name for sample in samples}
    if {path.name for path in cases.iterdir()} != expected:
        raise ValueError("Prediction directory has missing or unexpected files")
    if data.read_json(run_path) != config:
        raise ValueError("Run metadata changed during inference")
    bindings = {**config["code"], **config["model_files"],
                config["input"]: config["input_sha256"],
                str(prepared / "manifest.json"): config["manifest_sha256"],
                config["references"]: config["reference_sha256"]}
    for path, expected_hash in bindings.items():
        if data.file_hash(Path(path)) != expected_hash:
            raise ValueError("Inference inputs, references or code changed during the run")
    data.write_json(cases / "seal.json", {
        path.name: data.file_hash(path) for path in sorted(cases.iterdir())
    })
    _complete_predictions(output, len(samples))


def _complete_predictions(output: Path, samples: int) -> None:
    complete = {"samples": samples,
                "prediction_seal_sha256": data.file_hash(output / "predictions/seal.json"),
                "run_sha256": data.file_hash(output / "run.json"), "annotation_reads": 0}
    path = output / "complete.json"
    if path.exists():
        if data.read_json(path) != complete:
            raise ValueError("Completion metadata differs")
    else:
        data.write_json(path, complete)


def _validate_prediction(prediction: dict, sample) -> list[dict]:
    """Check saved token identities and offsets without reading any labels."""
    import math

    identity = {"id": sample.id, "source_id": sample.source_id,
                "response_sha256": data.text_hash(sample.response)}
    if any(prediction.get(key) != value for key, value in identity.items()):
        raise ValueError("Prediction belongs to different input")
    tokens = prediction["tokens"]
    if not tokens:
        raise ValueError("A prediction contains no observed tokens")
    previous = -1
    for row in tokens:
        start, end = row["start"], row["end"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(sample.response) or start < previous
                or type(row["token_id"]) is not int or row["token_id"] < 0
                or row["text"] != sample.response[start:end]
                or not math.isfinite(row["score"])):
            raise ValueError("Prediction token identity, offsets or score is invalid")
        previous = start
    return tokens

def _verify_predictions(directory: Path) -> None:
    """Verify the exact completed prediction file set before any evaluation."""
    sealed = data.read_json(directory / "seal.json")
    if {path.name for path in directory.iterdir()} != {*sealed, "seal.json"}:
        raise ValueError("The sealed prediction file set differs")
    for name, expected in sealed.items():
        if data.file_hash(directory / name) != expected:
            raise ValueError(f"Sealed prediction changed: {name}")
