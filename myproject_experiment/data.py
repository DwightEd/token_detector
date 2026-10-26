"""Optional dataset splits, compact experiment storage and file seals."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

TASKS = ("QA", "Summary", "Data2txt")
FIELDS = ("delta", "route", "target", "token_id", "answer_index", "unit_index", "development")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file atomically, without replacing existing results."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pack(path: Path) -> dict:
    """Read only the seven numerical fields required by the frozen R4 method."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in FIELDS}


def select_rows(pack: dict, partition: str) -> dict:
    if partition == "test":
        return pack
    mask = pack["development"] if partition == "dev" else ~pack["development"]
    return {name: values[mask] for name, values in pack.items()}


def partition_records(records: list[dict]) -> list[dict]:
    """Keep all generators of a source together using the historical SHA42 split."""
    development = {}
    for task in TASKS:
        training = {str(row["source_id"]) for row in records
                    if row["task"] == task and row["split"] == "train"}
        ordered = sorted(training, key=lambda source: hashlib.sha256(f"42:{source}".encode()).hexdigest())
        development[task] = set(ordered[:int(np.ceil(.2 * len(ordered)))])
    return [dict(row, partition="test" if row["split"] == "test" else
                 "dev" if str(row["source_id"]) in development[row["task"]] else "fit")
            for row in records]


def prepare(cache: Path, output: Path) -> None:
    """Index existing source/response inputs without copying any raw artifacts."""
    cache = cache.resolve()
    manifest = read_json(cache / "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "dataset": manifest["dataset"], "model": manifest["model"],
        "response_root": str(cache), "records": partition_records(manifest["records"]),
    })


def read_response(metadata: dict, record: dict) -> dict:
    return read_json(Path(metadata["response_root"]) / record["directory"] / "response.json")


def pack_path(root: Path, task: str, partition: str) -> Path:
    split = "test" if partition == "test" else "train"
    return Path(root) / f"{task}_{split}.npz"


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prediction_path(directory: Path, sample_id: str) -> Path:
    """IDs stay in the record; hashed filenames also support arbitrary native IDs."""
    return directory / f"{text_hash(sample_id)}.json"
