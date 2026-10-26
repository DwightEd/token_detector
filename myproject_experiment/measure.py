"""Four source worlds and frozen-cache measurement on the original token axis."""

from pathlib import Path

import numpy as np

from . import data
from myproject.measure import answer_rows


def cached_measurement(root: Path, record: dict) -> dict:
    """Reuse primitive source effects/route; never load labels or old detector code."""
    path = root / record["directory"] / "observations.npz"
    with np.load(path, allow_pickle=False) as saved:
        return {
            "delta": np.column_stack((saved["source_local"], saved["source_full"])),
            "route": saved["raw_route"], "token_id": saved["token_id"],
        }


def measure(input_root: Path, output: Path, task: str, split: str,
            native: bool = False, device: str = "cuda:0") -> None:
    """Create one compact task/split pack; native=True replays the frozen observer."""
    manifest = data.read_json(input_root / "manifest.json")
    records = [row for row in manifest["records"] if row["task"] == task and row["split"] == split]
    root = Path(manifest["response_root"])
    observer = None
    if native:
        from myproject.model import load_model
        from myproject.observer import Observer

        observer = Observer(load_model(manifest["model"], device))
    batches, packed_records, cursor = [], [], 0
    for index, record in enumerate(records):
        response = data.read_response(manifest, record)
        if native:
            source = data.read_json(root / record["source_file"])
            observed = observer.measure(source, response)
        else:
            observed = cached_measurement(root, record)
        rows = answer_rows(observed, response, index)
        rows["development"] = np.full(len(rows["target"]), record["partition"] == "dev", dtype=bool)
        count = len(rows["target"])
        batches.append(rows)
        packed_records.append(dict(record, packed_start=cursor, packed_stop=cursor + count))
        cursor += count
        if (index + 1) % 300 == 0:
            print(f"measure {task}/{split} {index + 1}/{len(records)}", flush=True)
    packed = {name: np.concatenate([row[name] for row in batches]) for name in data.FIELDS}
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{task}_{split}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **packed)
    data.write_json(path.with_suffix(".json"), {
        "task": task, "split": split, "dataset": manifest["dataset"],
        "response_root": str(root), "records": packed_records,
        "measurement": "native observer replay" if native else "frozen native observations",
    })
