"""Official text to native input IDs; no annotation fields enter the method."""

import json
from pathlib import Path

from . import data
from myproject.inputs import encode_response, encode_source


def source_span(source: dict) -> tuple[int, int]:
    prompt = source["prompt"]
    if source["task_type"] == "QA":
        start = prompt.index("passages:\n") + len("passages:\n")
        stops = [prompt.find(marker, start) for marker in ("\nIn case the passages", "\noutput:")]
        return start, min(position for position in stops if position >= 0)
    text = str(source["source_info"])
    if prompt.count(text) != 1:
        raise ValueError("The official source must occur exactly once in the prompt")
    start = prompt.index(text)
    return start, start + len(text)


def prepare_official(dataset: Path, model: Path, output: Path) -> None:
    """Prepare all three tasks from official text without using annotation labels."""
    from transformers import AutoTokenizer

    with (dataset / "source_info.jsonl").open() as stream:
        sources = {str(row["source_id"]): row for row in map(json.loads, stream)}
    with (dataset / "response.jsonl").open() as stream:
        rows = [{key: row[key] for key in ("id", "source_id", "response", "split", "model")}
                for row in map(json.loads, stream)]
    rows.sort(key=lambda row: (sources[str(row["source_id"])]["task_type"], row["split"],
                               row["model"], str(row["id"])))
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    source_files, records = {}, []
    for index, row in enumerate(rows):
        source_id = str(row["source_id"])
        if source_id not in source_files:
            source_files[source_id] = f"sources/{len(source_files):06d}.json"
            source = sources[source_id]
            start, stop = source_span(source)
            data.write_json(output / source_files[source_id],
                            encode_source(tokenizer, source["prompt"], source["prompt"][start:stop]))
        response = encode_response(tokenizer, row["response"])
        directory = f"responses/{index:06d}"
        (output / directory).mkdir(parents=True)
        data.write_json(output / directory / "response.json", response)
        records.append({
            "id": str(row["id"]), "source_id": source_id, "task": sources[source_id]["task_type"],
            "split": row["split"], "generator": row["model"], "directory": directory,
            "source_file": source_files[source_id], "tokens": len(response["answer_ids"]),
        })
    data.write_json(output / "manifest.json", {
        "dataset": str(dataset.resolve()), "model": str(model.resolve()),
        "response_root": str(output.resolve()), "records": data.partition_records(records),
    })
