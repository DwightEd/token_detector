"""Export official HalluRAG pickle rows to the dataset adapter's JSONL format."""

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from . import data


class PlainUnpickler(pickle.Unpickler):
    """Released text/label rows require no executable pickle globals."""

    def find_class(self, module, name):
        raise ValueError(f"Unexpected pickle global: {module}.{name}")


def export_hallurag(files: list[Path], output: Path) -> dict:
    """Remove internal_states; preserve native text, targets and original IDs.

    Supply every official member of one split. No generator filtering, label
    projection or sentence alignment repair is performed at this stage.
    """
    paths = sorted(Path(path).resolve() for path in files)
    if not paths or len({path.name for path in paths}) != len(paths):
        raise ValueError("Provide distinct official pickle members")
    splits = {path.name.split("_", 1)[0] for path in paths}
    if len(splits) != 1 or not splits <= {"train", "val", "test"}:
        raise ValueError("Export one official train, val or test split at a time")
    split = next(iter(splits))
    output = Path(output)
    receipt = output.with_suffix(".receipt.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Export output or receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    members, targets = [], Counter()
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                            prefix=f".{output.name}.", suffix=".tmp") as stream:
        for path in paths:
            digest = data.file_hash(path)
            with path.open("rb") as source:
                rows = PlainUnpickler(source).load()
            for index, row in enumerate(rows):
                record = {**row, "id": f"hallurag:{path.name}:{index}",
                          "original_file": path.name, "original_index": index,
                          "original_split": split,
                          "sentence_data": [{key: value for key, value in sentence.items()
                                             if key != "internal_states"}
                                            for sentence in row["sentence_data"]]}
                targets.update(sentence["target"] for sentence in record["sentence_data"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            members.append({"path": str(path), "sha256": digest, "responses": len(rows)})
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, output)
    report = {"split": split, "members": members,
              "responses": sum(member["responses"] for member in members),
              "sentence_targets": dict(targets), "output_sha256": data.file_hash(output),
              "policy": "Only internal_states removed; native labels and text preserved"}
    data.write_json(receipt, report)
    return report
