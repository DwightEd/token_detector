"""Dataset adapters with separate inference inputs and native annotations."""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from . import data
from .inputs import source_span


@dataclass(frozen=True)
class Sample:
    """An existing response and the evidence actually available when it was made."""

    id: str
    source_id: str
    split: str
    task: str
    prompt: str
    source: str
    response: str
    generator: str = ""
    evidence_path: str = ""

    @property
    def inputs(self) -> dict[str, str]:
        """Return the text detector's input, rejecting unsupported structured evidence."""
        if not all(isinstance(value, str) for value in (self.prompt, self.source, self.response)):
            raise ValueError(f"Sample {self.id} prompt, source and response must be strings")
        if not self.source:
            raise ValueError(
                f"Sample {self.id} has structured evidence; the text detector requires "
                "a source passage. Database claims cannot be scored without an explicit "
                "structured-evidence method."
            )
        if self.prompt.count(self.source) != 1:
            raise ValueError(f"Sample {self.id} source must occur exactly once in its prompt")
        return {"prompt": self.prompt, "source": self.source, "response": self.response}


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number} must contain a JSON object")
                yield row


def read_samples(path: Path) -> Iterator[Sample]:
    """Read prepared, inference-only rows and reject ambiguous sample joins."""
    seen = set()
    for row in read_jsonl(path):
        sample = Sample(**row)
        if not isinstance(sample.id, str) or not sample.id or sample.id in seen:
            raise ValueError(f"Duplicate or empty sample ID: {sample.id!r}")
        seen.add(sample.id)
        yield sample


def _annotation(sample: Sample, granularity: str, **fields) -> dict:
    return {"id": sample.id, "response_sha256": data.text_hash(sample.response),
            "granularity": granularity, **fields}


def _check_spans(response: str, spans: list[dict]) -> None:
    for span in spans:
        start, end = span["start"], span["end"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(response)):
            raise ValueError(f"Invalid character span: {span}")


def sentence_token_labels(annotation: dict, tokens: list[dict]) -> list[dict]:
    """Broadcast sentence labels to observed tokens, without reading token scores.

    Any positive overlap wins. Otherwise unknown or uncovered characters are
    ignored; a negative token must be fully covered by known negative sentences.
    Duplicate BPE offsets remain separate observations on the original token axis.
    """
    if annotation["granularity"] != "sentence":
        raise ValueError("Sentence projection requires native sentence annotations")
    if annotation["coverage"] == "unalignable":
        return [{"label": None, "sentence_indices": [], "exclusion": "unalignable"}
                for _ in tokens]
    spans = annotation["spans"]
    previous_end = 0
    for span in spans:
        start, end, label = span["start"], span["end"], span["label"]
        if (type(start) is not int or type(end) is not int
                or not previous_end <= start < end
                or (label is not None and (type(label) is not int or label not in (0, 1)))):
            raise ValueError("Invalid native sentence label, offset or ordering")
        previous_end = end
    projected = []
    for token in tokens:
        start, end = token["start"], token["end"]
        indices = [index for index, span in enumerate(spans)
                   if start < span["end"] and end > span["start"]]
        overlaps = [spans[index] for index in indices]
        covered = sum(min(end, span["end"]) - max(start, span["start"])
                      for span in overlaps)
        if any(span["label"] == 1 for span in overlaps):
            label, exclusion = True, None
        elif any(span["label"] is None for span in overlaps):
            label, exclusion = None, "undefined"
        elif covered != end - start:
            label, exclusion = None, "uncovered"
        else:
            label, exclusion = False, None
        projected.append({"label": label, "sentence_indices": indices,
                          "exclusion": exclusion})
    return projected


def _ragtruth(path: Path, split: str, task: str | None):
    if split not in {"train", "test"}:
        raise ValueError("RAGTruth split must be train or test")
    sources = {}
    for row in read_jsonl(path / "source_info.jsonl"):
        source_id = str(row["source_id"])
        if source_id in sources:
            raise ValueError(f"Duplicate RAGTruth source ID: {source_id}")
        sources[source_id] = row
    for row in read_jsonl(path / "response.jsonl"):
        source_id = str(row["source_id"])
        source = sources[source_id]
        if row["split"] != split or (task is not None and source["task_type"] != task):
            continue
        start, end = source_span(source)
        sample = Sample(str(row["id"]), source_id, split, source["task_type"],
                        source["prompt"], source["prompt"][start:end], row["response"],
                        row["model"])
        spans = [{"start": item["start"], "end": item["end"], "label": 1}
                 for item in row["labels"]]
        _check_spans(sample.response, spans)
        yield sample, _annotation(sample, "token", spans=spans, coverage="complete")


def _claimdb(path: Path, split: str, task: str | None):
    if split not in {"public", "private"}:
        raise ValueError("ClaimDB split must be public or private; private labels are hidden")
    if task is not None and task != "ClaimVerification":
        raise ValueError("ClaimDB task is ClaimVerification")
    for row in read_jsonl(path / f"test-{split}.jsonl"):
        database = row["db_name"]
        evidence = path / "test_dbs" / f"{database}.sqlite"
        prompt = "Verify the claim using the database."
        if row.get("extra_info"):
            prompt += "\nAdditional information: " + row["extra_info"]
        sample = Sample(str(row["claim_id"]), database, split, "ClaimVerification",
                        prompt, "", row["claim"],
                        evidence_path=str(evidence.resolve()) if evidence.exists() else "")
        native_label = row.get("label")
        if native_label is not None and native_label not in {
            "ENTAILED", "CONTRADICTED", "NOT ENOUGH INFO",
        }:
            raise ValueError(f"Unknown ClaimDB label: {native_label!r}")
        yield sample, _annotation(sample, "claim", native_label=native_label,
                                  coverage="complete" if native_label is not None else "hidden")


def _hallurag_source(prompt: dict) -> tuple[str, str]:
    """Extract retrieved chunks from released messages, never the oracle passage."""
    messages = prompt["rag_prompt"]
    users = [row["content"] for row in messages if row["role"] == "user"]
    if len(users) != 1:
        raise ValueError("HalluRAG must have exactly one user message")
    user = users[0]
    template = prompt["prompt_template_name"]
    if template == "template_langchain_hub":
        marker, suffix = "\nContext: ", " \nAnswer:"
        start = user.index(marker) + len(marker)
        end = user.rindex(suffix)
    elif template in {"template_1", "template_2"}:
        prefix = ("Your knowledge is limited to only this information:\n"
                  if template == "template_1" else
                  "Only use the information included in these chunks to answer the question:\n")
        if not user.startswith(prefix):
            raise ValueError(f"Unexpected HalluRAG {template} prompt")
        start, end = len(prefix), user.rindex("\nQUESTION: ")
    else:
        raise ValueError(f"Unknown HalluRAG prompt template: {template}")
    source = user[start:end]
    text = "\n\n".join(row["content"] for row in messages)
    if not source.strip() or text.count(source) != 1:
        raise ValueError("HalluRAG retrieved context must occur once in its prompt")
    return text, source


def _hallurag(path: Path, split: str, task: str | None):
    if split not in {"train", "val", "test"}:
        raise ValueError("HalluRAG split must be train, val or test")
    if task is not None and task != "QA":
        raise ValueError("HalluRAG task is QA")
    for row in read_jsonl(path / f"{split}.jsonl"):
        if row["original_split"] != split:
            raise ValueError("HalluRAG export contains a different official split")
        prompt, source = _hallurag_source(row["prompt"])
        quantization = row["quantization"]
        generator = row["model"] + (f" ({quantization})" if quantization else "")
        sample = Sample(row["id"], str(row["prompt"]["useful_art_i"]), split, "QA",
                        prompt, source, row["llm_response"], generator)
        spans, previous, alignment_error = [], "", ""
        for sentence in row["sentence_data"]:
            cumulative = sentence["cum_sentence"]
            if (not cumulative.startswith(previous) or cumulative == previous
                    or not sample.response.startswith(cumulative)):
                alignment_error = "Sentence labels are not growing prefixes of the original response"
                break
            native_label = sentence["target"]
            labels = {"hallucinated": 1, "non-hallucinated": 0, "invalid": None,
                      "undefined": None, None: None}
            if native_label not in labels:
                raise ValueError(f"Unknown HalluRAG sentence label: {native_label!r}")
            spans.append({"start": len(previous), "end": len(cumulative),
                          "label": labels[native_label], "native_label": native_label})
            previous = cumulative
        _check_spans(sample.response, spans)
        annotation = _annotation(sample, "sentence", spans=spans,
                                 coverage="annotated_sentences")
        if alignment_error:
            annotation.update(spans=[], coverage="unalignable", alignment_error=alignment_error,
                              native_sentences=[{"cum_sentence": item["cum_sentence"],
                                                 "target": item["target"]}
                                                for item in row["sentence_data"]])
        yield sample, annotation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_dataset(name: str, path: Path, output: Path, split: str = "test",
                    task: str | None = None, limit: int | None = None,
                    ids: list[str] | None = None) -> dict:
    """Prepare official rows without inventing splits or token-level labels."""
    path, output = Path(path).resolve(), Path(output)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    requested = None
    if ids is not None:
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(key, str) or not key for key in ids)):
            raise ValueError("ids must be a nonempty list of nonempty strings")
        requested = set(ids)
        if len(requested) != len(ids):
            raise ValueError("Duplicate requested sample ID")
        if limit is not None:
            raise ValueError("ID selection cannot be combined with limit")
    provenance_files = []
    if name == "ragtruth":
        examples = _ragtruth(path, split, task)
        raw_files = [path / "source_info.jsonl", path / "response.jsonl"]
        modality, version = "text", "official RAGTruth JSONL"
        prompt_policy = "unchanged official prompt"
        source_grouping = "official source_id"
        bootstrap_type = "source"
    elif name == "hallurag":
        split = "val" if split == "validation" else split
        examples = _hallurag(path, split, task)
        raw_files = [path / f"{split}.jsonl"]
        modality, version = "text", "doi:10.17879/84958668505; record revision 7"
        prompt_policy = "original message contents joined by two newlines for the text observer"
        source_grouping = "useful_art_i; all questions and generators of an article stay together"
        bootstrap_type = "article"
        receipt_name = ("ACQUISITION_RECEIPT.json" if split == "test" else
                        f"{split.upper()}_ACQUISITION_RECEIPT.json")
        receipt = path.parent / "provenance" / receipt_name
        if receipt.exists():
            provenance_files.append(receipt)
        export_receipt = path / f"{split}.receipt.json"
        if export_receipt.exists():
            provenance_files.append(export_receipt)
    elif name == "claimdb":
        examples = _claimdb(path, split, task)
        raw_files = [path / f"test-{split}.jsonl"]
        modality, version = "sqlite", "michaeltheologitis/claimdb@309fa756bfcdf56c47e4f96be5cdc1764ac0aa91"
        prompt_policy = "neutral verification instruction and released extra_info; no oracle metadata"
        source_grouping = "db_name"
        bootstrap_type = "database"
        receipt = path / "ACQUISITION_RECEIPT.json"
        if receipt.exists():
            provenance_files.append(receipt)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    inputs, annotations, seen = [], [], set()
    for sample, annotation in examples:
        if sample.id in seen:
            raise ValueError(f"Duplicate sample ID: {sample.id}")
        seen.add(sample.id)
        if requested is not None and sample.id not in requested:
            continue
        if modality == "text":
            _ = sample.inputs
        inputs.append(asdict(sample))
        annotations.append(annotation)
        if limit is not None and len(inputs) == limit:
            break
    if requested is not None and (missing := requested - seen):
        raise ValueError(f"Requested IDs not found in the selected split/task: {sorted(missing)}")
    if not inputs:
        raise ValueError("No samples match the requested dataset split and task")
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "inputs.jsonl", inputs)
    _write_jsonl(output / "annotations.jsonl", annotations)
    manifest = {
        "dataset": name, "version": version, "split": split, "modality": modality,
        "prompt_policy": prompt_policy, "source_grouping": source_grouping,
        "bootstrap_type": bootstrap_type,
        "samples": len(inputs), "sources": len({row["source_id"] for row in inputs}),
        "tasks": dict(Counter(row["task"] for row in inputs)),
        "granularities": dict(Counter(row["granularity"] for row in annotations)),
        "coverage": dict(Counter(row["coverage"] for row in annotations)),
        "selection": {"task": task, "limit": limit, "order": "original file order"},
        "raw_files": {str(file): data.file_hash(file) for file in raw_files},
        "provenance_files": {str(file): data.file_hash(file) for file in provenance_files},
        "evidence_files": {file: data.file_hash(Path(file))
                           for file in sorted({row["evidence_path"] for row in inputs
                                               if row["evidence_path"]})},
        "inputs_sha256": data.file_hash(output / "inputs.jsonl"),
        "annotations_sha256": data.file_hash(output / "annotations.jsonl"),
    }
    if ids is not None:
        manifest["selection"]["ids"] = list(ids)
    data.write_json(output / "manifest.json", manifest)
    return manifest
