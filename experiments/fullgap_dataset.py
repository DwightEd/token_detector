"""Sealed full-gap collection and original-token evaluation of released TEST sets."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments import data
from experiments.datasets import Sample, _hallurag_source, read_jsonl, read_samples
from experiments.inputs import source_span
from experiments.provenance import completed_identity, research_identity
from experiments.span_markup import fred_text
from myproject.measure import answer_axis, token_scores
from myproject.shared import SharedDetector


EXPECTED_SAMPLES = 1782
EXPECTED_GROUPS = 162
RESPONSES_PER_GROUP = 11
EXPECTED_UNALIGNABLE = 3
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 17
FULL_GAP_METHODS = ("witness", "matched_local", "symmetric_local")
OBSERVATION_PRIMITIVES = {
    "full_gap": "source_absent_logp - source_present_logp",
    "source_nll": "-source_present_logp",
}
ALLOWED_FULL_GAP_METHODS = (*FULL_GAP_METHODS, "fixed_local", "fixed_persistence",
                          *OBSERVATION_PRIMITIVES)
RAGTRUTH_SAMPLES = 900
RAGTRUTH_GROUPS = 150
RAGTRUTH_RESPONSES_PER_GROUP = 6
FRED_SAMPLES = {"finqa": 162, "tatqa": 130}
CONTRASTS = {
    "matched_local_minus_witness": ("witness", "matched_local"),
    "symmetric_local_minus_witness": ("witness", "symmetric_local"),
}
FIXED_CONTRAST = {"fixed_local_minus_witness": ("witness", "fixed_local")}
FIXED_PERSISTENCE_CONTRAST = {"fixed_persistence_minus_witness": ("witness", "fixed_persistence")}
PRIMITIVE_CONTRASTS = {f"{method}_minus_witness": ("witness", method)
                      for method in OBSERVATION_PRIMITIVES}
TOKEN_IDENTITY = (
    "id", "source_id", "generator", "response_sha256", "token_index", "token_id",
    "text", "start", "end", "native_sentence_indices", "label", "exclusion",
)
OBSERVATION_FIELDS = {
    "logp", "answer_ids", "offsets", "special_ids", "unit_start", "unit_stop",
    "target", "token_id", "answer_index", "unit_index",
}


def _hash_files(paths) -> dict[str, str]:
    return {str(Path(path).resolve()): data.file_hash(path) for path in paths}


def _model_files(model_path: Path) -> list[Path]:
    model_path = Path(model_path).resolve()
    if not model_path.is_dir():
        raise ValueError("Model path must be a local directory")
    names = {"tokenizer.model", "merges.txt", "vocab.txt"}
    files = sorted(path for path in model_path.rglob("*") if path.is_file()
                   and (path.suffix in {".json", ".safetensors", ".bin"}
                        or path.name in names))
    if not files or not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("Model identity must include local weight files")
    return files


def _verify_paths(bindings: dict[str, str], message: str) -> None:
    if any(not Path(path).is_file() or data.file_hash(Path(path)) != expected
           for path, expected in bindings.items()):
        raise ValueError(message)


def _label_contract(dataset: str) -> dict[str, str]:
    contracts = {
        "hallurag": {"native_granularity": "sentence",
                     "label_provenance": "sentence-derived token pseudo-labels",
                     "label_scope": "released automatic sentence-derived token pseudo-labels; not native human token labels"},
        "ragtruth": {"native_granularity": "token",
                     "label_provenance": "human character-span-derived token labels",
                     "label_scope": "released human character-span annotations projected by token overlap"},
        "fred": {"native_granularity": "token",
                 "label_provenance": "released synthetic/silver edit spans; not human natural errors",
                 "label_scope": "released synthetic/silver edit spans projected by token overlap; not human natural errors"},
    }
    return contracts[dataset]


def _original_samples(manifest, task):
    """Read only original inference fields, without interpreting released labels."""
    paths = {Path(path).name: Path(path) for path in manifest["raw_files"]}
    if len(paths) != len(manifest["raw_files"]):
        raise ValueError("Ambiguous bound raw filenames")
    dataset = manifest["dataset"]
    if dataset == "hallurag":
        if set(paths) != {"test.jsonl"}:
            raise ValueError("Expected the bound HalluRAG TEST JSONL")
        for row in read_jsonl(paths["test.jsonl"]):
            prompt, source = _hallurag_source(row["prompt"])
            quantization = row["quantization"]
            generator = row["model"] + (f" ({quantization})" if quantization else "")
            yield Sample(row["id"], str(row["prompt"]["useful_art_i"]),
                         row["original_split"], "QA", prompt, source,
                         row["llm_response"], generator)
    elif dataset == "ragtruth":
        if set(paths) != {"source_info.jsonl", "response.jsonl"}:
            raise ValueError("Expected the two bound RAGTruth raw JSONL files")
        sources = {}
        for row in read_jsonl(paths["source_info.jsonl"]):
            key = str(row["source_id"])
            if key in sources:
                raise ValueError("Duplicate original RAGTruth source ID")
            sources[key] = row
        for row in read_jsonl(paths["response.jsonl"]):
            source_id = str(row["source_id"])
            source = sources[source_id]
            if row["split"] != "test" or source["task_type"] != task:
                continue
            start, end = source_span(source)
            yield Sample(str(row["id"]), source_id, "test", task, source["prompt"],
                         source["prompt"][start:end], row["response"], row["model"])
    else:
        component = manifest["selection"]["task"]
        name = f"test_{component}.csv"
        if set(paths) != {name}:
            raise ValueError("Expected one bound FRED TEST component CSV")
        with paths[name].open(newline="", encoding="utf-8-sig") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                source, response = fred_text(row["prompt"])
                yield Sample(f"fred:{component}:test:{index}", data.text_hash(source),
                             "test", "QA", f"Reference material:\n{source}\n\nResponse:\n",
                             source, response, "not_released")


def _verify_original_inputs(manifest, samples, task):
    originals = {}
    for sample in _original_samples(manifest, task):
        if sample.id in originals:
            raise ValueError("Duplicate original response ID")
        originals[sample.id] = sample
    if (set(originals) != {sample.id for sample in samples}
            or any(originals[sample.id] != sample for sample in samples)):
        raise ValueError("Prepared inputs differ from the complete original raw response population")


def _prepared(prepared: Path):
    """Verify the complete inference population without opening annotations."""
    prepared = Path(prepared).resolve()
    manifest_path, input_path = prepared / "manifest.json", prepared / "inputs.jsonl"
    manifest = data.read_json(manifest_path)
    samples = list(read_samples(input_path))
    selection = manifest.get("selection", {})
    dataset = manifest.get("dataset")
    tasks = {sample.task for sample in samples}
    if (dataset not in {"hallurag", "ragtruth", "fred"}
            or manifest.get("split") != "test" or manifest.get("modality") != "text"
            or not samples or len(tasks) != 1 or manifest.get("samples") != len(samples)
            or selection.get("limit") is not None or selection.get("ids")
            or any(sample.split != "test" for sample in samples)):
        raise ValueError("Require a complete, unfiltered supported TEST text population")
    task = next(iter(tasks))
    if dataset == "hallurag":
        expected = EXPECTED_SAMPLES
        valid_task = task == "QA" and selection.get("task") in {None, "QA"}
    elif dataset == "ragtruth":
        expected = RAGTRUTH_SAMPLES
        valid_task = task in {"QA", "Summary", "Data2txt"} and selection.get("task") == task
    else:
        expected = FRED_SAMPLES.get(selection.get("task"))
        valid_task = task == "QA" and expected is not None
    if not valid_task or len(samples) != expected:
        raise ValueError(f"Require the complete, unfiltered {dataset} TEST task population")
    if dataset != "hallurag" and (
            manifest.get("granularities") != {"token": expected}
            or manifest.get("coverage") != {"complete": expected}
            or manifest.get("tasks") != {task: expected}):
        raise ValueError("Require complete native token annotations for the selected task")
    if data.file_hash(input_path) != manifest.get("inputs_sha256"):
        raise ValueError("Prepared full-gap inputs changed")
    raw = dict(manifest.get("raw_files", {}))
    provenance = dict(manifest.get("provenance_files", {}))
    _verify_paths(raw, "Raw full-gap acquisition changed")
    _verify_paths(provenance, "Full-gap provenance receipt changed")
    _verify_original_inputs(manifest, samples, task)

    groups = defaultdict(list)
    for answer_index, sample in enumerate(samples):
        inputs = sample.inputs
        if not sample.response:
            raise ValueError("Every original response must be nonempty")
        groups[(inputs["prompt"], inputs["source"])].append((answer_index, sample))
    if dataset != "fred":
        group_count = EXPECTED_GROUPS if dataset == "hallurag" else RAGTRUTH_GROUPS
        per_group = RESPONSES_PER_GROUP if dataset == "hallurag" else RAGTRUTH_RESPONSES_PER_GROUP
        if (len(groups) != group_count
                or any(len(rows) != per_group
                       or len({sample.generator for _, sample in rows}) != per_group
                       or len({sample.source_id for _, sample in rows}) != 1
                       for rows in groups.values())
                or (dataset == "ragtruth"
                    and len({sample.source_id for sample in samples}) != group_count)):
            raise ValueError(f"Expected {group_count} exact prompt/source groups with {per_group} original generators each")
    bindings = {
        str(input_path): manifest["inputs_sha256"],
        str(manifest_path): data.file_hash(manifest_path),
        **raw, **provenance,
    }
    return prepared, manifest, samples, list(groups.items()), bindings


def _axis_arrays(encoded: dict, answer_index: int) -> dict[str, np.ndarray]:
    """Use the public response-axis contract and retain its unit boundaries."""
    axis = answer_axis(encoded, answer_index)
    units = encoded["units"]
    starts = np.asarray([unit["start"] for unit in units], dtype=np.int32)
    stops = np.asarray([unit["stop"] for unit in units], dtype=np.int32)
    if (not len(axis["target"]) or not len(units) or starts[0] != 0
            or stops[-1] != len(encoded["answer_ids"])
            or not np.array_equal(starts[1:], stops[:-1]) or np.any(stops <= starts)):
        raise ValueError("Original response has no valid tokens or invalid text units")
    return {**axis, "unit_start": starts, "unit_stop": stops}


def collect(prepared: Path, model_path: Path, protocol: Path, output: Path,
            device: str = "cuda:0") -> None:
    """Measure all original responses with two matched-SDPA worlds."""
    import torch

    from myproject.api import HallucinationDetector

    prepared, manifest, samples, groups, prepared_bindings = _prepared(prepared)
    model_path, protocol, output = (Path(model_path).resolve(), Path(protocol).resolve(),
                                    Path(output).resolve())
    if not protocol.is_file():
        raise ValueError("A frozen collection protocol is required")
    model_bindings = _hash_files(_model_files(model_path))
    configured_limit = data.read_json(model_path / "config.json").get("max_position_embeddings")
    context_limit = (configured_limit if type(configured_limit) is int and configured_limit > 0
                     else None)
    group_sizes = [len(rows) for _, rows in groups]
    identity = research_identity()
    plan = {
        "prepared": str(prepared), "input": str(prepared / "inputs.jsonl"),
        "input_sha256": manifest["inputs_sha256"],
        "manifest_sha256": prepared_bindings[str(prepared / "manifest.json")],
        "raw_files": manifest.get("raw_files", {}),
        "provenance_files": manifest.get("provenance_files", {}),
        "annotation_sha256_from_manifest": manifest.get("annotations_sha256"),
        "model": str(model_path), "model_files": model_bindings,
        "protocol": str(protocol), "protocol_sha256": data.file_hash(protocol),
        "source": identity, "task": samples[0].task, "dataset": manifest["dataset"],
        "label_contract": _label_contract(manifest["dataset"]),
        "method_at_load": "witness", "device": device,
        "world_columns": ["source_present", "source_absent"],
        "score_gap": "source_absent_logp - source_present_logp",
        "backend": "matched_sdpa", "archived_eager_byte_equal_claim": False,
        "context_budget": {"max_position_embeddings": context_limit, "truncation": False,
                           "policy": "reject original prompt plus response exceeding the declared model limit"},
        "population": {"responses": len(samples), "prefix_groups": len(groups),
                       "responses_per_group": (group_sizes[0] if len(set(group_sizes)) == 1 else None),
                       "response_group_size_counts": dict(Counter(group_sizes))},
        "annotation_reads": 0,
        "claim_scope": "predictive full-gap observations; no factuality calibration",
    }
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / "plan.json", plan)
    destination = output / "observations"
    destination.mkdir()
    torch.manual_seed(42)
    detector = HallucinationDetector.from_pretrained(
        model_path, task=samples[0].task, method="witness", device=device,
    )
    records = []
    for group_index, ((prompt, source), rows) in enumerate(groups):
        responses = [sample.response for _, sample in rows]
        if context_limit is not None:
            from myproject.inputs import encode_response, encode_source

            encoded_source = encode_source(detector.tokenizer, prompt, source)
            prefix_length = max(len(encoded_source[name]) for name in
                                ("prompt_with_source", "prompt_without_source"))
            if any(prefix_length + len(encode_response(detector.tokenizer, text)["answer_ids"])
                   > context_limit for text in responses):
                raise ValueError("Original prompt plus response exceeds model context budget; truncation is forbidden")
        measured, encodings = detector.measure_responses(prompt, responses, source)
        if len(measured) != len(rows) or len(encodings) != len(rows):
            raise ValueError("Grouped measurement dropped an original response")
        for (answer_index, sample), logp, encoded in zip(
                rows, measured, encodings, strict=True):
            logp = np.asarray(logp, dtype=np.float64)
            answer_ids = np.asarray(encoded["answer_ids"])
            offsets = np.asarray(encoded["offsets"], dtype=np.int64).reshape(-1, 2)
            if (encoded.get("text") != sample.response or logp.shape != (len(answer_ids), 2)
                    or offsets.shape != (len(answer_ids), 2) or not np.isfinite(logp).all()):
                raise ValueError("Measurement changed an original response axis")
            axis = _axis_arrays(encoded, answer_index)
            file = data.prediction_path(destination, sample.id).with_suffix(".npz")
            with file.open("xb") as stream:
                np.savez_compressed(
                    stream, logp=logp, answer_ids=answer_ids, offsets=offsets,
                    special_ids=np.asarray(encoded["special_ids"]), **axis,
                )
            records.append({
                "id": sample.id, "source_id": sample.source_id,
                "answer_index": answer_index, "prefix_group": group_index,
                "file": str(file.relative_to(output)), "sha256": data.file_hash(file),
                "response_sha256": data.text_hash(sample.response),
            })
        print("FULLGAP_COLLECT", group_index + 1, len(groups), flush=True)
    if [record["answer_index"] for record in sorted(records, key=lambda row: row["answer_index"])] != list(range(len(samples))):
        raise ValueError("Collection did not preserve every original response")
    _verify_paths(prepared_bindings, "Prepared full-gap inputs changed during collection")
    if _hash_files(_model_files(model_path)) != model_bindings:
        raise ValueError("Model bytes changed during collection")
    if data.file_hash(protocol) != plan["protocol_sha256"]:
        raise ValueError("Collection protocol changed during measurement")
    collection = {
        "records": sorted(records, key=lambda row: row["answer_index"]),
        "source": completed_identity(identity), "annotation_reads": 0,
        "counts": {"responses": len(records), "measure_responses_calls": len(groups),
                   "matched_sdpa_prefills": 2 * len(groups)},
    }
    data.write_json(output / "collection.json", collection)
    data.write_json(output / "seal.json", {
        name: data.file_hash(output / name) for name in ("plan.json", "collection.json")
    })


def _verify_simple_seal(root: Path, *, expected=None, required=()) -> None:
    data.verify_seal(root, expected=expected, required=required)


def _load_observation(path: Path, sample, answer_index: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != OBSERVATION_FIELDS:
            raise ValueError("Full-gap observation schema changed")
        raw = {name: saved[name] for name in saved.files}
    answer_ids, offsets, logp = raw["answer_ids"], raw["offsets"], raw["logp"]
    if (logp.shape != (len(answer_ids), 2) or offsets.shape != (len(answer_ids), 2)
            or not np.isfinite(logp).all() or np.any(offsets < 0)
            or np.any(offsets > len(sample.response)) or np.any(offsets[:, 1] < offsets[:, 0])):
        raise ValueError("Invalid raw full-gap response axis")
    valid = ((offsets[:, 1] > offsets[:, 0])
             & ~np.isin(answer_ids, raw["special_ids"]))
    target = np.flatnonzero(valid)
    starts, stops = raw["unit_start"], raw["unit_stop"]
    if (not len(starts) or starts.shape != stops.shape or starts[0] != 0
            or stops[-1] != len(answer_ids) or not np.array_equal(starts[1:], stops[:-1])
            or np.any(stops <= starts)):
        raise ValueError("Invalid saved original text units")
    unit_index = np.empty(len(answer_ids), dtype=np.int32)
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        unit_index[int(start):int(stop)] = index
    if (not np.array_equal(raw["target"], target)
            or not np.array_equal(raw["token_id"], answer_ids[target])
            or not np.array_equal(raw["answer_index"], np.full(len(target), answer_index))
            or not np.array_equal(raw["unit_index"], unit_index[target])):
        raise ValueError("Saved valid-token or unit axis changed")
    return raw


def _read_native(native: Path):
    native = Path(native).resolve()
    _verify_simple_seal(native, expected={'plan.json', 'collection.json'})
    plan, collection = data.read_json(native / "plan.json"), data.read_json(native / "collection.json")
    prepared, manifest, samples, _groups, prepared_bindings = _prepared(Path(plan["prepared"]))
    if (plan["input_sha256"] != manifest["inputs_sha256"]
            or plan["manifest_sha256"] != prepared_bindings[str(prepared / "manifest.json")]
            or data.file_hash(Path(plan["protocol"])) != plan["protocol_sha256"]
            or plan.get("backend") != "matched_sdpa"
            or plan.get("world_columns") != ["source_present", "source_absent"]):
        raise ValueError("Native full-gap provenance changed")
    if (plan.get("task") != samples[0].task
            or plan.get("dataset", manifest["dataset"]) != manifest["dataset"]
            or plan.get("label_contract", _label_contract(manifest["dataset"])) !=
            _label_contract(manifest["dataset"])):
        raise ValueError("Native dataset/task/label contract changed")
    _verify_paths(prepared_bindings, "Prepared full-gap provenance changed")
    records = collection["records"]
    if len(records) != len(samples):
        raise ValueError("Native full-gap response coverage changed")
    expected_files = set()
    for answer_index, (record, sample) in enumerate(zip(records, samples, strict=True)):
        file = native / record["file"]
        expected_files.add(file.name)
        if (record["id"] != sample.id or record["source_id"] != sample.source_id
                or record["answer_index"] != answer_index
                or record["response_sha256"] != data.text_hash(sample.response)
                or data.file_hash(file) != record["sha256"]):
            raise ValueError("Native full-gap response identity changed")
        _load_observation(file, sample, answer_index)
    if {path.name for path in (native / "observations").iterdir()} != expected_files:
        raise ValueError("Native full-gap observation file set changed")
    return plan, samples, records


def _score_run(native: Path, native_plan: dict, method: str, directory: Path,
               samples, records, identity: dict) -> None:
    from experiments.pipeline import _validate_prediction

    primitive = method in OBSERVATION_PRIMITIVES
    readout = None if primitive else SharedDetector(native_plan["task"], method=method)
    cases = directory / "predictions"
    cases.mkdir(parents=True)
    for answer_index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        raw = _load_observation(native / record["file"], sample, answer_index)
        target = raw["target"]
        gap = raw["logp"][target, 1] - raw["logp"][target, 0]
        if method == "full_gap":
            scores = gap
        elif method == "source_nll":
            scores = -raw["logp"][target, 0]
        else:
            scores = readout.score_full_gap(gap, raw["answer_index"], raw["unit_index"])
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != gap.shape or not np.isfinite(scores).all():
            raise ValueError("Full-gap readout returned invalid original-token scores")
        tokens = token_scores(
            {"target": target, "token_id": raw["token_id"]},
            {"offsets": raw["offsets"], "text": sample.response}, scores,
        )
        prediction = {"id": sample.id, "source_id": sample.source_id,
                      "response_sha256": data.text_hash(sample.response), "tokens": tokens}
        _validate_prediction(prediction, sample)
        data.write_json(data.prediction_path(cases, sample.id), prediction)
    data.write_json(cases / "seal.json", {
        path.name: data.file_hash(path) for path in sorted(cases.iterdir())
    })
    run = {
        "input": native_plan["input"], "input_sha256": native_plan["input_sha256"],
        "manifest_sha256": native_plan["manifest_sha256"], "task": native_plan["task"], "method": method,
        "native": str(native), "native_seal_sha256": data.file_hash(native / "seal.json"),
        "observation_source": native_plan["source"], "source": identity,
        "gap_orientation": "source_absent_logp - source_present_logp",
        "score_kind": "observation primitive, not latent state" if primitive else "latent-state readout",
        "score_definition": OBSERVATION_PRIMITIVES.get(method, "public SharedDetector.score_full_gap"),
        "claim_scope": "ranking score; no factuality calibration", "annotation_reads": 0,
    }
    data.write_json(directory / "run.json", run)
    data.write_json(directory / "complete.json", {
        "samples": len(samples),
        "prediction_seal_sha256": data.file_hash(cases / "seal.json"),
        "run_sha256": data.file_hash(directory / "run.json"), "annotation_reads": 0,
    })


def score(native: Path, protocol: Path, output: Path,
          methods: tuple[str, ...] = FULL_GAP_METHODS) -> None:
    """Apply public full-gap readouts; labels remain unopened until evaluation."""
    native, protocol, output = Path(native).resolve(), Path(protocol).resolve(), Path(output).resolve()
    if (not methods or len(set(methods)) != len(methods)
            or any(method not in ALLOWED_FULL_GAP_METHODS for method in methods)):
        raise ValueError(f"Methods must be distinct members of {ALLOWED_FULL_GAP_METHODS}")
    if not protocol.is_file():
        raise ValueError("A frozen scoring protocol is required")
    native_plan, samples, records = _read_native(native)
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "native": str(native), "native_seal_sha256": data.file_hash(native / "seal.json"),
        "protocol": str(protocol), "protocol_sha256": data.file_hash(protocol),
        "methods": list(methods), "source": identity,
        "observation_source": native_plan["source"], "annotation_reads": 0,
        "observation_controls": {
            method: {"kind": "observation primitive, not latent state", "definition": definition}
            for method, definition in OBSERVATION_PRIMITIVES.items() if method in methods},
        "claim_scope": "full-gap ranking comparison; no factuality calibration",
    }
    data.write_json(output / "plan.json", plan)
    for method in methods:
        _score_run(native, native_plan, method, output / method, samples, records, identity)
    _read_native(native)
    if data.file_hash(protocol) != plan["protocol_sha256"]:
        raise ValueError("Scoring protocol changed during inference")
    completed = completed_identity(identity)
    data.write_json(output / "complete.json", {
        "methods": list(methods), "samples": len(samples), "source": completed,
        "annotation_reads": 0,
    })
    files = {"plan.json": data.file_hash(output / "plan.json"),
             "complete.json": data.file_hash(output / "complete.json")}
    for method in methods:
        for name in ("run.json", "complete.json", "predictions/seal.json"):
            files[f"{method}/{name}"] = data.file_hash(output / method / name)
    data.write_json(output / "seal.json", files)


def _verify_scores(native: Path, scores: Path):
    from experiments.pipeline import _validate_prediction, _verify_predictions

    _verify_simple_seal(scores, required={'plan.json', 'complete.json'})
    score_plan = data.read_json(scores / "plan.json")
    score_complete = data.read_json(scores / "complete.json")
    if (score_plan["native"] != str(native)
            or score_plan["native_seal_sha256"] != data.file_hash(native / "seal.json")
            or data.file_hash(Path(score_plan["protocol"])) != score_plan["protocol_sha256"]):
        raise ValueError("Score/native/protocol binding changed")
    methods = score_plan["methods"]
    if (not methods or len(set(methods)) != len(methods)
            or any(method not in ALLOWED_FULL_GAP_METHODS for method in methods)):
        raise ValueError("Sealed method set changed")
    _verify_simple_seal(scores, expected={'plan.json', 'complete.json'} | {
        f'{method}/{name}' for method in methods for name in
        ('run.json', 'complete.json', 'predictions/seal.json')})
    # Authenticate every score file before any raw-file identity/metadata join.
    for method in methods:
        _verify_predictions(scores / method / "predictions")
    native_plan, samples, records = _read_native(native)
    native_tokens = defaultdict(list)
    for (sample_id, _ordinal), token in _native_token_identities(native, samples, records).items():
        native_tokens[sample_id].append(token)
    if score_complete != {
            "methods": methods, "samples": len(samples), "source": score_plan["source"],
            "annotation_reads": 0}:
        raise ValueError("Completed full-gap method coverage changed")
    for method in methods:
        directory = scores / method
        expected_files = {data.prediction_path(directory / 'predictions', sample.id).name
                          for sample in samples}
        if set(data.read_json(directory / 'predictions/seal.json')) != expected_files:
            raise ValueError('Prediction filenames differ from the native population')
        run = data.read_json(directory / "run.json")
        complete = data.read_json(directory / "complete.json")
        if (complete != {"samples": len(samples),
                         "prediction_seal_sha256": data.file_hash(directory / "predictions/seal.json"),
                         "run_sha256": data.file_hash(directory / "run.json"),
                         "annotation_reads": 0}
                or run["input"] != native_plan["input"]
                or run["input_sha256"] != native_plan["input_sha256"]
                or run["manifest_sha256"] != native_plan["manifest_sha256"]
                or run["native"] != str(native)
                or run["native_seal_sha256"] != data.file_hash(native / "seal.json")
                or run["task"] != native_plan["task"]
                or run["method"] != method or run["annotation_reads"] != 0):
            raise ValueError("Completed full-gap score binding changed")
        for sample in samples:
            prediction = data.read_json(data.prediction_path(directory / "predictions", sample.id))
            tokens = _validate_prediction(prediction, sample)
            expected = native_tokens[sample.id]
            if (len(tokens) != len(expected) or any(
                    any(token.get(name) != original[name] for name in ('token_id', 'text', 'start', 'end'))
                    for token, original in zip(tokens, expected, strict=True))):
                raise ValueError('Prediction differs from the original native token axis')
    return native_plan, samples, records, methods


def _response_metadata(manifest, samples):
    """Join released TEST metadata by exact ID after all scores have verified."""
    if manifest["dataset"] != "hallurag":
        _verify_paths(manifest.get("raw_files", {}), "Raw response metadata changed")
        metadata = [{"id": sample.id, "source_id": sample.source_id,
                     "response_sha256": data.text_hash(sample.response)} for sample in samples]
        return metadata, {"raw_files": manifest["raw_files"], "responses": len(samples),
                          "join": "exact original inference identities verified against bound raw files"}
    raw_files = manifest.get("raw_files", {})
    if len(raw_files) != 1 or Path(next(iter(raw_files))).suffix != ".jsonl":
        raise ValueError("Expected one bound HalluRAG TEST raw JSONL")
    _verify_paths(raw_files, "Raw HalluRAG metadata changed")
    raw_path, raw_hash = next(iter(raw_files.items()))
    rows = {}
    for row in read_jsonl(Path(raw_path)):
        key = row.get("id")
        if not isinstance(key, str) or not key or key in rows:
            raise ValueError("Duplicate or invalid released metadata ID")
        rows[key] = row
    if set(rows) != {sample.id for sample in samples}:
        raise ValueError("Raw metadata requires exact prepared sample IDs")
    metadata = []
    for sample in samples:
        row = rows[sample.id]
        prompt = row.get("prompt", {})
        if not isinstance(prompt, dict):
            raise ValueError("Invalid released HalluRAG prompt metadata")
        answerable = prompt.get("answerable")
        index = prompt.get("answer_chunk_index")
        if (type(answerable) is not bool or "answer_chunk_index" not in prompt
                or (index is not None and (type(index) is not int or index < 0))):
            raise ValueError("Invalid typed HalluRAG answerability/evidence metadata")
        try:
            text, source = _hallurag_source(prompt)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid released HalluRAG prompt/source") from error
        if (row.get("original_split") != "test" or row.get("llm_response") != sample.response
                or "useful_art_i" not in prompt
                or str(prompt["useful_art_i"]) != sample.source_id
                or text != sample.prompt or source != sample.source):
            raise ValueError("Raw metadata response/source/prompt/split differs from prepared input")
        metadata.append({"id": sample.id, "source_id": sample.source_id,
                         "response_sha256": data.text_hash(sample.response),
                         "answerable": answerable, "answer_chunk_index": index,
                         "raw_sha256": raw_hash})
    binding = {"raw_path": raw_path, "raw_sha256": raw_hash, "responses": len(metadata),
               "join": "exact released ID and verified response/source/prompt",
               "partitions_coincide": all(row["answerable"] ==
                   (row["answer_chunk_index"] is not None) for row in metadata),
               "availability_definition": "released answer_chunk_index is not None; not a new content audit"}
    return metadata, binding


def _native_token_identities(native, samples, records):
    """Canonical valid-token ordinals; duplicate character offsets stay distinct."""
    identities = {}
    for answer_index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        raw = _load_observation(native / record["file"], sample, answer_index)
        response_hash = data.text_hash(sample.response)
        for ordinal, target in enumerate(raw["target"]):
            start, end = map(int, raw["offsets"][target])
            identities[(sample.id, ordinal)] = {
                "id": sample.id, "source_id": sample.source_id, "generator": sample.generator,
                "response_sha256": response_hash, "token_index": ordinal,
                "token_id": int(raw["token_id"][ordinal]), "text": sample.response[start:end],
                "start": start, "end": end,
            }
    return identities


def _native_token_units(native, samples, records):
    """Retain the original answer/span axes when unknown labels are filtered later."""
    axes = {}
    for answer_index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        raw = _load_observation(native / record["file"], sample, answer_index)
        axes.update({(sample.id, ordinal): (answer_index, int(unit))
                     for ordinal, unit in enumerate(raw["unit_index"])})
    return axes


def _verify_method_evaluation(directory, score_directory, input_hash, annotation_hash):
    expected = {"evaluation.json", "units.jsonl", "token_labels.jsonl"}
    if (set(data.read_json(directory / "seal.json")) != expected
            or {path.name for path in directory.iterdir()} != expected | {"seal.json"}):
        raise ValueError("Method evaluation sealed file set changed")
    _verify_simple_seal(directory)
    report = data.read_json(directory / "evaluation.json")
    if (report["input_sha256"] != input_hash or report["annotation_sha256"] != annotation_hash
            or report["prediction_seal_sha256"] !=
            data.file_hash(score_directory / "predictions/seal.json")):
        raise ValueError("Method evaluation input/annotation/prediction binding changed")


def _token_key(row):
    if (not isinstance(row.get("id"), str)
            or type(row.get("token_index")) is not int or row["token_index"] < 0):
        raise ValueError("Invalid evaluation token identity")
    return row["id"], row["token_index"]


def _evaluation_rows(directory, native_identities):
    """Read sealed projection products without reimplementing label projection."""
    labels = {}
    for row in read_jsonl(directory / "token_labels.jsonl"):
        key = _token_key(row)
        if key in labels or key not in native_identities:
            raise ValueError("Duplicate or extra all-token evaluation key")
        if (any(row.get(name) != value for name, value in native_identities[key].items())
                or any(name not in row for name in TOKEN_IDENTITY)
                or (row["label"] is not None and type(row["label"]) is not bool)
                or not isinstance(row["native_sentence_indices"], list)
                or (row["label"] is None) != (row["exclusion"] is not None)):
            raise ValueError("Evaluation token/native identity or label exclusion changed")
        labels[key] = row
    if set(labels) != set(native_identities):
        raise ValueError("All-token evaluation native-axis coverage changed")
    known = {key for key, row in labels.items() if row["label"] is not None}
    units = {}
    for row in read_jsonl(directory / "units.jsonl"):
        key = _token_key(row)
        if key in units or key not in known:
            raise ValueError("Duplicate or extra known-token evaluation key")
        if (any(row.get(name) != labels[key][name] for name in
                ("id", "source_id", "generator", "response_sha256", "token_index", "token_id", "text", "start", "end", "label"))
                or type(row.get("score")) not in (int, float) or not np.isfinite(row["score"])):
            raise ValueError("Known-token evaluation identity/label/score changed")
        units[key] = row
    if set(units) != known:
        raise ValueError("Known-token evaluation coverage changed")
    return labels, units


def _paired_views(keys, labels, method_units, metadata, native_units=None, *, contrasts=None):
    from experiments.evaluate import paired_source_differences, ranking
    from experiments.local_evaluation import paired_span_differences

    by_id = {row["id"]: row for row in metadata}
    definitions = ({**CONTRASTS, **FIXED_CONTRAST, **FIXED_PERSISTENCE_CONTRAST,
                    **PRIMITIVE_CONTRASTS} if contrasts is None else contrasts)
    available = {name: pair for name, pair in definitions.items()
                 if all(method in method_units for method in pair)}

    def view(ids):
        all_keys = [key for key in keys if key[0] in ids]
        known = [key for key in all_keys if labels[key]["label"] is not None]
        y = np.asarray([labels[key]["label"] for key in known], dtype=bool)
        sources = np.asarray([labels[key]["source_id"] for key in known])
        arrays = {method: np.asarray([units[key]["score"] for key in known])
                  for method, units in method_units.items()}
        metrics = {method: ranking(y, values) if known else None
                   for method, values in arrays.items()}
        differences = {}
        for name, (baseline, candidate) in available.items():
            point = {metric: (metrics[candidate][metric]-metrics[baseline][metric]
                             if known and metrics[baseline][metric] is not None else None)
                     for metric in ("auroc", "ap")}
            intervals = (paired_source_differences(
                y, arrays[baseline], arrays[candidate], sources,
                repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED,
            ) if known else {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS,
                "unit": "source_id", **{metric: {"defined": 0, "degenerate": BOOTSTRAP_REPETITIONS,
                    "interval95": None} for metric in ("auroc", "ap")}})
            differences[name] = {"point": point, "bootstrap": intervals,
                                 "sources_evaluated": len(set(sources))}
            if native_units is not None:
                answers = np.asarray([native_units[key][0] for key in known], dtype=np.int64)
                spans = np.asarray([native_units[key][1] for key in known], dtype=np.int64)
                differences[name]["within_span"] = paired_span_differences(
                    y, arrays[baseline], arrays[candidate], sources, answers, spans,
                    repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED,
                )
        exclusions = Counter(labels[key]["exclusion"] for key in all_keys
                             if labels[key]["label"] is None)
        return {"counts": {
            "responses_scored": len(ids), "responses_evaluable": len({key[0] for key in known}),
            "sources_scored": len({by_id[key]["source_id"] for key in ids}),
            "sources_evaluable": len(set(sources)), "tokens_scored": len(all_keys),
            "tokens_evaluable": len(known), "positive_tokens": int(y.sum()),
            "negative_tokens": int(len(y)-y.sum()), "unknown_tokens": sum(exclusions.values()),
            "unknown_exclusions": dict(exclusions),
            "unalignable_responses": len({key[0] for key in all_keys
                                          if labels[key]["exclusion"] == "unalignable"}),
        }, "methods": metrics, "paired_differences": differences,
            "metric_status": ("no evaluable tokens" if not known else
                              "single class; AUROC undefined" if not y.any() or y.all() else "defined")}

    pooled = view(set(by_id))
    if not all("answerable" in row and "answer_chunk_index" in row for row in metadata):
        return pooled, {}
    definitions = {
        "answerable.true": ("released prompt.answerable is True", lambda row: row["answerable"]),
        "answerable.false": ("released prompt.answerable is False", lambda row: not row["answerable"]),
        "target_evidence.available": ("released answer_chunk_index is not None",
                                      lambda row: row["answer_chunk_index"] is not None),
        "target_evidence.unavailable": ("released answer_chunk_index is None",
                                        lambda row: row["answer_chunk_index"] is None),
    }
    strata = {name: {"definition": definition, "role": "descriptive secondary",
                     **view({row["id"] for row in metadata if predicate(row)})}
              for name, (definition, predicate) in definitions.items()}
    return pooled, strata


def evaluate(native: Path, scores: Path, output: Path) -> dict:
    """Join released labels only after every original-token score artifact verifies."""
    from experiments.evaluate import evaluate_dataset

    native, scores, output = Path(native).resolve(), Path(scores).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output already exists")
    identity = research_identity()
    native_plan, samples, records, methods = _verify_scores(native, scores)
    prepared = Path(native_plan["prepared"])
    manifest = data.read_json(prepared / "manifest.json")
    metadata, binding = _response_metadata(manifest, samples)
    contract = _label_contract(manifest["dataset"])
    native_identities = _native_token_identities(native, samples, records)
    native_units = _native_token_units(native, samples, records)
    contrasts = {**CONTRASTS, **(FIXED_CONTRAST if "fixed_local" in methods else {}),
                 **(FIXED_PERSISTENCE_CONTRAST if "fixed_persistence" in methods else {}),
                 **{name: pair for name, pair in PRIMITIVE_CONTRASTS.items()
                    if all(method in methods for method in pair)}}
    score_plan = data.read_json(scores / "plan.json")
    start = {"source": identity, "native_seal_sha256": data.file_hash(native / "seal.json"),
             "score_seal_sha256": data.file_hash(scores / "seal.json"),
             "annotation_sha256": manifest["annotations_sha256"], "metadata_binding": binding,
             "protocol_sha256": score_plan["protocol_sha256"], "methods": methods,
             "contrasts": contrasts, "label_contract": contract}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / "evaluation_start.json", start)
    data.write_json(output / "response_metadata.json", {"responses": metadata, "binding": binding})
    reports, all_labels, method_units = {}, {}, {}
    for method in methods:
        report = evaluate_dataset(
            scores / method, prepared / "annotations.jsonl", output / method,
            repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED,
            label_granularity="token",
        )
        if (report.get("native_granularity") != contract["native_granularity"]
                or report.get("label_provenance") != contract["label_provenance"]
                or report.get("samples_scored") != len(samples)
                or report.get("unalignable_responses_excluded") !=
                   (EXPECTED_UNALIGNABLE if manifest["dataset"] == "hallurag" else 0)
                or report.get("coverage") != manifest.get("coverage")
                or report["bootstrap"]["seed"] != BOOTSTRAP_SEED
                or report["bootstrap"]["repetitions"] != BOOTSTRAP_REPETITIONS):
            raise ValueError("Released label scope or coverage changed")
        reports[method] = report
        _verify_method_evaluation(output / method, scores / method,
                                  native_plan["input_sha256"], manifest["annotations_sha256"])
        all_labels[method], method_units[method] = _evaluation_rows(output / method, native_identities)
    baseline = "witness" if "witness" in methods else methods[0]
    labels = all_labels[baseline]
    for method in methods:
        if any(any(all_labels[method][key][name] != labels[key][name]
                   for name in TOKEN_IDENTITY) for key in native_identities):
            raise ValueError("Paired methods changed original token/label/unknown identity")
    pooled, strata = _paired_views(list(native_identities), labels, method_units, metadata, native_units)
    missing = [name for name in CONTRASTS if name not in pooled["paired_differences"]]
    _verify_scores(native, scores)
    _verify_paths({str(native / "seal.json"): start["native_seal_sha256"],
                   str(scores / "seal.json"): start["score_seal_sha256"],
                   str(prepared / "annotations.jsonl"): start["annotation_sha256"]},
                  "Evaluation input bindings changed during evaluation")
    final_metadata, final_binding = _response_metadata(manifest, samples)
    if metadata != final_metadata or binding != final_binding:
        raise ValueError("Released metadata changed during evaluation")
    for method in methods:
        _verify_method_evaluation(output / method, scores / method,
                                  native_plan["input_sha256"], manifest["annotations_sha256"])
    summary = {
        "methods": {method: report["metrics"] for method, report in reports.items()},
        "score_seal_sha256": data.file_hash(scores / "seal.json"),
        "samples_scored": len(samples), "dataset": manifest["dataset"], "task": samples[0].task,
        "native_granularity": contract["native_granularity"],
        "label_scope": contract["label_scope"],
        "unknown_policy": "undefined, uncovered, and unalignable token labels are retained as unknown then excluded",
        "factuality_calibration": False,
        "bootstrap": {"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED,
                      "unit": "source_id", "cluster_semantics": manifest["bootstrap_type"],
                      "cluster_population": "sources represented by evaluable tokens in each view"},
        "population": pooled["counts"], "paired_differences": pooled["paired_differences"],
        "strata": strata, "metadata_binding": binding, "missing_contrasts": missing,
        "protocol_comparison_complete": set(FULL_GAP_METHODS).issubset(methods) and not missing,
        "contrast_definitions": contrasts,
        "observation_controls": score_plan.get("observation_controls", {}),
        "observation_source": native_plan["source"], "scorer_source": score_plan["source"],
        "evaluation_source": completed_identity(identity),
        "native_seal_sha256": start["native_seal_sha256"],
        "annotation_sha256": start["annotation_sha256"],
        "protocol_sha256": start["protocol_sha256"],
    }
    data.write_json(output / "evaluation.json", summary)
    data.write_json(output / "seal.json", {
        "evaluation.json": data.file_hash(output / "evaluation.json"),
        "evaluation_start.json": data.file_hash(output / "evaluation_start.json"),
        "response_metadata.json": data.file_hash(output / "response_metadata.json"),
        **{f"{method}/seal.json": data.file_hash(output / method / "seal.json")
           for method in methods},
    })
    return summary
