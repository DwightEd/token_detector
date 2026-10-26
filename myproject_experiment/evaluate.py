"""Official labels are joined only after the saved scores pass their seal."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import data


def token_labels(annotation: dict, response: dict) -> np.ndarray:
    """Map official character overlaps to the unchanged original token axis."""
    offsets = np.asarray(response["offsets"])
    labels = np.zeros(len(offsets), dtype=bool)
    for span in annotation["labels"]:
        labels |= (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])
    return labels


def official_labels(scores: dict, metadata: dict) -> np.ndarray:
    """Check token identities before joining annotations; never change scores."""
    records = metadata["records"]
    wanted = {str(records[index]["id"]) for index in np.unique(scores["answer_index"])}
    annotations = {}
    with (Path(metadata["dataset"]) / "response.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["id"]) in wanted:
                annotations[str(row["id"])] = row
    labels = np.empty(len(scores["score"]), dtype=bool)
    for index in np.unique(scores["answer_index"]):
        record = records[index]
        response = data.read_response(metadata, record)
        row = annotations[str(record["id"])]
        if row["response"] != response["text"]:
            raise ValueError("Official response text differs from the frozen measurement")
        selected = scores["answer_index"] == index
        target = scores["target"][selected]
        if not np.array_equal(np.asarray(response["answer_ids"])[target], scores["token_id"][selected]):
            raise ValueError("Score token identities differ from prepared inputs")
        labels[selected] = token_labels(row, response)[target]
    return labels


def ranking(labels: np.ndarray, score: np.ndarray) -> dict:
    positives = int(labels.sum())
    alarm = np.zeros(len(score), dtype=bool)
    alarm[np.argsort(-score, kind="stable")[:int(np.ceil(.05 * len(score)))]] = True
    return {
        "tokens": len(score), "errors": positives,
        "auroc": float(roc_auc_score(labels, score)) if 0 < positives < len(score) else None,
        "ap": float(average_precision_score(labels, score)) if positives else None,
        "top5pct": {"tp": int((alarm & labels).sum()), "fp": int((alarm & ~labels).sum()),
                    "fn": int((~alarm & labels).sum())},
    }


def evaluate(output: Path) -> dict:
    """Verify sealed files, then read Gold and produce one compact report."""
    seal = data.read_json(output / "seal.json")
    for name, expected in seal.items():
        if data.file_hash(output / name) != expected:
            raise ValueError(f"Sealed prediction artifact changed: {name}")
    metadata = data.read_json(output / "metadata.json")
    with np.load(output / "scores.npz", allow_pickle=False) as saved:
        scores = {name: saved[name] for name in saved.files}
    report = ranking(official_labels(scores, metadata), scores["score"])
    data.write_json(output / "evaluation.json", report)
    return report


def _span_score(tokens: list[dict], start: int, end: int) -> float:
    """Average token evidence by character overlap with one native labelled unit."""
    weights = np.asarray([max(0, min(end, row["end"]) - max(start, row["start"]))
                          for row in tokens], dtype=float)
    if not weights.sum():
        raise ValueError("An annotated unit has no observed tokens")
    return float(np.average([row["score"] for row in tokens], weights=weights))


def _cluster_intervals(labels, scores, groups, repetitions, seed):
    """Bootstrap whole groups, reusing score order and ties across all draws."""
    _, index = np.unique(groups, return_inverse=True)
    count = int(index.max()) + 1
    order = np.argsort(scores, kind="stable")[::-1]
    labels, index = labels[order], index[order]
    first = np.r_[0, np.flatnonzero(scores[order][1:] != scores[order][:-1]) + 1]
    values = {"auroc": [], "ap": []}
    draws = np.random.default_rng(seed).integers(0, count, (repetitions, count))
    for draw in draws:
        weights = np.bincount(draw, minlength=count)[index]
        positive = np.add.reduceat(weights * labels, first)
        total = np.add.reduceat(weights, first)
        negative = total - positive
        npos, nneg = positive.sum(), negative.sum()
        if npos and nneg:
            # Each positive beats negatives below its tie and half within it.
            concordance = np.sum(positive * (nneg - np.cumsum(negative) + .5 * negative))
            values["auroc"].append(float(concordance / (npos * nneg)))
        if npos:
            cumulative = np.cumsum(total)
            precision = np.divide(np.cumsum(positive), cumulative,
                                  out=np.zeros(len(total), float), where=cumulative > 0)
            values["ap"].append(float(positive @ precision / npos))
    return {name: {"defined": len(samples), "degenerate": repetitions - len(samples),
                   "interval95": np.quantile(samples, [.025, .975]).tolist() if samples else None}
            for name, samples in values.items()}


def evaluate_dataset(output: Path, annotations: Path | None, destination: Path,
                     repetitions: int = 2000, seed: int = 17,
                     claim_positive_labels: list[str] | None = None,
                     label_granularity: str = "native") -> dict:
    """Join sealed scores to native labels or explicitly requested token labels."""
    from collections import Counter

    from .datasets import _check_spans, read_jsonl, read_samples, sentence_token_labels
    from .pipeline import _validate_prediction, _verify_predictions

    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    if label_granularity not in {"native", "token"}:
        raise ValueError("Label granularity must be native or token")
    output = Path(output).resolve()
    complete = data.read_json(output / "complete.json")
    run = data.read_json(output / "run.json")
    cases = output / "predictions"
    _verify_predictions(cases)
    if (data.file_hash(cases / "seal.json") != complete["prediction_seal_sha256"]
            or data.file_hash(output / "run.json") != complete["run_sha256"]):
        raise ValueError("Completed prediction bindings differ")
    input_path = Path(run["input"])
    prepared = input_path.parent
    manifest = data.read_json(prepared / "manifest.json")
    if (data.file_hash(input_path) != run["input_sha256"]
            or data.file_hash(prepared / "manifest.json") != run["manifest_sha256"]):
        raise ValueError("Prepared inputs differ from the prediction run")
    annotations = Path(annotations) if annotations else prepared / "annotations.jsonl"
    if data.file_hash(annotations) != manifest["annotations_sha256"]:
        raise ValueError("Annotation provenance differs from the prepared manifest")
    samples = {sample.id: sample for sample in read_samples(input_path)}
    gold = {}
    for row in read_jsonl(annotations):
        if row["id"] in gold:
            raise ValueError("Duplicate annotation ID")
        gold[row["id"]] = row
    if set(gold) != set(samples) or complete["samples"] != len(samples):
        raise ValueError("Predictions, inputs and annotations require an exact sample join")
    if set(data.read_json(cases / "seal.json")) != {
            data.prediction_path(cases, key).name for key in samples}:
        raise ValueError("Prediction files differ from prepared sample IDs")
    granularities = {row["granularity"] for row in gold.values()}
    if len(granularities) != 1:
        raise ValueError("Evaluate one annotation granularity at a time")
    native_granularity = granularities.pop()
    project_sentences = label_granularity == "token" and native_granularity == "sentence"
    granularity = "token" if label_granularity == "token" else native_granularity
    if label_granularity == "token" and native_granularity == "claim":
        raise ValueError("Claim annotations do not provide token labels")
    if granularity == "claim" and not claim_positive_labels:
        raise ValueError("Claim evaluation needs an explicit binary label projection")
    units, token_dataset, unknown, unalignable = [], [], 0, 0
    exclusions, boundary_tokens = Counter(), 0
    for key, sample in samples.items():
        annotation = gold[key]
        prediction = data.read_json(data.prediction_path(cases, key))
        response_hash = data.text_hash(sample.response)
        if (annotation["response_sha256"] != response_hash
                or prediction["response_sha256"] != response_hash
                or prediction["id"] != key or prediction["source_id"] != sample.source_id):
            raise ValueError("Prediction/annotation response identity differs")
        tokens = _validate_prediction(prediction, sample)
        common = {"id": key, "source_id": sample.source_id, "generator": sample.generator}
        if project_sentences:
            _check_spans(sample.response, annotation["spans"])
            projected = sentence_token_labels(annotation, tokens)
            unknown += sum(span["label"] is None for span in annotation["spans"])
            unalignable += annotation["coverage"] == "unalignable"
            for index, (token, target) in enumerate(zip(tokens, projected, strict=True)):
                identity = {"token_index": index, **{name: token[name] for name in
                            ("token_id", "text", "start", "end")}}
                label_row = {**common, "response_sha256": response_hash, **identity,
                             "label": target["label"],
                             "native_sentence_indices": target["sentence_indices"],
                             "exclusion": target["exclusion"]}
                token_dataset.append(label_row)
                boundary_tokens += len(target["sentence_indices"]) > 1
                if target["label"] is None:
                    exclusions[target["exclusion"]] += 1
                else:
                    units.append({**common, **identity, "label": target["label"],
                                  "score": token["score"]})
        elif granularity == "token":
            if annotation["coverage"] != "complete":
                raise ValueError("Token labels require complete native annotation coverage")
            for row in tokens:
                label = any(row["start"] < span["end"] and row["end"] > span["start"]
                            for span in annotation["spans"] if span["label"] == 1)
                units.append({**common, "start": row["start"], "end": row["end"],
                              "label": label, "score": row["score"]})
        elif granularity == "sentence":
            if annotation["coverage"] == "unalignable":
                unalignable += 1
            for span in annotation["spans"]:
                if span["label"] is None:
                    unknown += 1
                    continue
                if span["label"] not in (0, 1) or not 0 <= span["start"] < span["end"] <= len(sample.response):
                    raise ValueError("Invalid native sentence label or offset")
                units.append({**common, "start": span["start"], "end": span["end"],
                              "label": bool(span["label"]),
                              "score": _span_score(tokens, span["start"], span["end"])})
        elif granularity == "claim":
            label = annotation["native_label"]
            if label is None:
                unknown += 1
                continue
            units.append({**common, "start": 0, "end": len(sample.response),
                          "label": label in claim_positive_labels,
                          "native_label": label,
                          "score": _span_score(tokens, 0, len(sample.response))})
        else:
            raise ValueError(f"Unsupported annotation granularity: {granularity}")
    if not units:
        raise ValueError("No evaluable native annotations remain")
    labels = np.asarray([row["label"] for row in units], dtype=bool)
    scores = np.asarray([row["score"] for row in units])
    groups = np.asarray([row["source_id"] for row in units])
    point = ranking(labels, scores)
    point["units"] = point.pop("tokens")
    by_generator = {}
    for generator in sorted({row["generator"] for row in units}):
        mask = np.asarray([row["generator"] == generator for row in units])
        value = ranking(labels[mask], scores[mask])
        value["units"] = value.pop("tokens")
        by_generator[generator] = value
    report = {
        "dataset": manifest["dataset"], "split": manifest["split"],
        "granularity": granularity, "recipe": run["task"], "samples_scored": len(samples),
        "samples_evaluated": len({row["id"] for row in units}),
        "groups_evaluated": len(set(groups)), "group_type": manifest["bootstrap_type"],
        "coverage": dict(Counter(row["coverage"] for row in gold.values())),
        "unknown_units_excluded": unknown, "unalignable_responses_excluded": unalignable,
        "aggregation": "native token score" if granularity == "token" else
                       "character-overlap-weighted mean of final token scores",
        "claim_positive_labels": claim_positive_labels, "metrics": point,
        "by_generator": by_generator,
        "bootstrap": {"seed": seed, "repetitions": repetitions,
                      "unit": manifest["bootstrap_type"],
                      **_cluster_intervals(labels, scores, groups, repetitions, seed)},
        "input_sha256": run["input_sha256"], "annotation_sha256": data.file_hash(annotations),
        "prediction_seal_sha256": complete["prediction_seal_sha256"],
        "evaluation_code": {str(path.resolve()): data.file_hash(path) for path in
                            (Path(__file__), Path(__file__).with_name("data.py"),
                             Path(__file__).with_name("datasets.py"))},
    }
    if project_sentences:
        report.update(
            native_granularity=native_granularity,
            label_provenance="sentence-derived token pseudo-labels",
            aggregation="direct final token score (no aggregation)",
            label_projection={"positive": "any positive sentence overlap",
                              "negative": "fully covered by known negative sentences",
                              "ignored": "otherwise undefined, uncovered or unalignable"},
            tokens_scored=len(token_dataset), tokens_evaluated=len(units),
            tokens_excluded=dict(exclusions), tokens_crossing_sentence_boundaries=boundary_tokens,
            unknown_units_excluded=exclusions["undefined"], native_unknown_sentences=unknown,
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data.write_json(destination / "evaluation.json", report)
    with (destination / "units.jsonl").open("x", encoding="utf-8") as stream:
        for row in units:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if project_sentences:
        with (destination / "token_labels.jsonl").open("x", encoding="utf-8") as stream:
            for row in token_dataset:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    data.write_json(destination / "seal.json", {
        path.name: data.file_hash(path) for path in sorted(destination.iterdir())
    })
    return report
