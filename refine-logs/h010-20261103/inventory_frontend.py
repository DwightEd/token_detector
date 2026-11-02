"""CPU NER/POS/parser engineering probe and inference-input-only span inventory."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from collections import Counter
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
ENV = Path("D:/projects/research/envs/token-detector-nlp")
INPUTS = ROOT.parents[1] / "outputs/research-goal-20261102/case_panel_v1/inputs.jsonl"
OUTPUT = ROOT / "pilot-inventory.jsonl"
MAX_CHARACTERS = 100_000
MAX_TOKENS = 20_000
DISABLED = ["lemmatizer"]
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write(name, data):
    with (ROOT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def token_row(token):
    return {"index": token.i, "start_char": token.idx, "end_char": token.idx + len(token.text),
            "text": token.text, "pos": token.pos_, "tag": token.tag_, "dependency": token.dep_,
            "head_token_index": token.head.i}


def extract(text):
    common = {"text_sha256": text_sha(text), "characters": len(text), "offset_unit": "Python Unicode code points; half-open",
              "semantic_answerability": "unknown_not_assessed", "entities": [], "tokens": [], "sentences": []}
    if len(text) > MAX_CHARACTERS:
        return {**common, "status": "unknown", "reason": "character_budget_exceeded", "token_count": None}
    doc = nlp.make_doc(text)
    common["token_count"] = len(doc)
    if len(doc) > MAX_TOKENS:
        return {**common, "status": "unknown", "reason": "token_budget_exceeded"}
    doc = nlp(doc)
    assert doc.text == text
    assert "".join(token.text_with_ws for token in doc) == text
    occurrences = Counter()
    previous_end = 0
    for entity in doc.ents:
        assert entity.text == text[entity.start_char:entity.end_char]
        assert 0 <= entity.start_char < entity.end_char <= len(text)
        assert previous_end <= entity.start_char
        previous_end = entity.end_char
        key = (entity.label_, entity.text)
        occurrence = occurrences[key]
        occurrences[key] += 1
        common["entities"].append({"occurrence_id": len(common["entities"]), "same_value_occurrence": occurrence,
            "text": entity.text, "type": entity.label_, "start_char": entity.start_char, "end_char": entity.end_char,
            "start_token": entity.start, "end_token": entity.end,
            "root_token_index": entity.root.i, "root_pos": entity.root.pos_, "root_dependency": entity.root.dep_,
            "root_head_token_index": entity.root.head.i, "semantic_answerability": "unknown_not_assessed"})
    for token in doc:
        assert text[token.idx:token.idx + len(token.text)] == token.text
        common["tokens"].append(token_row(token))
    for sentence in doc.sents:
        assert text[sentence.start_char:sentence.end_char] == sentence.text
        common["sentences"].append({"start_char": sentence.start_char, "end_char": sentence.end_char})
    return {**common, "status": "processed", "reason": None,
            "slot_availability": "model_spans_present" if common["entities"] else "unknown_no_model_spans"}


try:
    if OUTPUT.exists() or (ROOT / "pilot-freeze.json").exists():
        raise FileExistsError("fresh pilot outputs required; no automatic rerun")
    assert Path(sys.prefix).resolve() == ENV.resolve()
    readiness = json.loads((ROOT / "readiness.json").read_text(encoding="utf-8"))
    assert readiness["status"] == "assets_installed_cpu_probe_pending"
    assert importlib.metadata.version("spacy") == "3.8.18"
    assert importlib.metadata.version("en_core_web_sm") == "3.8.0"
    # Whitelist inference fields; no manifests, evaluation_only files, labels, or scores opened.
    rows = []
    with INPUTS.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                original = json.loads(line)
                rows.append({key: original[key] for key in ("id", "source", "response")})
    assert len(rows) == 18 and len({row["id"] for row in rows}) == 18
    assert all(isinstance(row[key], str) for row in rows for key in ("id", "source", "response"))
    import spacy
    assert spacy.require_cpu()
    nlp = spacy.load("en_core_web_sm", disable=DISABLED)
    nlp.max_length = MAX_CHARACTERS
    model_package = Path(importlib.util.find_spec("en_core_web_sm").origin).parent
    identity = {"spacy": spacy.__version__, "model": "en_core_web_sm", "model_version": "3.8.0",
                "model_wheel_sha256": readiness["model"]["sha256"], "environment": str(ENV),
                "enabled_components": nlp.pipe_names, "disabled_components": DISABLED,
                "model_files_sha256": {str(p.relative_to(model_package)).replace("\\", "/"): sha(p)
                                       for p in sorted(model_package.rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
                "external_supervision": readiness["supervision"], "task_gold_used": False, "gpu_calls": 0}
    freeze = {"schema": "h010-span-pilot-freeze-v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "inputs": str(INPUTS), "inputs_sha256": sha(INPUTS), "cases": 18, "fields": ["source", "response"],
              "read_whitelist": ["id", "source", "response"], "script_sha256": sha(Path(__file__)),
              "readiness_sha256": sha(ROOT / "readiness.json"), "model_identity": identity,
              "runtime": {"device": "cpu", "processes": 1, "thread_limit": 2, "python_version": sys.version},
              "budgets": {"max_characters_per_field": MAX_CHARACTERS, "max_tokens_per_field": MAX_TOKENS,
                          "over_budget": "retain case and field as explicit unknown; no truncation/drop/windowing"},
              "scope": "Span/POS/dependency inventory only; no source-support, QA, score, AUROC, or semantic coverage assessment."}
    write("pilot-freeze.json", freeze)
    synthetic = [
        ("repeated", "Ada Lovelace met Ada Lovelace in London. Ada Lovelace returned to London."),
        ("unicode", "😀 Beyoncé visited Zürich on 3 May 2024. Cafe\u0301 Labs paid €12.50; Beyoncé received 10%."),
        ("quantity", "Acme Corp paid $25 million on June 4, 2024. Acme Corp paid $25 million again."),
        ("whitespace", "\tAlice Smith visited Paris.\n\nAlice Smith left Paris.  "),
        ("empty", ""),
    ]
    probes = [{"id": label, "text": text, **extract(text)} for label, text in synthetic]
    repeated = probes[0]["entities"]
    by_value = Counter((e["type"], e["text"]) for e in repeated)
    assert max(by_value.values(), default=0) >= 2, "repeated-output preservation engineering condition not exercised"
    assert any(e["start_char"] > 0 for e in probes[1]["entities"]), "Unicode offset condition not exercised"
    write("synthetic-probe.json", {"schema": "h010-synthetic-offset-engineering-v1", "probes": probes,
          "checks": {"raw_string_roundtrip": True, "every_entity_exact_offset": True, "every_token_exact_offset": True,
                     "repeated_detected_values_not_deduplicated": True, "unicode_and_combining_marks_unchanged": True},
          "semantic_NER_accuracy_assessed": False})
    inventories = []
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            item = {"id": row["id"], "source": extract(row["source"]), "response": extract(row["response"])}
            inventories.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
    assert sha(INPUTS) == freeze["inputs_sha256"]
    counts = {field: Counter(e["type"] for row in inventories for e in row[field]["entities"]) for field in ("source", "response")}
    summary = {"schema": "h010-span-inventory-summary-v1", "status": "inventory_complete", "cases_retained": len(inventories),
               "fields_retained": 2 * len(inventories), "processed_fields": sum(row[f]["status"] == "processed" for row in inventories for f in ("source", "response")),
               "unknown_fields": sum(row[f]["status"] == "unknown" for row in inventories for f in ("source", "response")),
               "model_spans_by_type": {key: dict(value) for key, value in counts.items()},
               "fields_without_model_spans": {f: sum(not row[f]["entities"] for row in inventories) for f in ("source", "response")},
               "max_tokens_observed": {f: max(row[f]["token_count"] or 0 for row in inventories) for f in ("source", "response")},
               "every_emitted_span_exact_offset": True, "repeated_emitted_values_preserved": True,
               "semantic_answerability_or_truth_coverage": "unknown_not_assessed", "factuality_scores_computed": 0,
               "gpu_calls": 0, "task_gold_opened": False, "external_supervision": readiness["supervision"],
               "elapsed_seconds": time.monotonic() - START, "model_identity": identity,
               "evidence_sha256": {name: sha(ROOT / name) for name in ("readiness.json", "pilot-freeze.json", "synthetic-probe.json", "pilot-inventory.jsonl")}}
    write("pilot-summary.json", summary)
    write("pilot-seal.json", {"schema": "h010-engineering-inventory-seal-v1", "files_sha256":
          {name: sha(ROOT / name) for name in ("pilot-freeze.json", "synthetic-probe.json", "pilot-inventory.jsonl", "pilot-summary.json")}})
    print(json.dumps({key: summary[key] for key in ("status", "cases_retained", "fields_retained", "unknown_fields", "model_spans_by_type", "elapsed_seconds")}))
except Exception as error:
    path = ROOT / "pilot-failure.json"
    if not path.exists():
        write(path.name, {"status": "engineering_failed_no_retry", "error_type": type(error).__name__, "error": str(error),
                         "elapsed_seconds": time.monotonic() - START, "traceback": traceback.format_exc()})
    raise
