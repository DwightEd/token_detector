"""Source-licence and original-axis gates for the finite head/carrier pilot.

This M0 reader never decodes attention, values, logits, hidden states or labels.
It consumes a manually frozen source-only licence; it does not infer semantics.
"""

from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from experiments.data import file_hash, read_json
from myproject.inputs import text_units
from myproject.measure import answer_axis


CASE_SPECS = {"00012": (535, 196, 730, 0), "00013": (535, 147, 681, 1)}
SOURCE_ID = "14375"
SMALL_BYTES = 16 * 1024**2
SAMPLE_FIELDS = ("token_ids", "token_text", "special_mask", "prompt_length")
_DIGIT = re.compile(r"[0-9]")
_LETTER = re.compile(r"[A-Za-z]")
_BOUNDARY = re.compile(r"[.!?\r\n]")


def _require(test, reason):
    if not test:
        raise ValueError(reason)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    _require(path.is_relative_to(root) and path.is_file(), "Missing or escaping input: " + name)
    return path


def _pin(path, digest=None):
    path = Path(path).resolve()
    before = path.stat()
    actual = file_hash(path) if digest is None else digest
    _require(_sha(actual), "Invalid declared file digest")
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             "Input changed while pinning")
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": actual, "hash_mode": ("actual_small_file" if digest is None
                                               else "qualified_manifest_not_rehashed")}


def _small_npy(path):
    _require(path.stat().st_size <= SMALL_BYTES, "Axis NPY exceeds metadata budget")
    _header(path)
    value = np.load(path, allow_pickle=False)
    _require(not value.dtype.hasobject and value.nbytes <= SMALL_BYTES,
             "Unsafe axis array")
    return value


def _small_sample(path):
    """Read only four named small members, never the archive's large arrays."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Duplicate NPZ member")
        for field in SAMPLE_FIELDS:
            info = archive.getinfo(field + ".npy")
            _require(info.file_size <= SMALL_BYTES, "NPZ metadata member exceeds budget")
            with archive.open(info) as stream:
                shape, dtype = _read_header(stream)
                _require(stream.tell() + math.prod(shape) * dtype.itemsize == info.file_size,
                         "Truncated or extended NPZ metadata member")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in SAMPLE_FIELDS}
    _require(all(not value.dtype.hasobject and value.nbytes <= SMALL_BYTES
                 for value in result.values()), "Unsafe NPZ metadata")
    return result


def _read_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported NPY version")
    _require(not fortran and not dtype.hasobject, "Unsafe graph NPY layout")
    _require(all(_integer(v) and v >= 0 for v in shape), "Invalid array shape")
    return shape, dtype


def _header(path):
    """Inspect a plain NPY header and physical length without mapping its values."""
    with path.open("rb") as stream:
        shape, dtype = _read_header(stream)
        start = stream.tell()
    payload = math.prod(shape) * dtype.itemsize
    _require(path.stat().st_size == start + payload, "Truncated or extended graph NPY")
    return {"shape": list(shape), "dtype": str(dtype)}


def _jsonl(path, key):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            identity = str(value[key])
            _require(identity not in result, "Duplicate metadata identity")
            result[identity] = value
    return result


def _target_table(ids, pieces, special, response, prompt_tokens):
    """Use only each consumed prefix for boundaries; chosen-y selection is posthoc."""
    count = len(ids)
    base = [{"t": t, "q": prompt_tokens + t - 1, "target_id": int(ids[t]),
             "token_text": str(pieces[t]), "selected": False, "status": "excluded",
             "reason": "", "prefix_char_end": None} for t in range(count)]
    rendered = ["" if bool(marked) else str(piece)
                for piece, marked in zip(pieces, special, strict=True)]
    if "".join(rendered) != response or any("\ufffd" in part for part in rendered):
        for row in base:
            row.update(status="unknown", reason="unambiguous_original_text_alignment_unavailable")
        return base
    ends = np.cumsum([0] + [len(part) for part in rendered]).tolist()
    offsets = list(zip(ends[:-1], ends[1:], strict=True))
    special_ids = np.asarray(ids)[np.asarray(special)].tolist()
    encoded = {"answer_ids": ids, "offsets": offsets, "special_ids": special_ids,
               "units": text_units(rendered)}
    valid = set(answer_axis(encoded)["target"].tolist())
    for row, piece in zip(base, rendered, strict=True):
        t = row["t"]
        row["prefix_char_end"] = ends[t]
        row["offset"] = list(offsets[t])
        if t not in valid:
            row["reason"] = "special_or_empty_original_target"
        elif not _DIGIT.search(piece):
            row["reason"] = "chosen_target_has_no_ASCII_digit"
        else:
            # Only the already consumed prefix is inspected. In particular a
            # decimal separator is not repaired by peeking at its right side.
            prefix = response[:ends[t]]
            sentence = _BOUNDARY.split(prefix)[-1]
            if _LETTER.search(sentence):
                row.update(selected=True, status="selected",
                           reason="chosen_digit_and_past_sentence_ASCII_letter")
            else:
                row["reason"] = "past_sentence_has_no_ASCII_letter"
    return base


def _span(prompt, value, name):
    _require(isinstance(value, list) and len(value) == 2
             and all(_integer(v) for v in value), "Invalid licence " + name)
    left, right = value
    _require(0 <= left < right <= len(prompt), "Out-of-bounds licence " + name)
    return left, right


def _licence_pairs(licence, prompt, prompt_ids, prompt_text, source_mask):
    """Validate provenance and explicit licence flags, without assigning relations."""
    _require(licence.get("schema") == "source14375-duration-carrier-license-v1"
             and licence.get("source_id") == SOURCE_ID and licence.get("source_only") is True,
             "Wrong source-only licence")
    _require(hashlib.sha256(prompt.encode()).hexdigest() == licence.get("raw_prompt_sha256"),
             "Licence prompt identity differs")
    chat = "".join(prompt_text)
    shift = licence.get("chat_prompt_shift")
    _require(_integer(shift) and chat.count(prompt) == 1 and chat.index(prompt) == shift,
             "Source prompt cannot be joined to saved chat text")
    ends = np.cumsum([0] + [len(part) for part in prompt_text]).tolist()
    records = licence.get("records")
    _require(isinstance(records, list) and len(records) == licence.get("all_source_time_unit_occurrences"),
             "Incomplete source occurrence inventory")
    by_position, by_key, groups = {}, {}, {}
    for record in records:
        position, key, token_id = (record.get("absolute_source_token"), record.get("key"),
                                   record.get("unit_token_id"))
        _require(_integer(position) and 0 <= position < len(prompt_ids)
                 and _integer(token_id) and isinstance(key, str)
                 and position not in by_position and key not in by_key,
                 "Duplicate or invalid source carrier")
        _require(source_mask[position] and int(prompt_ids[position]) == token_id
                 and record.get("role") == 0 and record.get("complete_single_token_unit") is True,
                 "Carrier is not the declared original source token")
        _require(isinstance(record.get("eligible_for_minute_association_probe"), bool),
                 "Missing strict source eligibility; core_closed is insufficient")
        left, right = _span(prompt, record.get("raw_prompt_unit_span"), "unit span")
        _require(prompt[left:right] == record.get("word")
                 and record.get("chat_unit_span") == [shift + left, shift + right]
                 and ends[position] <= shift + left < shift + right <= ends[position + 1],
                 "Carrier unit text/span differs")
        lo, hi = _span(prompt, record.get("raw_prompt_context_span"), "context span")
        _require(prompt[lo:hi] == record.get("exact_context") and lo <= left < right <= hi,
                 "Source context span differs")
        for field in ("raw_prompt_action_span", "raw_prompt_numeric_span"):
            a, b = _span(prompt, record.get(field), field)
            _require(lo <= a < b <= hi, "Source fact span escapes its context")
            positions_field = ("action_token_positions" if "action" in field
                               else "numeric_token_positions")
            overlaps = [i for i in range(len(prompt_ids))
                        if ends[i] < shift + b and ends[i + 1] > shift + a]
            _require(record.get(positions_field) == overlaps,
                     "Source fact span/token positions differ")
        if record["eligible_for_minute_association_probe"]:
            _require(record.get("local_relation_closed_at_or_before_carrier") is True,
                     "Strict carrier lacks local relation closure")
            for field in ("action_token_positions", "numeric_token_positions"):
                positions = record.get(field)
                _require(isinstance(positions, list) and positions
                         and all(_integer(v) and 0 <= v <= position for v in positions),
                         "Licensed relation requires a future source token")
        by_position[position], by_key[key] = record, record
        groups.setdefault(token_id, []).append(position)
    expected = {tuple(pair) for positions in groups.values()
                for pair in combinations(sorted(positions), 2)}
    pairs, seen, pair_ids = [], set(), set()
    duplicates = list(licence.get("genuine_duplicate_instruction_groups", {}).values())
    for pair in licence.get("pairs", []):
        positions = pair.get("carrier_positions")
        _require(isinstance(positions, list) and len(positions) == 2
                 and all(_integer(v) for v in positions), "Invalid source pair positions")
        positions = tuple(positions)
        identity = pair.get("id")
        _require(positions in expected and positions not in seen and isinstance(identity, str)
                 and identity not in pair_ids, "Missing/duplicate/unordered source pair")
        seen.add(positions); pair_ids.add(identity)
        records_pair = [by_position[p] for p in positions]
        _require(pair.get("unit_token_id") == records_pair[0]["unit_token_id"]
                 and pair.get("carrier_keys") == [v["key"] for v in records_pair],
                 "Pair source identities differ")
        eligible, kind = pair.get("eligible"), pair.get("kind")
        _require(isinstance(eligible, bool) and isinstance(pair.get("reason"), str),
                 "Missing pair eligibility provenance")
        if eligible:
            _require(kind in {"SEM", "EQUIV"} and all(
                v["eligible_for_minute_association_probe"] for v in records_pair),
                "Pair promotes an unlicensed carrier")
            if kind == "EQUIV":
                _require(any(all(p in group for p in positions) for group in duplicates),
                         "EQUIV lacks explicit duplicate-instruction licence")
        else:
            _require(kind == "UNKNOWN", "Unlicensed pair must retain UNKNOWN")
        pairs.append({"pair_id": identity, "kind": kind, "positions": list(positions),
                      "token_id": pair["unit_token_id"], "eligible": eligible,
                      "reason": pair["reason"], "carrier_keys": pair["carrier_keys"]})
    _require(seen == expected, "Incomplete all-same-ID source pair inventory")
    return pairs


def prepare_frontend(graph_root, samples_root, licence_path, qualification_path):
    """Freeze M0 without A/V effects, tokenization, model calls or score creation.

    Normal semantic/coverage/budget refusal returns ``gate_passed=False`` with
    complete tables. Corrupt or mismatched physical identities raise ValueError.
    Large-file digests are reused qualified manifest witnesses, not rehashed.
    """
    graph_root, samples_root = Path(graph_root).resolve(), Path(samples_root).resolve()
    licence_path, qualification_path = Path(licence_path).resolve(), Path(qualification_path).resolve()
    pins = {}

    def pin(path, digest=None):
        value = _pin(path, digest)
        pins[value["path"]] = value
        return value["sha256"]

    manifest_path, settings_path = (_path(graph_root, name) for name in ("manifest.json", "settings.json"))
    manifest_hash, settings_hash = pin(manifest_path), pin(settings_path)
    manifest, settings = read_json(manifest_path), read_json(settings_path)
    pin(licence_path); pin(qualification_path)
    licence, qualification = read_json(licence_path), read_json(qualification_path)
    _require(qualification.get("passed") is True
             and qualification.get("original_manifest_sha256") == manifest_hash
             and qualification.get("settings_sha256") == settings_hash,
             "Graph qualification or input identity differs")
    _require(licence.get("graph_manifest_sha256") == manifest_hash,
             "Licence graph manifest differs")
    prompts_path, samples_path = (_path(samples_root, name) for name in ("prompts.jsonl", "samples.jsonl"))
    prompt_digest = pin(prompts_path); pin(samples_path)
    _require(licence.get("prompt_file_sha256") == prompt_digest, "Licence prompt file differs")
    prompts, samples = _jsonl(prompts_path, "source_id"), _jsonl(samples_path, "trace")
    _require(SOURCE_ID in prompts and isinstance(prompts[SOURCE_ID].get("prompt"), str),
             "Missing original source prompt")
    cases, shared = [], None
    for case_id, (p, t, n, seed) in CASE_SPECS.items():
        trace = case_id + ".npz"
        sample = samples.get(trace)
        _require(sample and str(sample.get("source_id")) == SOURCE_ID
                 and sample.get("seed") == seed and sample.get("tokens") == t,
                 "Original sample identity differs")
        npz_path = _path(samples_root, trace)
        sample_digest = settings.get("input_sha256", {}).get(trace)
        _require(_sha(sample_digest), "Missing qualified sample archive SHA")
        pin(npz_path, sample_digest)
        original = _small_sample(npz_path)
        ids, pieces, special = (original[k] for k in SAMPLE_FIELDS[:3])
        _require(ids.shape == (p + t,) and ids.dtype.kind in "iu"
                 and pieces.shape == ids.shape and pieces.dtype.kind == "U"
                 and special.shape == ids.shape and special.dtype == np.bool_
                 and original["prompt_length"].shape == ()
                 and original["prompt_length"].dtype.kind in "iu"
                 and int(original["prompt_length"]) == p and n == p + t - 1,
                 "Original sample token axes differ")
        arrays = {}
        for field in ("input_ids", "target_ids", "roles", "prompt_length"):
            relative = case_id + "/" + field + ".npy"
            path = _path(graph_root, relative)
            _require(pin(path) == manifest.get(relative), "Graph metadata SHA differs")
            arrays[field] = _small_npy(path)
        input_ids, target_ids, roles = (arrays[k] for k in ("input_ids", "target_ids", "roles"))
        _require(input_ids.shape == (n,) and target_ids.shape == (t,)
                 and input_ids.dtype.kind in "iu" and target_ids.dtype.kind in "iu"
                 and roles.shape == (n,) and roles.dtype == np.int8
                 and arrays["prompt_length"].shape == ()
                 and arrays["prompt_length"].dtype.kind in "iu"
                 and int(arrays["prompt_length"]) == p
                 and np.array_equal(input_ids, ids[:-1])
                 and np.array_equal(target_ids, ids[p:]), "Fresh/original token axes differ")
        source_mask = roles[:p] == 0
        expected_roles = np.full(n, 2, dtype=np.int8)
        expected_roles[:p] = np.where(source_mask, 0, 1)
        expected_roles[special[:-1]] = 3
        _require(np.array_equal(roles, expected_roles), "Fresh source/history/special roles differ")
        metadata_path = _path(graph_root, case_id + "/metadata.json")
        _require(pin(metadata_path) == manifest.get(case_id + "/metadata.json"),
                 "Graph case metadata SHA differs")
        metadata = read_json(metadata_path)
        text = pieces.tolist()
        source_hash = hashlib.sha256(source_mask.tobytes()).hexdigest()
        qcase = qualification.get("cases", {}).get(case_id, {})
        _require(metadata.get("trace") == trace and str(metadata.get("source_id")) == SOURCE_ID
                 and metadata.get("seed") == seed and metadata.get("response") == sample.get("response")
                 and metadata.get("input_token_text") == text[:-1]
                 and metadata.get("source_mask_sha256") == source_hash
                 and qcase.get("passed") is True and qcase.get("P") == p and qcase.get("T") == t
                 and qcase.get("input_nodes") == n and qcase.get("source_mask_sha256") == source_hash,
                 "Qualified source/text/axis metadata differs")
        proof = licence.get("graph_source_prompt_prefixes", {}).get(case_id, {})
        _require(proof.get("metadata_sha256") == manifest[case_id + "/metadata.json"]
                 and proof.get("input_ids_file_sha256") == manifest[case_id + "/input_ids.npy"]
                 and proof.get("roles_file_sha256") == manifest[case_id + "/roles.npy"]
                 and licence.get("P") == p, "Source licence prefix proof differs")
        for field, shape in (("attention", [32, 32, n, n]), ("values", [32, n, 8, 128])):
            relative = case_id + "/" + field + ".npy"
            path = _path(graph_root, relative)
            header = _header(path)
            _require(header == {"shape": shape, "dtype": "float32"}, "Graph physical schema differs")
            digest = manifest.get(relative)
            _require(_sha(digest), "Missing qualified graph tensor SHA")
            pin(path, digest)
        prefix = (ids[:p].tolist(), text[:p], source_mask.tolist())
        _require(shared is None or shared == prefix, "Two cases do not share the original source prefix")
        shared = prefix
        rows = _target_table(target_ids, pieces[p:], special[p:], sample["response"], p)
        cases.append({"case_id": case_id, "source_id": SOURCE_ID, "P": p, "T": t, "N": n,
                      "all_targets": rows, "selected_q": [row["q"] for row in rows if row["selected"]],
                      "source_mask_sha256": source_hash, "source_positions": np.flatnonzero(source_mask).tolist()})
    pairs = _licence_pairs(licence, prompts[SOURCE_ID]["prompt"], *shared)
    eligible = [pair for pair in pairs if pair["eligible"]]
    f, q = len(eligible), sum(len(case["selected_q"]) for case in cases)
    reasons = []
    if not any(pair["kind"] == "SEM" for pair in eligible): reasons.append("no_licensed_SEM_pair")
    if not any(pair["kind"] == "EQUIV" for pair in eligible): reasons.append("no_licensed_EQUIV_pair")
    if f > 8: reasons.append("source_pair_budget_exceeded")
    if not q: reasons.append("no_unambiguous_selected_original_query")
    if q > 343 or 32 * 48 * f * q > 1_000_000: reasons.append("tuple_or_query_budget_exceeded")
    # All qualified carriers are source nodes; verify causal visibility explicitly.
    _require(all(position <= row["q"] for pair in eligible for position in pair["positions"]
                 for case in cases for row in case["all_targets"] if row["selected"]),
             "Carrier is not visible at an original query")
    for item in pins.values():
        path = Path(item["path"])
        stat = path.stat()
        _require((stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]),
                 "Input changed during M0")
        if item["hash_mode"] == "actual_small_file":
            _require(file_hash(path) == item["sha256"], "Metadata changed during M0")
    return {"schema": "head-carrier-frontend-v1", "gate_passed": not reasons,
            "reason": reasons[0] if reasons else "qualified_M0", "reasons": reasons,
            "source_id": SOURCE_ID, "cases": cases, "pairs": pairs, "eligible_pairs": eligible,
            "F": f, "Q": q, "tuple_count": 32 * 48 * f * q,
            "source_licence_scope": licence.get("strict_minute_eligibility", {}).get("definition"),
            "input_files": list(pins.values()), "selection_is_retrospective": True,
            "no_effect_reads": True, "model_calls": 0, "tokenizer_calls": 0,
            "risk_scores": 0, "GT_reads": 0}
