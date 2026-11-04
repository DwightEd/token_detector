"""Synthetic M0 boundaries/identities; no natural graph effects or tokenizer."""

import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from experiments import head_carrier_frontend as frontend
from experiments.data import file_hash


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """Tiny physical dimensions isolate gates; production specs remain fixed."""
    specs = {"00012": (12, 7, 18, 0), "00013": (12, 6, 17, 1)}
    monkeypatch.setattr(frontend, "CASE_SPECS", specs)
    graphs, samples = tmp_path / "graphs", tmp_path / "samples"
    graphs.mkdir(); samples.mkdir()
    prefix = ["<s>", "User ", "A 1", " minutes", " B 2", " minutes",
              " B 2", " minutes", " D 3", " minutes", " end", "Assistant "]
    prompt = "".join(prefix[2:11])
    chat = "".join(prefix)
    shift = chat.index(prompt)
    pids = np.array([1, 2, 301, 4520, 302, 4520, 302, 4520, 303, 4520, 9, 10])
    manifest, settings, qcases, proofs, sample_rows = {}, {"input_sha256": {}}, {}, {}, []
    for case_id, (p, t, n, seed) in specs.items():
        pieces = (["Cook ", "1", ".", " 2", " times", " 30", "<eos>"] if seed == 0
                  else ["Cook ", "2", ".", "3", " done", "<eos>"])
        rids = np.arange(100, 100 + t, dtype=np.int64)
        rids[-1] = 99
        ids = np.r_[pids, rids]
        special = np.zeros(p + t, dtype=bool)
        special[[0, -1]] = True
        text = prefix + pieces
        response = "".join(pieces[:-1])
        source = np.zeros(p, dtype=bool); source[2:11] = True
        roles = np.full(n, 2, dtype=np.int8)
        roles[:p] = np.where(source, 0, 1)
        roles[special[:-1]] = 3
        directory = graphs / case_id; directory.mkdir()
        arrays = {"input_ids": ids[:-1], "target_ids": rids, "roles": roles,
                  "prompt_length": np.array(p),
                  "attention": np.zeros((32, 32, n, n), np.float32),
                  "values": np.zeros((32, n, 8, 128), np.float32)}
        for name, value in arrays.items():
            np.save(directory / (name + ".npy"), value)
        source_hash = hashlib.sha256(source.tobytes()).hexdigest()
        metadata = {"trace": case_id + ".npz", "source_id": "14375", "seed": seed,
                    "response": response, "input_token_text": text[:-1],
                    "source_mask_sha256": source_hash}
        _json(directory / "metadata.json", metadata)
        for path in directory.iterdir():
            manifest[f"{case_id}/{path.name}"] = file_hash(path)
        np.savez(samples / (case_id + ".npz"), token_ids=ids, token_text=np.array(text),
                 special_mask=special, prompt_length=np.array(p),
                 attention=np.array([123.], np.float32))  # must never be accessed
        settings["input_sha256"][case_id + ".npz"] = file_hash(samples / (case_id + ".npz"))
        sample_rows.append({"source_id": "14375", "seed": seed, "response": response,
                            "tokens": t, "trace": case_id + ".npz"})
        qcases[case_id] = {"passed": True, "P": p, "T": t, "input_nodes": n,
                           "source_mask_sha256": source_hash}
        proofs[case_id] = {"metadata_sha256": manifest[f"{case_id}/metadata.json"],
                          "input_ids_file_sha256": manifest[f"{case_id}/input_ids.npy"],
                          "roles_file_sha256": manifest[f"{case_id}/roles.npy"]}
    _json(graphs / "manifest.json", manifest); _json(graphs / "settings.json", settings)
    (samples / "prompts.jsonl").write_text(json.dumps({"source_id": "14375", "prompt": prompt}) + "\n")
    (samples / "samples.jsonl").write_text("\n".join(json.dumps(v) for v in sample_rows) + "\n")
    qualification = {"passed": True, "original_manifest_sha256": file_hash(graphs / "manifest.json"),
                     "settings_sha256": file_hash(graphs / "settings.json"), "cases": qcases}
    qp = tmp_path / "qualification.json"; _json(qp, qualification)
    ends = np.cumsum([0] + [len(v) for v in prefix]).tolist()
    records = []
    for position in (3, 5, 7, 9):
        context = prefix[position - 1].lstrip() + prefix[position]
        start = ends[position - 1] + len(prefix[position - 1]) - len(prefix[position - 1].lstrip()) - shift
        unit_start = ends[position] + 1 - shift
        action_end = start + 1
        number_start = action_end + 1
        records.append({"key": f"fact{position}", "absolute_source_token": position,
                        "unit_token_id": 4520, "role": 0, "word": "minutes",
                        "complete_single_token_unit": True,
                        "eligible_for_minute_association_probe": True,
                        "local_relation_closed_at_or_before_carrier": True,
                        "raw_prompt_unit_span": [unit_start, unit_start + 7],
                        "chat_unit_span": [shift + unit_start, shift + unit_start + 7],
                        "exact_context": context,
                        "raw_prompt_context_span": [start, unit_start + 7],
                        "raw_prompt_action_span": [start, action_end],
                        "raw_prompt_numeric_span": [number_start, number_start + 1],
                        "action_token_positions": [position - 1],
                        "numeric_token_positions": [position - 1]})
    pairs = [{"id": f"minutes-{a}-{b}", "carrier_positions": [a, b],
              "carrier_keys": [f"fact{a}", f"fact{b}"], "unit_token_id": 4520,
              "eligible": True, "kind": "EQUIV" if (a, b) == (5, 7) else "SEM",
              "reason": "frozen synthetic source licence"}
             for a, b in combinations((3, 5, 7, 9), 2)]
    licence = {"schema": "source14375-duration-carrier-license-v1", "source_id": "14375",
               "source_only": True, "P": 12, "raw_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "chat_prompt_shift": shift, "all_source_time_unit_occurrences": 4,
               "graph_manifest_sha256": file_hash(graphs / "manifest.json"),
               "prompt_file_sha256": file_hash(samples / "prompts.jsonl"),
               "graph_source_prompt_prefixes": proofs, "records": records, "pairs": pairs,
               "genuine_duplicate_instruction_groups": {"repeatB": [5, 7]},
               "strict_minute_eligibility": {"definition": "local synthetic facts only"}}
    lp = tmp_path / "licence.json"; _json(lp, licence)
    return graphs, samples, lp, qp


def test_complete_gate_retains_all_targets_pairs_and_never_reads_effect_arrays(inputs, monkeypatch):
    real_load = np.load

    def guarded_load(path, *args, **kwargs):
        assert Path(path).name not in {"attention.npy", "values.npy"}
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    real_get = np.lib.npyio.NpzFile.__getitem__

    def guarded_member(self, key):
        assert key in frontend.SAMPLE_FIELDS
        return real_get(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded_member)
    result = frontend.prepare_frontend(*inputs)
    assert result["gate_passed"] and result["F"] == 6 and result["Q"] == 3
    assert result["tuple_count"] == 32 * 48 * 6 * 3
    assert [len(v["all_targets"]) for v in result["cases"]] == [7, 6]
    assert result["cases"][0]["all_targets"][1]["q"] == 12
    assert result["cases"][0]["all_targets"][3]["selected"] is False
    assert result["cases"][0]["all_targets"][-1]["reason"] == "special_or_empty_original_target"
    assert all(set(("pair_id", "kind", "positions", "token_id")) <= set(v)
               for v in result["eligible_pairs"])
    assert all(Path(v["path"]).is_absolute() for v in result["input_files"])
    assert any(v["hash_mode"] == "qualified_manifest_not_rehashed" for v in result["input_files"])
    assert result["no_effect_reads"] and not result["model_calls"] and not result["GT_reads"]


def test_current_target_letters_and_future_decimal_context_do_not_enter_prefix():
    pieces = ["1 Cook", ".", "5", " times ", "7"]
    rows = frontend._target_table(np.arange(5), pieces, np.zeros(5, bool), "".join(pieces), 10)
    assert [r["selected"] for r in rows] == [False, False, False, False, True]
    assert [r["q"] for r in rows] == list(range(9, 14))


@pytest.mark.parametrize("response,pieces", [("Cook 2", ["Cook ", "3"]), ("Cook \ufffd2", ["Cook ", "\ufffd2"])])
def test_unrecoverable_text_is_explicit_unknown_not_retokenized(response, pieces):
    rows = frontend._target_table(np.array([10, 11]), pieces, np.zeros(2, bool), response, 5)
    assert len(rows) == 2 and all(r["status"] == "unknown" and not r["selected"] for r in rows)


def test_no_equiv_exits_with_complete_m0_not_mechanism_fallback(inputs):
    licence = json.loads(inputs[2].read_text())
    for pair in licence["pairs"]: pair["kind"] = "SEM"
    _json(inputs[2], licence)
    result = frontend.prepare_frontend(*inputs)
    assert result["gate_passed"] is False and result["reason"] == "no_licensed_EQUIV_pair"
    assert len(result["pairs"]) == 6 and len(result["cases"]) == 2


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d["pairs"].pop(), "Incomplete all-same-ID"),
    (lambda d: d["records"][0].pop("eligible_for_minute_association_probe"), "Missing strict"),
    (lambda d: d["records"][0].update(action_token_positions=[3]), "span/token"),
    (lambda d: d["records"][0].update(raw_prompt_unit_span=[0, 7]), "unit text/span"),
])
def test_unknown_carrier_or_missing_inventory_is_never_backfilled(inputs, mutation, reason):
    licence = json.loads(inputs[2].read_text())
    mutation(licence); _json(inputs[2], licence)
    with pytest.raises(ValueError, match=reason): frontend.prepare_frontend(*inputs)


def test_original_graph_axis_drift_rejected(inputs):
    graph = inputs[0]
    path = graph / "00012/target_ids.npy"
    value = np.load(path); value[1] += 1; np.save(path, value)
    with pytest.raises(ValueError, match="metadata SHA"):
        frontend.prepare_frontend(*inputs)


def test_missing_big_digest_never_falls_back_to_hashing_npz(inputs, monkeypatch):
    settings_path = inputs[0] / "settings.json"
    settings = json.loads(settings_path.read_text()); settings["input_sha256"].clear()
    _json(settings_path, settings)
    qualification = json.loads(inputs[3].read_text())
    qualification["settings_sha256"] = file_hash(settings_path); _json(inputs[3], qualification)
    real_hash = frontend.file_hash

    def guarded_hash(path):
        assert Path(path).suffix != ".npz"
        return real_hash(path)

    monkeypatch.setattr(frontend, "file_hash", guarded_hash)
    with pytest.raises(ValueError, match="Missing qualified sample archive SHA"):
        frontend.prepare_frontend(*inputs)


def test_oversized_declared_npz_shape_rejected_before_loading(tmp_path):
    path = tmp_path / "bad.npz"
    import io
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(member, {"descr": "<i8", "fortran_order": False,
                                                "shape": (10**12,)})
    with zipfile.ZipFile(path, "w") as archive:
        for name in frontend.SAMPLE_FIELDS:
            archive.writestr(name + ".npy", member.getvalue())
    with pytest.raises(ValueError, match="NPZ metadata member"):
        frontend._small_sample(path)


def test_original_numpy_axis_reuse_does_not_import_torch_or_transformers():
    code = ("import sys; from experiments.head_carrier_frontend import text_units, answer_axis; "
            "assert text_units(['Cook ', '2']) == [{'start': 0, 'stop': 2, 'text_start': 0, 'text_stop': 2}]; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   timeout=30, env=environment)
