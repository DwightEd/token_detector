"""Synthetic original axes and fake classifier only; no HF assets or labels."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, nli_dataset as ds, nli_qualification as nli


class TinyTokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False}
        return [100 + ord(c) for c in text]

    def num_special_tokens_to_add(self, pair):
        assert pair is True
        return 3

    def build_inputs_with_special_tokens(self, source, hypothesis):
        return [1] + source + [2] + hypothesis + [2]


class FakeBackend:
    def __init__(self, *, fault=None, unknown=False):
        self.model_identity = dict(model_id=nli.REPO, revision=nli.REVISION,
            model_files_sha256=nli.PINNED_SHA256, runtime=nli.HF_RUNTIME)
        self.tokenizer = None
        self.calls, self.fault, self.unknown = 0, fault, unknown
        self.execution_mode = 'dataset-fp32-batch1-v1'
        self.closed = False

    def close(self):
        self.closed = True

    def verify_frozen_identity(self):
        pass

    def prepare_pairs(self, source, claim):
        if self.tokenizer is None:
            self.tokenizer = TinyTokenizer()
        return nli._prepare_pairs(self.tokenizer, source, 'H' * 509 if self.unknown else claim)

    def classify(self, pairs):
        assert self.tokenizer is not None, 'Fresh scoring must initialize the existing tokenizer first'
        assert len(pairs) == 1
        self.calls += 1
        pair = pairs[0]
        logits = [[3., 1., 2.]]
        if self.fault == 'canary' and self.calls == 2:
            logits = [[4., 1., 2.]]
        if self.fault == 'shape' and self.calls == 3:
            logits = [[3., 1.]]
        if self.fault == 'signed-zero':
            logits = [[0. if self.calls == 1 else -0., 1., 2.]]
        return dict(logits=logits, actual_forward_passes=1,
            input_tokens=len(pair['input_ids']), padded_input_tokens=len(pair['input_ids']), seconds=.01)


@pytest.fixture
def population(tmp_path, monkeypatch):
    root = tmp_path / 'prepared'
    root.mkdir()
    samples, records = [], []
    for i in range(2):
        response = dict(answer_ids=[101, 102, 103, 104], offsets=[[0, 1], [1, 2], [2, 4], [4, 5]],
            text='A. B.', token_text=['A', '.', ' B', '.'], special_ids=[], units=[
                dict(start=0, stop=1, text_start=0, text_stop=2),
                dict(start=1, stop=2, text_start=0, text_stop=2),
                dict(start=2, stop=4, text_start=2, text_stop=4)])
        path = root / f'response-{i}.json'
        data.write_json(path, response)
        source = 'X' * 600
        samples.append(dict(id=str(i), source_id='source', split='test', task='Data2txt',
            prompt='Source: ' + source, source=source, response=response['text']))
        claims = []
        for j, (span, units, targets, chars) in enumerate((([0, 2], [0, 1], [0, 1], [0, 2]),
                                                         ([2, 4], [2], [2, 3], [2, 5]))):
            literal = response['text'][slice(*chars)]
            claims.append(dict(claim_index=j, original_sentence_token_span=span,
                original_measurement_unit_ids=units, character_span=chars, claim_text=literal,
                claim_sha256=data.text_hash(literal), target_indices=targets))
        records.append(dict(id=str(i), source_id='source', source_sha256=data.text_hash(source),
            response_sha256=data.text_hash(response['text']), primitive_response_path=str(path),
            primitive_response_sha256=data.file_hash(path), claims=claims))
    inputs = root / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(s) + '\n' for s in samples), encoding='utf-8')
    data.write_json(root / 'manifest.json', dict(inputs_sha256=data.file_hash(inputs), samples=2))
    inventory = root / 'inventory.json'
    data.write_json(inventory, dict(all900_original_axes_and_units_match=True, records=records))
    protocol = deepcopy(ds.DEFAULT_PROTOCOL)
    protocol.update(expected_samples=2, expected_sources=1, expected_sentence_units=4,
        expected_measurement_units=6, expected_original_valid_targets=8,
        original_inputs_sha256=data.file_hash(inputs),
        original_manifest_sha256=data.file_hash(root / 'manifest.json'),
        inventory_sha256=data.file_hash(inventory))
    monkeypatch.setattr(ds, 'DEFAULT_PROTOCOL', protocol)
    protocol_path = root / 'protocol.json'
    data.write_json(protocol_path, protocol)
    identity = {'source_sha256': {'fake.py': 'a' * 64}}
    monkeypatch.setattr(ds, 'research_identity', lambda: deepcopy(identity))
    monkeypatch.setattr(ds, 'completed_identity', lambda start: start == identity or pytest.fail('identity'))
    return inputs, inventory, protocol_path, tmp_path


def test_full_planning_no_calls_and_scoring_preserves_axes_capmerge_and_nodedup(population):
    inputs, inventory, protocol, root = population
    backend = FakeBackend()
    plan = ds.prepare(inputs, inventory, protocol, root / 'plan', backend)
    assert backend.calls == 0 and plan['counts']['units'] == 4
    assert plan['counts']['measurement_units'] == 6 and plan['counts']['valid_targets'] == 8
    assert plan['counts']['pairs'] == 8  # Exact duplicate pairs across two cases are NOT deduplicated.
    assert plan['longest']['case_index'] == 0 and plan['longest']['claim_index'] == 0
    assert plan['longest']['chunk_index'] == 0 and plan['longest']['framed_tokens'] == 512
    assert plan['original_forward_budget'] == 9
    # A separate scoring process has a fresh backend and no initialized tokenizer.
    backend = FakeBackend()
    assert backend.tokenizer is None
    result = ds.predict(inputs, inventory, protocol, root / 'plan', root / 'scores', backend)
    assert result['actual_forward_attempts'] == backend.calls == 9
    assert result['logical_pairs'] == 8 and result['original_valid_targets'] == 8
    assert result['model_unknown_targets'] == 0 and result['canary_reused']
    ds.data.verify_seal(root / 'scores')
    for row in ds.read_jsonl(inputs):
        prediction = data.read_json(data.prediction_path(root / 'scores/predictions', row['id']))
        assert prediction['original_target_indices'] == [0, 1, 2, 3]
        assert prediction['original_unit_indices'] == [0, 1, 2, 2]
        assert prediction['model_unknown_mask'] == [False] * 4
        assert [t['text'] for t in prediction['tokens']] == ['A', '.', ' B', '.']
        assert [t['score'] for t in prediction['tokens']] == pytest.approx([1 - nli._scores([3., 1., 2.])[0]] * 4)
    assert data.read_json(root / 'scores/complete.json')['annotation_reads'] == 0


def test_all_predefined_long_hypotheses_remain_full_axis_unknown_without_model(population):
    inputs, inventory, protocol, root = population
    backend = FakeBackend(unknown=True)
    plan = ds.prepare(inputs, inventory, protocol, root / 'plan', backend)
    assert plan['counts']['unknown_units'] == 4 and plan['counts']['pairs'] == 0
    assert plan['longest'] is None and plan['original_forward_budget'] == 0
    result = ds.predict(inputs, inventory, protocol, root / 'plan', root / 'scores', backend)
    assert backend.calls == 0 and result['model_unknown_targets'] == 8
    assert result['model_unknown_units'] == 4 and result['model_unknown_cases'] == 2
    prediction = data.read_json(data.prediction_path(root / 'scores/predictions', '0'))
    assert all(t['score'] == .5 for t in prediction['tokens'])
    assert all(prediction['model_unknown_mask'])


@pytest.mark.parametrize('fault,calls', [('canary', 2), ('shape', 3), ('signed-zero', 2)])
def test_numerical_failure_retains_raw_and_attempts_without_completing(population, fault, calls):
    inputs, inventory, protocol, root = population
    backend = FakeBackend(fault=fault)
    ds.prepare(inputs, inventory, protocol, root / 'plan', backend)
    with pytest.raises(ValueError):
        ds.predict(inputs, inventory, protocol, root / 'plan', root / 'scores', backend)
    failure = data.read_json(root / 'scores/failure.json')
    assert failure['actual_forward_attempts'] == backend.calls == calls
    assert not (root / 'scores/complete.json').exists()
    assert len(list((root / 'scores').glob('attempt-*.json'))) == calls
    if fault != 'shape':
        assert (root / 'scores/canary-adapter-raw.json').exists()
    else:
        files = list((root / 'scores/units').rglob('*-raw.json'))
        assert len(files) == 1 and data.read_json(files[0])['value']['logits'] == [[3., 1.]]


def test_sealed_plan_mutation_rejected_before_classification(population):
    inputs, inventory, protocol, root = population
    backend = FakeBackend()
    ds.prepare(inputs, inventory, protocol, root / 'plan', backend)
    with (root / 'plan/units.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        ds.predict(inputs, inventory, protocol, root / 'plan', root / 'scores', backend)
    assert backend.calls == 0 and not (root / 'scores').exists()


def test_original_inventory_drift_and_primitive_offset_corruption_fail_before_calls(population):
    inputs, inventory, protocol, root = population
    backend = FakeBackend()
    primitive = inputs.parent / 'response-0.json'
    text = primitive.read_text(encoding='utf-8').replace('[', '[ ', 1)
    primitive.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='primitive response metadata changed'):
        ds.prepare(inputs, inventory, protocol, root / 'plan', backend)
    assert backend.calls == 0


def test_malformed_window_cannot_hide_source_tail_and_protocol_bool_is_not_integer(population):
    inputs, inventory, protocol, root = population
    backend = FakeBackend()
    old = backend.prepare_pairs
    def broken(source, hypothesis):
        prepared = old(source, hypothesis)
        prepared['chunks'].pop()
        return prepared
    backend.prepare_pairs = broken
    with pytest.raises(ValueError, match='coverage is incomplete'):
        ds.prepare(inputs, inventory, protocol, root / 'plan', backend)
    value = data.read_json(protocol)
    value['batch_size'] = True
    other = root / 'bad-protocol.json'
    data.write_json(other, value)
    with pytest.raises(ValueError, match='exact full-population'):
        ds._protocol(other)
    assert backend.calls == 0


def test_empty_hypothesis_is_unknown_not_vacuously_supported():
    prepared = ds._prepared(FakeBackend(), 'source', '')
    assert prepared['status'] == 'unknown' and prepared['reason'] == 'empty_tokenized_hypothesis'
    assert prepared['chunks'] == []


def test_legacy_FP16_backend_cannot_masquerade_as_new_FP32_plan(population):
    inputs, inventory, protocol, root = population
    ds.prepare(inputs, inventory, protocol, root / 'plan', FakeBackend())
    backend = FakeBackend()
    backend.execution_mode = 'legacy-pilot'
    with pytest.raises(ValueError, match='exact FP32 dataset backend mode'):
        ds.predict(inputs, inventory, protocol, root / 'plan', root / 'scores', backend)
    assert backend.calls == 0 and not (root / 'scores').exists()


@pytest.fixture
def refused_prefix(population, monkeypatch):
    inputs, inventory, protocol, root = population
    class ResourceBackend(FakeBackend):
        def classify(self, pairs):
            if self.calls == 6:
                raise RuntimeError('Dataset NLI requires the frozen512MiB host/GPU floors')
            return super().classify(pairs)
    old = ResourceBackend()
    ds.prepare(inputs, inventory, protocol, root / 'old-plan', old)
    with pytest.raises(RuntimeError, match='frozen512MiB'):
        ds.predict(inputs, inventory, protocol, root / 'old-plan', root / 'failed', old)
    assert old.calls == 6
    failure = data.read_json(root / 'failed/failure.json')
    assert failure['actual_forward_attempts'] == 7 and failure['partial_logical_pairs'] == 5
    assert failure['completed_sentence_units'] == 2
    # A successful raw from an unfinished third unit must NOT be rerun.
    partial = root / 'failed/units' / data.prediction_path(Path('.'), '1').stem
    assert (partial / '00000-00000-raw.json').exists() and not (partial / '00000.json').exists()
    ds._seal(root / 'failed')
    monkeypatch.setattr(ds, '_recovery_source_proof', lambda *_: {'core_AST_unchanged': True, 'test_only': True})
    return population


def test_explicit_recovery_replays_every_successful_raw_including_partial_unit(refused_prefix, monkeypatch):
    from types import SimpleNamespace
    inputs, inventory, protocol, root = refused_prefix
    prepared = ds.prepare_recovery(inputs, inventory, protocol, root / 'old-plan', root / 'failed',
        root / 'historical.py', root / 'compatible', FakeBackend())
    assert prepared['recovery']['successful_model_results'] == 6
    assert prepared['recovery']['remaining_model_calls'] == 3
    for name in ('units.jsonl', 'protocol.json', 'model_identity.json'):
        assert (root / 'old-plan' / name).read_bytes() == (root / 'compatible' / name).read_bytes()
    fresh = FakeBackend()
    monkeypatch.setattr(nli, 'HFEntailment', lambda *a, **k: fresh)
    identity = root / 'identity.json'
    data.write_json(identity, fresh.model_identity)
    args = SimpleNamespace(inputs=inputs, inventory=inventory, protocol=protocol, model='unused',
        model_identity=identity, plan=root / 'compatible', reuse_failed=root / 'failed',
        reuse_stage_source=root / 'historical.py', output=root / 'recovered', device='cuda:0', prepare_only=False)
    result = ds.run(args)
    assert result['actual_forward_attempts'] == 9 and fresh.calls == 3 and fresh.closed
    sidecar = data.read_json(root / 'recovered.recovery.json')
    assert sidecar['completed'] and sidecar['replayed_valid_model_results'] == 6
    assert sidecar['new_successful_model_calls'] == sidecar['new_physical_interface_requests'] == 3
    assert sidecar['combined_successful_model_calls'] == 9
    assert sidecar['combined_physical_interface_requests'] == 10  # Exactly one old pre-model guard.
    assert sidecar['cached_canaries_reuse_original_qualification']
    assert not sidecar['fresh_resident_canary_performed']
    data.verify_seal(root / 'recovered')
    assert data.read_json(root / 'recovered/complete.json')['samples'] == 2


def test_replay_prefix_rejects_request_drift_before_any_new_model(refused_prefix):
    _, _, _, root = refused_prefix
    fresh = FakeBackend()
    proxy = ds.ReplayPrefixBackend(fresh, root / 'old-plan', root / 'failed')
    pair = deepcopy(data.read_json(root / 'old-plan/plan.json')['longest']['pair'])
    pair['input_ids'][1] += 1
    with pytest.raises(ValueError, match='canonical call sequence'):
        proxy.classify([pair])
    assert proxy.cached_calls == fresh.calls == proxy.new_interface_requests == 0


def test_tampered_retained_raw_is_not_accepted_as_reusable_model_result(refused_prefix):
    _, _, _, root = refused_prefix
    raw = root / 'failed/canary-direct-raw.json'
    raw.write_text(raw.read_text(encoding='utf-8').replace('3.0', '4.0'), encoding='utf-8')
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        ds.ReplayPrefixBackend(FakeBackend(), root / 'old-plan', root / 'failed')


def test_recovery_ast_gate_accepts_only_dispatch_growth_and_unchanged_core(tmp_path):
    old = tmp_path / 'old.py'
    new = tmp_path / 'new.py'
    original = 'def predict(x):\n    return x + 1\n\ndef run(args):\n    """dispatch"""\n    return predict(args)\n'
    old.write_text(original, encoding='utf-8')
    compatible = original.replace('    return predict(args)',
        "    if getattr(args, 'reuse_failed', None):\n        return recover(args)\n    return predict(args)")
    compatible += '\ndef recover(args):\n    return args\n'
    new.write_text(compatible, encoding='utf-8')
    ds._module_compatibility(old, new)
    new.write_text(compatible.replace('return x + 1', 'return x + 2'), encoding='utf-8')
    with pytest.raises(ValueError, match='core AST changed'):
        ds._module_compatibility(old, new)
