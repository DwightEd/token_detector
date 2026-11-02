"""Frozen NLI collector tests use synthetic text and fake classification only."""

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from experiments import data, nli_qualification as module


FILES = ('config.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'added_tokens.json', 'tokenizer.json', 'spm.model', 'model.safetensors')
MODEL = dict(model_id='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
    revision='6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
    model_files_sha256=dict(module.PINNED_SHA256),
    runtime={'torch': '2.8.0+cu128', 'transformers': '4.57.1'})


class Tokenizer:
    """One ID per character makes full source coverage independently checkable."""
    is_fast = True
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens=False, truncation=False, **kwargs):
        assert add_special_tokens is False and truncation is False
        return [] if text == 'EMPTY-TOKENIZED' else [ord(char) + 10 for char in text]

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def build_inputs_with_special_tokens(self, premise, hypothesis):
        return [1, *premise, 2, *hypothesis, 2]


class Backend:
    model_identity = MODEL

    def __init__(self, check=None, malformed=False):
        self.check, self.malformed = check, malformed
        self.calls, self.prepared = [], []

    def prepare_pairs(self, source, hypothesis):
        self.prepared.append((source, hypothesis))
        return module._prepare_pairs(Tokenizer(), source, hypothesis)

    def classify(self, pairs):
        if self.check:
            self.check(pairs)
        self.calls.append(deepcopy(pairs))
        return dict(logits=[[0., 1., 2.] for _ in pairs][1:] if self.malformed else
                    [[0., 1., 2.] for _ in pairs], actual_forward_passes=1,
                    input_tokens=sum(len(pair['input_ids']) for pair in pairs),
                    padded_input_tokens=len(pairs) * max(len(pair['input_ids']) for pair in pairs),
                    seconds=.125)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    monkeypatch.setattr(data, 'write_json', write)
    monkeypatch.setattr(module, 'research_identity', lambda: {'source_sha256': {'fixed': 'a' * 64}})
    monkeypatch.setattr(module, 'completed_identity', lambda start: start)

    def build(source='Synthetic evidence.', clause='Same.', unique_sources=False):
        panel = tmp_path / 'panel'
        panel.mkdir()
        inputs = panel / 'inputs.jsonl'
        rows = []
        for index in range(34):
            evidence = source + str(index) if unique_sources else source
            response = ' '.join([clause] * (4 if index < 2 else 3))
            rows.append(dict(id=f'METADATA_ONLY_{index}', source_id=f'source-{index}', split='test',
                task='QA', prompt='Read ' + evidence + ' Answer fully.', source=evidence,
                response=response, generator='synthetic-generator'))
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        for name in ('manifest.json', 'evaluation_only.jsonl', 'annotations.jsonl'):
            (panel / name).write_text('MUST NOT OPEN', encoding='utf-8')
        protocol = tmp_path / 'protocol.json'
        write(protocol, dict(module.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        return inputs, protocol, tmp_path / 'outputs' / 'nli', rows
    return build


def test_windows_cover_every_source_id_without_chunk_cap_or_truncation():
    source = ''.join(chr(1000 + index) for index in range(6200))
    hypothesis = 'h' * 450
    planned = module._prepare_pairs(Tokenizer(), source, hypothesis)
    assert planned['status'] == 'scored'
    assert planned['source_tokens'] == 6200 and planned['hypothesis_tokens'] == 450
    assert planned['budget'] == 59 and planned['overlap_tokens'] == 14
    chunks = planned['chunks']
    assert len(chunks) > 64
    covered = set()
    original_ids = Tokenizer().encode(source)
    for index, chunk in enumerate(chunks):
        start, end = chunk['start'], chunk['end']
        assert chunk['input_ids'][1:1 + end - start] == original_ids[start:end]
        assert len(chunk['input_ids']) == 3 + end - start + 450 <= 512
        assert chunk['attention_mask'] == [1] * len(chunk['input_ids'])
        if index:
            assert start == chunks[index - 1]['end'] - 14
        covered.update(range(start, end))
    assert chunks[0]['start'] == 0 and chunks[-1]['end'] == 6200
    assert covered == set(range(6200))


def test_exact_512_boundary_and_explicit_unknowns_are_not_truncated():
    planned = module._prepare_pairs(Tokenizer(), 'AB', 'h' * 508)
    assert planned['budget'] == 1 and planned['overlap_tokens'] == 0
    assert [(chunk['start'], chunk['end']) for chunk in planned['chunks']] == [(0, 1), (1, 2)]
    assert all(len(chunk['input_ids']) == 512 for chunk in planned['chunks'])
    for source, hypothesis in [('AB', 'h' * 509), ('AB', 'h' * 1200), ('EMPTY-TOKENIZED', 'h')]:
        unknown = module._prepare_pairs(Tokenizer(), source, hypothesis)
        assert unknown['status'] == 'unknown' and unknown['chunks'] == [] and unknown['reason']
        assert unknown['hypothesis_tokens'] == len(hypothesis)


def test_freezes_all_34_originals_and_104_units_before_first_classification(prepared, monkeypatch):
    inputs, protocol, output, rows = prepared()
    original_open = Path.open

    def no_gold(path, *args, **kwargs):
        assert path.name not in {'manifest.json', 'evaluation_only.jsonl', 'annotations.jsonl'}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', no_gold)

    def frozen(pairs):
        assert all((output / name).is_file() for name in
                   ['plan.json', 'protocol.json', 'source_identity.json', 'model_identity.json'])
        plan = data.read_json(output / 'plan.json')
        assert len(plan['records']) == 34
        assert sum(len(record['units']) for record in plan['records']) == 104
        assert all(set(pair) >= {'input_ids', 'attention_mask'} for pair in pairs)
        assert all(len(pair['input_ids']) <= 512 for pair in pairs)

    backend = Backend(check=frozen)
    complete = module.qualify(inputs, protocol, output, backend, MODEL)
    assert len(complete['records']) == 34
    assert sum(len(row['units']) for row in complete['records']) == 104
    assert len(backend.prepared) == 104
    assert all(source == 'Synthetic evidence.' and hypothesis in {'Same. ', 'Same.'}
               for source, hypothesis in backend.prepared)
    assert len(backend.calls) == 1 and len(backend.calls[0]) == 2
    denominator = sum(math.exp(value) for value in [0., 1., 2.])
    expected = [math.exp(index) / denominator for index in [0, 2, 1]]
    for original, row in zip(rows, complete['records']):
        assert row['id'] == original['id'] and row['source_id'] == original['source_id']
        assert ''.join(unit['text'] for unit in row['units']) == original['response']
        for unit in row['units']:
            assert unit['risk'] == pytest.approx(1.0 - expected[0])
            assert unit['class_scores'] == pytest.approx(dict(zip(['supported', 'contradicted', 'unresolved'], expected)))
    assert complete['costs']['actual_forward_passes'] == 1
    data.verify_seal(output)


def test_fixed_eight_pair_batches_and_duplicate_pair_reuse(prepared):
    inputs, protocol, output, _ = prepared(unique_sources=True)
    backend = Backend()
    result = module.qualify(inputs, protocol, output, backend, MODEL)
    sizes = [len(batch) for batch in backend.calls]
    assert sizes and all(size == 8 for size in sizes[:-1]) and 1 <= sizes[-1] <= 8
    frozen_pairs = [tuple(pair['input_ids']) for batch in backend.calls for pair in batch]
    assert len(frozen_pairs) == len(set(frozen_pairs)) < 104
    assert result['costs']['actual_forward_passes'] == len(sizes)


@pytest.mark.parametrize('kwargs', [dict(source='EMPTY-TOKENIZED'), dict(clause='h' * 510 + '.')])
def test_unknown_fallback_retains_every_unit_and_makes_no_forward(prepared, kwargs):
    inputs, protocol, output, _ = prepared(**kwargs)
    backend = Backend()
    result = module.qualify(inputs, protocol, output, backend, MODEL)
    units = [unit for row in result['records'] for unit in row['units']]
    assert len(result['records']) == 34 and len(units) == 104 and backend.calls == []
    assert all(unit['risk'] == 1.0 and unit['status'] == 'unknown' for unit in units)
    assert result['costs']['actual_forward_passes'] == 0
    data.verify_seal(output)


def test_bad_original_hash_rejects_before_backend_access(prepared):
    inputs, protocol, output, _ = prepared()
    value = data.read_json(protocol)
    value['original_inputs_sha256'] = 'f' * 64
    data.write_json(protocol, value)
    backend = Backend()
    with pytest.raises(ValueError, match='hash'):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.prepared == backend.calls == [] and not output.exists()


def test_no_protocol_chunk_cap_and_no_existing_run_retry(prepared):
    inputs, protocol, output, _ = prepared()
    value = data.read_json(protocol)
    value['max_chunks'] = 2
    data.write_json(protocol, value)
    backend = Backend()
    with pytest.raises(ValueError):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert not backend.prepared and not backend.calls
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert not backend.calls


def test_invalid_batch_shape_is_retained_and_never_marked_complete(prepared):
    inputs, protocol, output, _ = prepared()
    backend = Backend(malformed=True)
    with pytest.raises(ValueError):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert len(backend.calls) == 1 and not (output / 'complete.json').exists()
    assert (output / 'plan.json').is_file()
    assert any('logits' in path.read_text(encoding='utf-8') for path in output.rglob('*.json'))


def test_hf_constructor_is_lazy_and_resource_admission_is_explicit(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Construction must not access assets or import a model')
    monkeypatch.setattr(module.HFEntailment, 'verify_frozen_identity', forbidden)
    backend = module.HFEntailment(tmp_path / 'never-created', MODEL)
    assert backend.model is None and backend.tokenizer is None and backend.admitted is False
    with pytest.raises(RuntimeError, match='admission'):
        backend.classify([dict(input_ids=[1, 10, 2, 11, 2], attention_mask=[1] * 5)])


def test_hf_identity_verifies_exact_three_head_label_order_without_loading(tmp_path, monkeypatch):
    assets = tmp_path / 'synthetic-assets'
    assets.mkdir()
    for name in FILES:
        (assets / name).write_text('synthetic-only', encoding='utf-8')
    config = dict(id2label={'0': 'entailment', '1': 'neutral', '2': 'contradiction'},
                  max_position_embeddings=512, architectures=['DebertaV2ForSequenceClassification'])
    (assets / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    identity = deepcopy(MODEL)
    # Isolate the configuration gate from the independently tested asset hashes.
    monkeypatch.setattr(data, 'file_hash', lambda path: module.PINNED_SHA256[Path(path).name])
    backend = module.HFEntailment(assets, identity)
    backend.verify_frozen_identity()
    assert backend.model is None and backend.tokenizer is None
    config['id2label'] = {'0': 'entailment', '1': 'contradiction', '2': 'neutral'}
    (assets / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError, match='head'):
        module.HFEntailment(assets, identity).verify_frozen_identity()


@pytest.mark.parametrize('count', [0, 9])
def test_hf_batch_size_rejected_before_any_model_import(tmp_path, monkeypatch, count):
    import builtins
    original_import = builtins.__import__

    def no_model_import(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'transformers'}
        return original_import(name, *args, **kwargs)

    backend = module.HFEntailment(tmp_path / 'never-created', MODEL, admit_resources=True)
    monkeypatch.setattr(builtins, '__import__', no_model_import)
    with pytest.raises(ValueError, match='batches'):
        backend.classify([dict(input_ids=[1, 10, 2, 11, 2], attention_mask=[1] * 5)] * count)
    assert backend.model is None and backend.tokenizer is None


def test_backend_cannot_mutate_token_ids_without_changing_shape(prepared):
    inputs, protocol, output, _ = prepared()

    class MutatingBackend(Backend):
        def classify(self, pairs):
            pairs[0]['input_ids'][0] += 1
            return super().classify(pairs)

    backend = MutatingBackend()
    with pytest.raises(ValueError, match='(?i)mutat'):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert len(backend.calls) == 1 and not (output / 'complete.json').exists()
    assert len(list(output.rglob('backend_raw.json'))) == 1


def test_model_file_hashes_cannot_be_rebound_under_the_same_repository_identity(prepared):
    inputs, protocol, output, _ = prepared()
    identity = deepcopy(MODEL)
    identity['model_files_sha256']['model.safetensors'] = 'b' * 64
    backend = Backend()
    backend.model_identity = identity
    with pytest.raises(ValueError):
        module.qualify(inputs, protocol, output, backend, identity)
    assert backend.prepared == backend.calls == [] and not output.exists()
