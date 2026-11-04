"""Finite owner orchestration with fake prompts/backend; no real tokenizer/model."""

from contextlib import contextmanager
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import operator_owner_observation as stage
from experiments.data import file_hash, read_json, verify_seal, write_json
from experiments.datasets import Sample
from experiments.relation_intervention import _seal
from experiments.run import parser


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model, axes_dir, prepared, baseline, audit = (tmp_path/name for name in
        ('model', 'original-axes', 'prepared', 'baseline', 'audit'))
    for path in (model, axes_dir, prepared, baseline, audit):
        path.mkdir()
    write_json(model/'config.json', dict(num_hidden_layers=32, hidden_size=4096, vocab_size=128256))
    write_json(axes_dir/'seal.json', {'original-axis': 'f'*64})
    (prepared/'inputs.jsonl').write_text('{}\n', encoding='utf8')
    response = dict(text='x'*233, answer_ids=[1]*233, offsets=[[i, i+1] for i in range(233)],
                    special_ids=[], units=[dict(start=0, stop=233)])
    response['answer_ids'][73], response['answer_ids'][87] = 539, 2561
    primitive = tmp_path/'response.json'
    write_json(primitive, response)
    sample = Sample('5684', '13603', 'test', 'Data2txt', 'old-source', 'old-source', response['text'])
    prompt_ids = np.arange(803, dtype=np.int64)
    answer_ids = np.asarray(response['answer_ids'], dtype=np.int64)
    axes = dict(token_ids=np.concatenate((prompt_ids, answer_ids)), answer_ids=answer_ids,
        source_mask=np.array([True]*707 + [False]*(1036-707)),
        prechoice_positions=np.arange(802, 1035, dtype=np.int64),
        all_query_positions=np.arange(802, 1036, dtype=np.int64),
        full_offsets=np.column_stack((np.arange(1036), np.arange(1, 1037))).astype(np.int64),
        answer_offsets=np.asarray(response['offsets'], dtype=np.int64),
        target=np.arange(233, dtype=np.int64), token_id=answer_ids.copy(),
        answer_index=np.full(233, 8, dtype=np.int32), unit_index=np.zeros(233, dtype=np.int32))
    case = dict(sample=sample, response=response, axes=axes,
        axis={key: axes[key] for key in ('target', 'token_id', 'answer_index', 'unit_index')},
        metadata={'native_collection_answer_index': 8}, P=803, T=233, N=1036)
    frontend = tmp_path/'frontend.json'
    write_json(frontend, dict(source_fields=[], evidence_bindings={'input_sha256': file_hash(prepared/'inputs.jsonl')},
        cases=[dict(id='5684', source_id='13603', response_sha256=stage.text_hash(response['text']),
                    primitive_response_path=str(primitive), primitive_response_sha256=file_hash(primitive))]))
    old = dict(source_sha256={'experiments/operator_binding_observation.py': 'a'*64,
        'experiments.research/operator_binding.py': 'b'*64, 'experiments/run.py': 'c'*64},
        python='fake-runtime', dependencies={'torch': 'fake-torch'}, threads={'OMP_NUM_THREADS': '2'})
    identity = copy.deepcopy(old)
    identity['source_sha256'].update({name: 'e'*64 for name in stage.ADDED_MODULES})
    identity['source_sha256']['experiments/run.py'] = 'd'*64
    old_plan = dict(source_runtime=old, axes_seal_sha256=file_hash(axes_dir/'seal.json'),
        frontend_sha256=file_hash(frontend), inputs_sha256=file_hash(prepared/'inputs.jsonl'),
        assets={'files': {}}, observer_capture_identity={'fixed': True}, head_prerequisite={'fixed': True})
    write_json(baseline/'plan.json', old_plan)
    write_json(baseline/'result.json', dict(completed=True, annotation_reads=0,
        actual_counts=stage.BUDGET, source_runtime=old))
    child = baseline/'case-5684'; child.mkdir()
    write_json(child/'axis.json', dict(id='5684', P=803, T=233, N=1036,
        original_response_sha256=stage.text_hash(response['text'])))
    write_json(child/'result.json', dict(completed=True, annotation_reads=0, original_targets=233))
    write_json(child/'prediction.json', dict(id='5684', no_score_payload_needed_for_reuse=True))
    _seal(child); _seal(baseline)
    write_json(audit/'spec.json', dict(observations_seal_sha256=file_hash(baseline/'seal.json'),
        axes_seal_sha256=old_plan['axes_seal_sha256'], frontend_sha256=old_plan['frontend_sha256']))
    write_json(audit/'audit.json', dict(passed=True, complete_raw_audited=True, label_reads=0,
                                     cases={'5684': {'passed': True}}))
    _seal(audit)
    protocol = tmp_path/'protocol.json'
    write_json(protocol, dict(schema='h018-operator-owner-phase2-v1', world_order=list(stage.WORLD_ORDER),
        budget=stage.BUDGET, scientific_bindings={}, baseline_audit=dict(root=str(audit),
            seal_sha256=file_hash(audit/'seal.json'), audit_sha256=file_hash(audit/'audit.json'))))
    args = SimpleNamespace(protocol=protocol, axes=axes_dir, prepared=prepared, frontend_contract=frontend,
        baseline=baseline, model=model, asset_manifest=tmp_path/'assets.json',
        asset_verification=tmp_path/'verification.json', asset_state=tmp_path/'state.json',
        head_qualification=tmp_path/'head', capture_root=tmp_path/'capture', device='cuda:0',
        output=tmp_path/'plan', plan=None, prepare_only=True)
    monkeypatch.setattr(stage, 'research_identity', lambda: copy.deepcopy(identity))
    monkeypatch.setattr(stage, 'completed_identity', lambda value: value)
    monkeypatch.setattr(stage.original, 'load_cases', lambda *unused: {'5684': case})
    monkeypatch.setattr(stage, '_prerequisites', lambda args: dict(assets={'files': {}},
        observer_capture_identity={'fixed': True}, head_prerequisite={'fixed': True}))
    observed = dict(encodes=[], renders=[], models=[])

    class Tokenizer:
        is_fast = True
        def apply_chat_template(self, messages, **kwargs):
            observed['renders'].append(messages[0]['content'])
            return messages[0]['content']
        def __call__(self, text, **kwargs):
            observed['encodes'].append(text)
            return {'input_ids': []}

    monkeypatch.setattr(stage, '_tokenizer', lambda model: Tokenizer())

    def swaps(sample, fields, tokenizer, original_axes):
        worlds = {}
        for index, name in enumerate(stage.WORLD_ORDER):
            source = name + '-source'
            tokenizer.apply_chat_template([{'role': 'user', 'content': source}],
                tokenize=False, add_generation_prompt=True)
            tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
            new_axes = {key: value.copy() for key, value in original_axes.items()}
            new_axes['token_ids'][1+index:3+index] = new_axes['token_ids'][1+index:3+index][::-1]
            worlds[name] = dict(source=source, prompt=source, response=sample.response,
                P=803, T=233, N=1036, axes=new_axes, edits=[], field_proofs=[], chat=source,
                source_sha256=stage.text_hash(source), prompt_sha256=stage.text_hash(source),
                response_sha256=stage.text_hash(sample.response))
        return dict(worlds=worlds, audit=dict(passed=True, source_encodings=2, S0_encodings=0,
            response_tokenizations=0, response_decodes=0, model_calls=0, annotation_reads=0))

    monkeypatch.setattr(stage, 'prepare_swaps', swaps)
    return SimpleNamespace(args=args, case=case, identity=identity, observed=observed, old=old,
                           protocol=read_json(protocol), monkeypatch=monkeypatch)


def test_prepare_encodes_only_two_new_prompts_and_preserves_original_axis(setup):
    def forbidden(*args, **kwargs):
        raise AssertionError('prepare must not construct a model')
    setup.monkeypatch.setattr(stage, '_open_observer', forbidden)
    plan = stage.run(setup.args)
    assert setup.observed['encodes'] == ['target-source', 'control-source']
    assert plan['model_calls'] == 0 and plan['tokenizer_counts']['source_encodings'] == 2
    assert plan['tokenizer_counts']['response_tokenizations'] == 0
    assert plan['tokenizer_counts']['S0_encodings'] == 0
    assert plan['axes_seal_sha256'] == file_hash(setup.args.output/'frontend/seal.json')
    assert plan['frontend_sha256'] == file_hash(setup.args.output/'frontend/FRONTEND_CF.json')
    assert plan['baseline_reference']['S0_new_model_calls'] == 0
    assert plan['baseline_reference']['copied_raw_bytes'] == 0
    for name in stage.WORLD_ORDER:
        with np.load(setup.args.output/f'frontend/axes-{name}.npz') as saved:
            for key in ('answer_ids', 'answer_offsets', 'target', 'token_id', 'answer_index', 'unit_index'):
                np.testing.assert_array_equal(saved[key], setup.case['axes'][key])
        record = read_json(setup.args.output/f'frontend/world-{name}.json')
        assert record['constructed_counterfactual'] and record['original_id'] == '5684'
    verify_seal(setup.args.output)


@pytest.mark.parametrize('change', ['kernel', 'runtime', 'audit', 'assets'])
def test_original_baseline_dependency_or_audit_change_rejects_before_tokenization(setup, change):
    if change == 'kernel':
        setup.identity['source_sha256']['experiments.research/operator_binding.py'] = 'f'*64
    elif change == 'runtime':
        setup.identity['dependencies']['torch'] = 'other'
    elif change == 'audit':
        audit = setup.args.baseline.parent/'audit/audit.json'
        audit.write_text('{}', encoding='utf8')
    else:
        setup.monkeypatch.setattr(stage, '_prerequisites', lambda args: dict(assets={'changed': True},
            observer_capture_identity={'fixed': True}, head_prerequisite={'fixed': True}))
    with pytest.raises(ValueError):
        stage.run(setup.args)
    assert setup.observed['encodes'] == [] and setup.observed['models'] == []
    failure = read_json(setup.args.output/'failure.json')
    assert failure['actual_counts']['model_calls'] == 0
    verify_seal(setup.args.output)


def _execution(setup, fail=None):
    plan = stage.run(setup.args)
    args = copy.copy(setup.args)
    args.plan, args.output, args.prepare_only = setup.args.output, setup.args.output.parent/'observed', False

    class Observer:
        def resource_report(self):
            return {'all_parameters_meta': True}

    @contextmanager
    def opened(*unused):
        yield Observer()

    setup.monkeypatch.setattr(stage, '_open_observer', opened)

    def observe(observer, case, output, counts):
        name = case['sample'].id
        assert name in ('target', 'control') and name != '5684'
        setup.observed['models'].append(name)
        output.mkdir()
        counts['model_calls'] += 1
        if name == fail:
            write_json(output/'failure.json', {'injected': True, 'actual_counts': dict(counts)})
            _seal(output)
            raise RuntimeError('injected constructed-world failure')
        for key, value in stage.BUDGET.items():
            if key != 'model_calls':
                counts[key] += value // 2
        result = dict(completed=True, annotation_reads=0, phase2_admitted=False)
        write_json(output/'result.json', result)
        _seal(output)
        return result

    setup.monkeypatch.setattr(stage.original, 'observe_case', observe)
    setup.monkeypatch.setattr(stage, '_tokenizer', lambda *args: (_ for _ in ()).throw(
        AssertionError('execution must not tokenize')))
    return args, plan


def test_execute_calls_only_two_constructed_worlds_and_never_reencodes(setup):
    args, plan = _execution(setup)
    result = stage.run(args)
    assert setup.observed['models'] == ['target', 'control']
    assert setup.observed['encodes'] == ['target-source', 'control-source']
    assert all(result['actual_counts'][key] == value for key, value in stage.BUDGET.items())
    assert result['phase2_admitted'] and result['finite_control_only']
    assert not result['method_promotion'] and not result['detector_validated']
    assert result['annotation_reads'] == result['S0_new_model_calls'] == 0
    assert result['auroc'] is None
    verify_seal(args.output)
    with pytest.raises(FileExistsError):
        stage.run(args)
    assert setup.observed['models'] == ['target', 'control']


def test_failure_retains_completed_world_counters_and_raw_without_retry(setup):
    args, plan = _execution(setup, fail='control')
    with pytest.raises(RuntimeError, match='injected'):
        stage.run(args)
    failure = read_json(args.output/'failure.json')
    assert failure['actual_counts']['model_calls'] == 2
    assert failure['actual_counts']['head_calls'] == 4
    assert failure['completed'] == ['target'] and not failure['automatic_retry']
    assert (args.output/'case-target/result.json').is_file()
    assert (args.output/'case-control/failure.json').is_file()
    assert not (args.output/'result.json').exists()
    verify_seal(args.output)
    with pytest.raises(FileExistsError):
        stage.run(args)
    assert setup.observed['models'] == ['target', 'control']


def test_sealed_frontend_tamper_is_rejected_before_new_model(setup):
    args, plan = _execution(setup)
    (args.plan/'frontend/world-target.json').write_text('{}', encoding='utf8')
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        stage.run(args)
    assert setup.observed['models'] == []


def test_progress_io_failure_still_counts_sealed_completed_world_and_stops(setup):
    args, plan = _execution(setup)
    publish = stage.write_json

    def fail_progress(path, value):
        if path.name == 'progress-target.json':
            raise OSError('injected progress publication failure')
        return publish(path, value)

    setup.monkeypatch.setattr(stage, 'write_json', fail_progress)
    with pytest.raises(OSError, match='progress publication'):
        stage.run(args)
    failure = read_json(args.output/'failure.json')
    assert failure['type'] == 'OSError' and failure['completed'] == ['target']
    assert failure['actual_counts']['model_calls'] == 1
    assert failure['actual_counts']['head_calls'] == 4
    assert failure['completion_evidence']['target']['result_sha256'] == file_hash(
        args.output/'case-target/result.json')
    assert setup.observed['models'] == ['target']
    assert not (args.output/'progress-target.json').exists()
    assert not (args.output/'case-control').exists()
    verify_seal(args.output)


def test_output_cannot_be_nested_in_original_baseline(setup):
    setup.args.output = setup.args.baseline/'would-mutate-old-seal'
    with pytest.raises(ValueError, match='must not mutate'):
        stage.run(setup.args)
    assert not setup.args.output.exists() and setup.observed['encodes'] == []


def test_s0_or_extra_world_is_not_a_valid_new_observation(setup, tmp_path):
    with pytest.raises(ValueError, match='Only target/control'):
        stage.observe_worlds(None, {'5684': setup.case, 'target': setup.case}, tmp_path, {})
    assert setup.observed['models'] == []


def test_cli_owner_stage_is_separate_and_requires_all_pinned_inputs():
    options = []
    for key in ('protocol', 'axes', 'prepared', 'frontend-contract', 'baseline', 'model',
                'asset-manifest', 'asset-verification', 'asset-state',
                'head-qualification', 'capture-root', 'output'):
        options.extend(('--'+key, '/fake/'+key))
    args = parser().parse_args(['transfer', 'observe-operator-owner', '--prepare-only', *options])
    assert args.transfer_stage == 'observe-operator-owner' and args.prepare_only
    assert args.baseline.name == 'baseline' and args.plan is None
