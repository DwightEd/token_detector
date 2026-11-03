"""H014 cached-readout tests; synthetic donor/control only, no new models."""
from copy import deepcopy
import importlib
from pathlib import Path

import pytest

from experiments import data, masked_assignment as module
from experiments import masked_reconstruction_qualification as profile
from experiments.datasets import Sample
from experiments.research import bijection_readout
from tests.test_masked_reconstruction_evaluation import make_bridge
from tests.test_reconstruction_evaluation import make_run, synthetic48, write, seal


@pytest.fixture
def assignment_run(make_bridge, monkeypatch):
    def build():
        args = make_bridge()
        source = deepcopy(module.donor.research_identity())
        for name in (*module._REFERENCE_HELPERS, 'experiments/masked_assignment.py',
                     'experiments.research/bijection_readout.py'):
            path = Path(importlib.import_module(name[:-3].replace('/', '.')).__file__)
            source['source_sha256'][name] = data.file_hash(path)
        for target in (module, module.donor):
            monkeypatch.setattr(target, 'research_identity', lambda: deepcopy(source))
            monkeypatch.setattr(target, 'completed_identity', lambda start: start)
        module.donor.evaluate(*args)
        protocol = args[-1].parent / 'cached-assignment-protocol.json'
        write(protocol, module.DEFAULT_PROTOCOL)
        return args[0], args[3], args[4], args[6], args[6].with_name('cached-assignment'), protocol
    return build


def guard_reference(args, monkeypatch, allow=False):
    original, accesses = Path.open, []
    def guarded(path, *rest, **kwargs):
        if path.is_relative_to(args[3]):
            accesses.append(path)
            assert allow, 'H011 reference reached before complete cached-score admission'
            freeze = data.read_json(args[4] / 'freeze.json')
            assert set(freeze) == {'plan.json', 'protocol.json', 'source_identity.json', 'scores.json'}
            assert data.read_json(args[4] / 'plan.json')['state'] == 'scores_frozen_before_reference_release'
            scores = data.read_json(args[4] / 'scores.json')
            assert scores['evaluation_material_accessed'] is False
            assert len(scores['matrices']) == 24 and len(scores['queries']) == 48 and len(scores['records']) == 48
        return original(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    return accesses


def test_all24_scores_freeze_before_any_reference_and_both_readouts_keep_every_burden(assignment_run, monkeypatch):
    args = assignment_run()
    before = [{str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
              for root in (args[1], args[3])]
    accesses = guard_reference(args, monkeypatch, allow=True)
    summary = module.analyze(*args)
    assert accesses and summary['new_model_calls'] == 0 and summary['new_language_observations'] is False
    assert summary['synthetic_offsets_applied'] is False and summary['winning_readout_selected'] is False
    assert summary['source_burden_matrices'] == 24 and summary['cached_completions'] == 96
    assert summary['original_h011_summary'] == data.read_json(args[3] / 'summary.json')
    assert summary['donor_costs']['actual_forward_passes'] == 96
    for burden in ('0', '4', '12'):
        for readout in module.DEFAULT_PROTOCOL['readouts']:
            result = summary['by_burden'][burden]['readouts'][readout]
            assert result['source_assignment'] == dict(total=8, correct=8, wrong=0, ties=0, unknown=0)
            assert result['query_argmax'] == dict(total=16, correct=16, wrong=0, ties=0, unknown=0)
    assert len(data.read_json(args[4] / 'matrices.json')) == 24
    assert len(data.read_json(args[4] / 'queries.json')) == 48
    assert len(data.read_json(args[4] / 'cases.json')) == 48
    assert before == [{str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
                      for root in (args[1], args[3])]
    data.verify_seal(args[4])


@pytest.mark.parametrize('damage', ['root_extra', 'child_seal_escape'])
def test_all_donor_names_and_seal_keys_precede_every_payload_hash(assignment_run, monkeypatch, damage):
    args = assignment_run()
    if damage == 'root_extra':
        (args[1] / 'unexpected').write_text('must not hash', encoding='utf-8')
    else:
        child = next((args[1] / 'requests').iterdir())
        value = data.read_json(child / 'seal.json'); value['../escape'] = 'f' * 64
        write(child / 'seal.json', value); seal(args[1])
    accesses = guard_reference(args, monkeypatch)
    def forbidden(*rest, **kwargs): raise AssertionError('Payload/source reached before full namespace admission')
    monkeypatch.setattr(data, 'file_hash', forbidden)
    monkeypatch.setattr(module, 'research_identity', forbidden)
    with pytest.raises(ValueError): module.analyze(*args)
    assert accesses == [] and not args[4].exists()


@pytest.mark.parametrize('damage', ['raw', 'query', 'case'])
def test_resealed_h011_reference_raw_query_or_case_must_reconstruct_exactly(assignment_run, monkeypatch, damage):
    args = assignment_run()
    filename = {'raw': 'candidate-observations.json', 'query': 'queries.json', 'case': 'cases.json'}[damage]
    path = args[3] / filename; value = data.read_json(path)
    if damage == 'raw': value[0]['response']['joint_log_probability'] += .25
    elif damage == 'query': value[0]['reference_candidate'] = next(c for c in value[0]['candidates'] if c != value[0]['reference_candidate'])
    else: value[0]['case_max_slot_risk'] += .1
    write(path, value); seal(args[3])
    guard_reference(args, monkeypatch, allow=True)
    with pytest.raises(ValueError): module.analyze(*args)
    assert (args[4] / 'scores.json').is_file() and not (args[4] / 'complete.json').exists()


@pytest.mark.parametrize('target', ['reference', 'output'])
def test_final_reference_seal_or_output_mutation_revokes_success(assignment_run, monkeypatch, target):
    args = assignment_run(); original = module.shared._seal_cpu
    def sealing(root, names, hashes):
        original(root, names, hashes)
        if root == args[4]:
            path = args[3] / 'seal.json' if target == 'reference' else root / 'summary.json'
            if target == 'reference': path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            else: write(path, {'changed': True})
    monkeypatch.setattr(module.shared, '_seal_cpu', sealing)
    with pytest.raises(ValueError): module.analyze(*args)
    assert (args[4] / 'scores.json').is_file()
    assert not (args[4] / 'complete.json').exists() and not (args[4] / 'seal.json').exists()


def pure_inputs(mode='collapse_correct'):
    rows, _, _, answers = synthetic48()
    prepared = profile._prepare([Sample(**row) for row in rows])
    executions, responses = [], {}
    for key, query in prepared['queries'].items():
        names = query['candidates']
        events = prepared['sources'][data.text_hash(query['source'])]['events']
        row = events.index(query['event'])
        first_reference = names.index(answers[query['source'], events[0]])
        matrix = [[-1., -4.], [-1., -2.]]
        if first_reference == 1: matrix = [values[::-1] for values in matrix]
        if mode == 'collapse_wrong': matrix = [values[::-1] for values in matrix]
        elif mode == 'tie': matrix = [[-2., -2.], [-2., -2.]]
        for column, name in enumerate(names):
            execution_key = f'{key}:{column}'
            executions.append(dict(query_key=key, candidate=name, key=execution_key))
            responses[execution_key] = dict(joint_log_probability=matrix[row][column], candidate_tokens=2,
                actual_forward_passes=1, input_tokens=10, target_tokens=3, seconds=.1)
    complete = profile._completion(prepared, executions, responses, profile.DEFAULT_PROTOCOL)
    return prepared, executions, responses, complete, answers


@pytest.mark.parametrize('mode', ['collapse_correct', 'collapse_wrong', 'tie'])
def test_fixed_joint_can_succeed_fail_or_tie_without_selecting_a_readout(mode, monkeypatch):
    prepared, executions, responses, complete, answers = pure_inputs(mode)
    calls, original = [], module.assignment_distribution
    assert original is bijection_readout.assignment_distribution
    def observed(matrix, **kwargs):
        calls.append((deepcopy(matrix), kwargs))
        return original(matrix, **kwargs)
    monkeypatch.setattr(module, 'assignment_distribution', observed)
    scores = module._scores(prepared, executions, responses, complete)
    assert len(calls) == 24 and len(scores['matrices']) == 24
    assert all(kw['source_is_admitted_bijection'] is True for _, kw in calls)
    for matrix in scores['matrices'].values():
        assert matrix['events'] == sorted(matrix['events']) and matrix['candidates'] == sorted(matrix['candidates'])
        joint = matrix['bijection']
        if mode == 'tie':
            assert joint['tie'] is True and joint['map_assignment'] is None
        else:
            source = prepared['sources'][matrix['source_key']]['source']
            reference = [matrix['candidates'].index(answers[source, event]) for event in matrix['events']]
            assert (joint['map_assignment'] == reference) == (mode == 'collapse_correct')
            assert len({row['candidate_argmax_index'] for row in matrix['independent_rows']}) == 1
    assert scores['evaluation_material_accessed'] is False and scores['synthetic_offsets_applied'] is False


def test_nonbijection_original_mapping_is_retained_unknown_with_risk_one():
    prepared, executions, responses, _, _ = pure_inputs()
    prepared['originals'][0]['alignment']['slots'][1]['text'] = prepared['originals'][0]['alignment']['slots'][0]['text']
    complete = profile._completion(prepared, executions, responses, profile.DEFAULT_PROTOCOL)
    scores = module._scores(prepared, executions, responses, complete)
    row = scores['records'][0]
    assert row['observed_mapping_is_bijection'] is False
    assert row['bijection_joint_assignment_mass']['unknown_fallback'] is True
    assert row['bijection_joint_assignment_mass']['risk'] == 1.
    assert row['bijection_joint_assignment_mass']['unknown_reason'] == 'response_mapping_not_bijection'
    assert len(scores['records']) == 48


def test_wrong_suffix_is_rejected_without_using_reference_factors():
    prepared, executions, responses, complete, _ = pure_inputs()
    prepared['originals'][0]['sample']['response'] += ' Unsupported extra suffix.'
    with pytest.raises(ValueError, match='suffix'): module._scores(prepared, executions, responses, complete)
