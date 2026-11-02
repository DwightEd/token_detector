"""New cache-ablation checks using only the H008 suite's synthetic archives."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, reconstruction_assignment as module
from experiments import reconstruction_evaluation as baseline
from experiments import reconstruction_qualification as observer
from experiments.research import bijection_readout
from tests.test_reconstruction_evaluation import make_run as h008_run, write, seal


@pytest.fixture
def assignment_run(h008_run, monkeypatch):
    def build(preference='correct'):
        args = h008_run(preference)
        source = baseline.research_identity()
        source['source_sha256'].update({
            'experiments/reconstruction_evaluation.py': data.file_hash(Path(baseline.__file__)),
            'experiments/reconstruction_assignment.py': data.file_hash(Path(module.__file__)),
            'experiments.research/bijection_readout.py': data.file_hash(Path(bijection_readout.__file__))})
        for target in (baseline, module):
            monkeypatch.setattr(target, 'research_identity', lambda: deepcopy(source))
            monkeypatch.setattr(target, 'completed_identity', lambda start: start)
        baseline.evaluate(*args)
        protocol = args[6].parent / 'assignment-protocol.json'
        write(protocol, module.DEFAULT_PROTOCOL)
        return args[0], args[3], args[4], args[5], args[5].with_name('assignment'), protocol
    return build


def guard_references(args, monkeypatch, allow=False):
    original = Path.open; accesses = []
    def guarded(path, *rest, **kwargs):
        if path.is_relative_to(args[3]):
            accesses.append(path)
            assert allow, 'H008 reference tree reached before the H009 score freeze'
            freeze = data.read_json(args[4] / 'freeze.json')
            assert set(freeze) == {'plan.json', 'protocol.json', 'source_identity.json', 'scores.json'}
            assert data.read_json(args[4] / 'plan.json')['state'] == 'scores_frozen_before_reference_release'
            assert data.read_json(args[4] / 'scores.json')['evaluation_material_accessed'] is False
        return original(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    return accesses


def test_all_scores_frozen_before_references_and_all_fixed_arms_retained(assignment_run, monkeypatch):
    args = assignment_run()
    before = [{str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
              for root in (args[1], args[3])]
    accesses = guard_references(args, monkeypatch, allow=True)
    summary = module.analyze(*args)
    assert accesses and summary['new_model_calls'] == 0 and summary['new_language_observations'] is False
    assert summary['winning_readout_selected'] is False and summary['method_promoted'] is False
    assert summary['original_h008_summary'] == data.read_json(args[3] / 'summary.json')
    assert summary['donor_costs']['actual_forward_passes'] == 32
    assert summary['invariance']['sources'] == summary['invariance']['delta_invariant_sources'] == 8
    assert summary['invariance']['probability_invariant_sources'] == summary['invariance']['map_and_tie_invariant_sources'] == 8
    for arm in module.DEFAULT_PROTOCOL['arms']:
        joint = summary['arms'][arm]['bijection_joint_assignment_mass']
        assert joint['source_map'] == dict(total=8, correct=8, wrong=0, ties=0, unknown=0)
        assert joint['query_marginal_argmax'] == dict(total=16, correct=16, wrong=0, ties=0, unknown=0)
        assert joint['secondary_case_metrics']['auroc'] == joint['secondary_case_metrics']['ap'] == 1.
    independent = summary['arms']['fixed_additive_nuisance']['independent_rows_max_slot']
    assert independent['source_map']['correct'] == 0 and independent['query_marginal_argmax']['correct'] == 8
    assert len(data.read_json(args[4] / 'sources.json')) == 16
    assert len(data.read_json(args[4] / 'queries.json')) == 32
    assert len(data.read_json(args[4] / 'cases.json')) == 96
    assert all(data.read_json(args[4] / name) == data.read_json(args[3] / name.replace('h008-', ''))
               for name in ('h008-summary.json', 'h008-queries.json', 'h008-cases.json'))
    after = [{str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
             for root in (args[1], args[3])]
    assert before == after
    data.verify_seal(args[4])


@pytest.mark.parametrize('damage', ['root_extra', 'child_extra', 'child_seal_escape'])
def test_donor_inventory_rejected_before_hash_source_or_reference_access(assignment_run, monkeypatch, damage):
    args = assignment_run()
    if damage == 'root_extra': (args[1] / 'extra').write_text('no hash', encoding='utf-8')
    else:
        child = next((args[1] / 'requests').iterdir())
        if damage == 'child_extra': (child / 'extra').write_text('no hash', encoding='utf-8')
        else:
            value = data.read_json(child / 'seal.json'); value['../outside'] = 'f' * 64
            write(child / 'seal.json', value); seal(args[1])
    accesses = guard_references(args, monkeypatch)
    def forbidden(*a, **k): raise AssertionError('Donor payload/source read before path admission')
    monkeypatch.setattr(data, 'file_hash', forbidden); monkeypatch.setattr(module, 'research_identity', forbidden)
    with pytest.raises(ValueError): module.analyze(*args)
    assert accesses == [] and not args[4].exists()


@pytest.mark.parametrize('damage', ['offset', 'temperature', 'extra', 'alias'])
def test_strict_protocol_and_reference_alias_rejected_before_reference_open(assignment_run, monkeypatch, damage):
    args = list(assignment_run())
    protocol = data.read_json(args[5])
    if damage == 'offset': protocol['nuisance_column_offsets'] = [-1, -8]
    elif damage == 'temperature': protocol['temperature'] = .5
    elif damage == 'extra': protocol['selected_best_arm'] = 'joint'
    else: args[5] = args[3] / 'evaluation-protocol.json'
    if damage != 'alias': write(args[5], protocol)
    accesses = guard_references(args, monkeypatch)
    with pytest.raises(ValueError): module.analyze(*args)
    assert accesses == [] and not args[4].exists()


def test_wrong_reference_donor_seal_rejected_after_scores_are_frozen(assignment_run):
    args = assignment_run()
    value = data.read_json(args[3] / 'complete.json'); value['observation_initial_seal_sha256'] = 'f' * 64
    write(args[3] / 'complete.json', value); seal(args[3])
    with pytest.raises(ValueError, match='another donor'): module.analyze(*args)
    assert (args[4] / 'scores.json').is_file() and (args[4] / 'freeze.json').is_file()
    assert not (args[4] / 'complete.json').exists()


@pytest.mark.parametrize('damage', ['reference_candidate', 'h008_metric'])
def test_resealed_reference_values_must_reconstruct_original_h008_outputs(assignment_run, damage):
    args = assignment_run()
    path = args[3] / ('queries.json' if damage == 'reference_candidate' else 'summary.json')
    value = data.read_json(path)
    if damage == 'reference_candidate': value[0]['reference_candidate'] = value[0]['candidates'][1 - value[0]['candidates'].index(value[0]['reference_candidate'])]
    else: value['source_qa']['correct'] -= 1
    write(path, value); seal(args[3])
    with pytest.raises(ValueError, match='reconstruction changed'): module.analyze(*args)
    assert (args[4] / 'scores.json').is_file() and not (args[4] / 'complete.json').exists()


@pytest.mark.parametrize('target', ['donor', 'reference', 'output'])
def test_final_initial_seal_or_output_mutation_revokes_completion(assignment_run, monkeypatch, target):
    args = assignment_run(); original = baseline._seal_cpu
    def sealing(root, names, hashes):
        original(root, names, hashes)
        if root == args[4]:
            if target == 'output': write(root / 'summary.json', {'changed': True})
            else:
                path = args[1 if target == 'donor' else 3] / 'seal.json'
                path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    monkeypatch.setattr(baseline, '_seal_cpu', sealing)
    with pytest.raises(ValueError): module.analyze(*args)
    assert (args[4] / 'freeze.json').is_file()
    assert not (args[4] / 'complete.json').exists() and not (args[4] / 'seal.json').exists()


def test_explicit_nonbijection_response_mapping_stays_unknown_one(assignment_run):
    args = assignment_run()
    prepared, executions, responses, complete, _, _ = baseline._validate(
        args[0], args[1], args[2], module.research_identity())
    first = prepared['originals'][0]
    first['alignment']['slots'][1]['text'] = first['alignment']['slots'][0]['text']
    complete = observer._completion(prepared, executions, responses, data.read_json(args[1] / 'protocol.json'))
    result = module._scores(prepared, executions, responses, complete, module.DEFAULT_PROTOCOL)
    for arm in result['arms'].values():
        case = arm['records'][0]
        assert case['observed_mapping_is_bijection'] is False
        assert case['bijection_joint_assignment_mass']['unknown_fallback'] is True
        assert case['bijection_joint_assignment_mass']['risk'] == 1.
        assert case['bijection_joint_assignment_mass']['unknown_reason'] == 'response_mapping_not_bijection'


def test_independent_ties_do_not_inflate_source_or_marginal_correctness(assignment_run):
    args = assignment_run('tie')
    summary = module.analyze(*args)
    for name in module.DEFAULT_PROTOCOL['readouts']:
        counts = summary['arms']['baseline'][name]
        assert counts['source_map']['correct'] == 0 and counts['source_map']['ties'] == 8
        assert counts['query_marginal_argmax']['correct'] == 0 and counts['query_marginal_argmax']['ties'] == 16
    assert summary['original_h008_summary']['source_qa']['ties'] == 16


@pytest.mark.parametrize('name', ['construction_truth', 'evaluation_only', 'annotations'])
def test_external_protocol_label_bundle_is_rejected_before_any_payload_access(tmp_path, monkeypatch, name):
    def forbidden(*a, **k): raise AssertionError('Label-bundle protocol reached a payload read')
    monkeypatch.setattr(data, 'file_hash', forbidden); monkeypatch.setattr(data, 'read_json', forbidden)
    with pytest.raises(ValueError, match='evaluation-label bundle'):
        module.analyze(tmp_path / 'inference' / 'inputs.jsonl', tmp_path / 'donor', tmp_path / 'model',
            tmp_path / 'reference', tmp_path / 'outputs' / 'result', tmp_path / name / 'protocol.json')


def test_subnormal_energy_order_controls_marginal_decision_despite_rounded_equal_mass():
    candidates, events = ['Ana Vale', 'Bex Reed'], ['Amber', 'Cobalt']
    prepared = dict(sources={}, queries={}, originals=[])
    executions, responses = [], {}
    tiny = float.fromhex('0x0.0000000000001p-1022')
    for source in range(8):
        source_key = str(source)
        queries = [dict(query_key=f'{source}-{i}', event=event, candidates=candidates)
                   for i, event in enumerate(events)]
        prepared['sources'][source_key] = dict(status='ok', people=candidates, events=events, queries=queries)
        for i, query in enumerate(queries):
            prepared['queries'][query['query_key']] = query
            for j, candidate in enumerate(candidates):
                key = f'{source}-{i}-{j}'
                executions.append(dict(query_key=query['query_key'], candidate=candidate, key=key))
                responses[key] = dict(joint_log_probability=-tiny if i == j == 0 else 0.)
        for case in range(6):
            prepared['originals'].append(dict(source_key=source_key,
                sample=dict(id=f'{source}-{case}', source_id=source_key),
                alignment=dict(slots=[dict(event=event, text=candidate)
                    for event, candidate in zip(events, candidates, strict=True)])))
    complete = dict(records=[dict(case_max_slot_risk=.5) for _ in range(48)], coverage={}, costs={})
    scores = module._scores(prepared, executions, responses, complete, module.DEFAULT_PROTOCOL)
    for source in scores['arms']['baseline']['sources'].values():
        assert source['bijection']['tie'] is False and source['bijection']['map_assignment'] == [1, 0]
        assert source['bijection']['assignment_probabilities'] == [.5, .5]
        assert [row['marginal_argmax_index'] for row in source['independent_rows']] == [1, 0]
        assert all(not row['marginal_tie'] and row['rounded_marginal_probability_tie']
                   for row in source['independent_rows'])
