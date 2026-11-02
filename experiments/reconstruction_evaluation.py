"""CPU/tokenizer-only review of sealed source-role completion observations.

Reference design and construction truth are opened only after all observations
are reconstructed and the CPU evaluator is frozen. This narrow, exposed toy is
not natural coverage, a calibrated detector, or a selected main method.
"""
from collections import defaultdict
from copy import deepcopy
from importlib.metadata import version
import math
import os
from pathlib import Path

from . import data, reconstruction_qualification as observer
from .answer_completion import completion_metadata
from .datasets import read_jsonl, read_samples
from . import focused_verdict_evaluation as prior, verdict_evaluation as reader
from .provenance import completed_identity, research_identity
from .semantic_qualification import HFGenerator


DEFAULT_PROTOCOL = dict(schema='source-reconstruction-evaluation-v1',
    purpose='exposed_construction_source_role_qualification', required_cases=48,
    required_sources=8, required_queries=16, required_completions=32, required_slots=96,
    sourceflip_pairs=8, eventflip_pairs=8, model_calls_allowed=False,
    tokenizer_only_validation=True, primary='raw_joint_log_probability_flip_directions',
    secondary_case_risk='maximum_of_two_frozen_slot_risks',
    candidate_length_normalization=False, fitted_calibration_allowed=False,
    threshold_search_allowed=False, target_label_selection_allowed=False,
    natural_task_performance_claim=False, population_previously_exposed=True)
_HELPERS = ('experiments/reconstruction_qualification.py', 'experiments/answer_completion.py',
    'experiments.research/source_reconstruction.py', 'experiments.research/focused_verdict.py',
    'experiments/semantic_qualification.py', 'experiments/verdict_evaluation.py', 'experiments/data.py',
    'experiments/datasets.py', 'experiments/provenance.py')
_META = {'plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json',
         'freeze.json', 'complete.json'}
_CPU_FROZEN = {'plan.json', 'protocol.json', 'source_identity.json', 'freeze.json'}
_require = observer._require


def _paths(inputs, design, construction, observations, model_path, output, protocol):
    paths = dict(zip(('inputs', 'design', 'construction', 'root', 'model', 'output', 'protocol'),
        (Path(p).absolute() for p in (inputs, design, construction, observations, model_path, output, protocol))))
    _require(all(p.resolve() == p for p in paths.values()), 'Unaliased reconstruction evaluation paths required')
    i, d, c, r, m, o, p = (paths[k] for k in ('inputs', 'design', 'construction', 'root', 'model', 'output', 'protocol'))
    trees = (i.parent, c.parent, r, m, o)
    _require(i.name == 'inputs.jsonl' and d == i.parent / 'plan.json'
        and c.name == 'construction_truth.jsonl' and 'outputs' in o.parts
        and all(not a.is_relative_to(b) and not b.is_relative_to(a)
            for index, a in enumerate(trees) for b in trees[index + 1:])
        and all(not p.is_relative_to(tree) for tree in trees)
        and p.name.casefold() not in observer.ANNOTATION_NAMES | {'construction_truth.jsonl'},
        'Independent inference/observation/model/reference/output trees and external protocol required')
    if o.exists():
        raise FileExistsError('Retained reconstruction review must not be overwritten')
    return paths


def _inventory(root):
    """Admit every nested path and seal key before hashing archive payloads."""
    _require(root.is_dir() and root.resolve() == root
        and {p.name for p in root.iterdir()} == _META | {'requests', 'seal.json'},
        'Incomplete or aliased reconstruction archive')
    members, children = set(_META), []
    for name in _META | {'seal.json'}:
        reader._member(root, name)
    requests = reader._member(root, 'requests', directory=True)
    for entry in requests.iterdir():
        _require(reader._sha(entry.name), 'Invalid reconstruction request directory')
        child = reader._member(requests, entry.name, directory=True)
        expected = {'request.json', 'backend_raw.json', 'response.json'}
        _require({p.name for p in child.iterdir()} == expected | {'seal.json'},
                 'Unexpected or incomplete reconstruction request members')
        for name in expected | {'seal.json'}:
            members.add(str(reader._member(child, name).relative_to(root)))
        children.append((child, expected))
    _require(len(children) == 32, 'All32 completion request directories required')
    seal, raw = reader._seal_content(root, members)
    for child, expected in children:
        reader._seal_content(child, expected)
    return members, seal, data.text_hash(raw.decode('utf-8')), children


def _snapshot(root, inventory):
    current = _inventory(root)
    _require(current[0] == inventory[0] and current[2] == inventory[2],
             'Observation archive inventory or initial seal changed')
    reader._verify_observation_snapshot(root, *inventory)


def _cpu_tokenizer(model_path, identity):
    """Verify assets once, then load a local fast tokenizer; never prepare HF GPU."""
    verifier = HFGenerator(model_path, identity, admit_resources=False)
    verifier.verify_frozen_identity()  # Hash the complete frozen model manifest once.
    _require(verifier.model is None and verifier.tokenizer is None, 'Identity validation constructed runtime objects')
    files = identity['model_files_sha256']
    stamps = {name: ((model_path / name).stat().st_size, (model_path / name).stat().st_mtime_ns)
              for name in files}
    small = {name: digest for name, digest in files.items() if not name.endswith('.safetensors')}
    _require(version('transformers') == identity['runtime']['transformers'], 'Pinned tokenizer Transformers required')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    _require(tokenizer.is_fast, 'The frozen local fast tokenizer is required')

    def unchanged():
        _require({str(p.relative_to(model_path)) for p in model_path.rglob('*') if p.is_file()} == set(files)
            and all(not (model_path / name).is_symlink() and
                ((model_path / name).stat().st_size, (model_path / name).stat().st_mtime_ns) == stamp
                for name, stamp in stamps.items())
            and all(data.file_hash(model_path / name) == digest for name, digest in small.items()),
            'Model/tokenizer assets changed after initial identity validation')
    unchanged()
    return tokenizer, unchanged


def _validate(inputs, root, model_path, current):
    plan = data.read_json(root / 'plan.json')
    protocol = observer._protocol(root / 'protocol.json')
    model = observer._model_identity(data.read_json(root / 'model_identity.json'))
    source = data.read_json(root / 'source_identity.json')
    for name in _HELPERS:
        digest = source.get('source_sha256', {}).get(name)
        _require(reader._sha(digest) and digest == current.get('source_sha256', {}).get(name),
                 'Incompatible archived reconstruction helper: ' + name)
    input_hash = data.file_hash(inputs)
    _require(input_hash == protocol['original_inputs_sha256'] and reader._sha(plan.get('protocol_sha256')),
             'Frozen input or original protocol identity changed')
    prepared = observer._prepare(list(read_samples(inputs)))
    tokenizer, model_guard = _cpu_tokenizer(model_path, model)
    executions = []
    for row in prepared['candidates']:
        metadata = observer._metadata(completion_metadata(tokenizer, deepcopy(row['messages']), row['candidate']),
                                      row['candidate'])
        _require(metadata['input_tokens'] <= protocol['max_input_tokens'], 'Untruncated completion budget changed')
        execution = observer._execution(row, metadata, plan['protocol_sha256'], model, source)
        executions.append(dict(query_key=row['query_key'], candidate=row['candidate'],
                               key=observer._digest(execution), execution=execution))
    _require(len(executions) == 32 and len({e['key'] for e in executions}) == 32,
             'All32 unique completion requests required')
    expected = observer._plan(inputs, input_hash, plan['protocol_sha256'], protocol, model, source, prepared, executions)
    reader._equal(plan, expected, 'Full source-only query/slot/tokenizer/request plan changed')
    _require({p.name for p in (root / 'requests').iterdir()} == {e['key'] for e in executions},
             'Completion request population changed')
    freeze = data.read_json(root / 'freeze.json')
    expected_names = {'plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json'} | {
        str(Path('requests') / e['key'] / 'request.json') for e in executions}
    _require(isinstance(freeze, dict) and set(freeze) == expected_names
        and all(reader._sha(value) for value in freeze.values())
        and all(data.file_hash(root / name) == value for name, value in freeze.items()),
        'Pre-observation completion freeze changed')
    responses = {}
    for row in executions:
        child = root / 'requests' / row['key']
        reader._equal(data.read_json(child / 'request.json'), row['execution'], 'Frozen completion request changed')
        wrapper = data.read_json(child / 'backend_raw.json')
        _require(isinstance(wrapper, dict) and set(wrapper) == {'value'}, 'Invalid retained completion raw wrapper')
        response = observer._response(wrapper['value'], row['execution']['preflight'])
        reader._equal(data.read_json(child / 'response.json'), response, 'Token log-probability/joint sum/cost changed')
        responses[row['key']] = response
    complete = observer._completion(prepared, executions, responses, protocol)
    reader._equal(data.read_json(root / 'complete.json'), complete, 'Completion distributions/readouts/costs changed')
    model_guard()
    return prepared, executions, responses, complete, plan, model_guard


def _cpu_inventory(output, names, *, sealed=False):
    _require(output.resolve() == output and output.is_dir()
        and {p.name for p in output.iterdir()} == set(names) | ({'seal.json'} if sealed else set()),
        'Unexpected or incomplete CPU reconstruction review members')
    for name in set(names) | ({'seal.json'} if sealed else set()):
        reader._member(output, name)
    if sealed:
        reader._seal_content(output, names)


def _seal_cpu(output, names, hashes):
    _cpu_inventory(output, names)
    _require(set(hashes) == set(names), 'CPU seal must retain the exact recorded payload hashes')
    data.write_json(output / 'seal.json', hashes)


def _freeze_cpu(output, plan, protocol, source):
    output.mkdir(parents=True)
    metadata = [('plan.json', plan), ('protocol.json', protocol), ('source_identity.json', source)]
    for name, value in metadata:
        data.write_json(output / name, value)
    _cpu_inventory(output, _CPU_FROZEN - {'freeze.json'})
    for name, value in metadata:
        reader._equal(data.read_json(output / name), value, 'CPU metadata differs from prepared evaluator freeze')
    frozen = {name: data.file_hash(output / name) for name in ('plan.json', 'protocol.json', 'source_identity.json')}
    data.write_json(output / 'freeze.json', frozen)
    _cpu_inventory(output, _CPU_FROZEN)
    return frozen, data.file_hash(output / 'freeze.json')


def _check_cpu_freeze(output, snapshot, names=None, *, sealed=False):
    _cpu_inventory(output, _CPU_FROZEN if names is None else names, sealed=sealed)
    _require(isinstance(snapshot, tuple) and len(snapshot) == 2, 'CPU freeze must precede reference access')
    frozen, digest = snapshot
    _require(data.file_hash(output / 'freeze.json') == digest
        and data.read_json(output / 'freeze.json') == frozen
        and all(data.file_hash(output / name) == value for name, value in frozen.items()),
        'Frozen CPU evaluator metadata changed')


def _reference_snapshot(paths, cpu_freeze):
    """First reference seal/hash/read happens here, after the CPU freeze check."""
    _check_cpu_freeze(paths['output'], cpu_freeze)
    snapshots = []
    for parent, names in [(paths['inputs'].parent, {'inputs.jsonl', 'plan.json'}),
                          (paths['construction'].parent, {'construction_truth.jsonl'})]:
        _require(parent.resolve() == parent and parent.is_dir()
            and {p.name for p in parent.iterdir()} == names | {'seal.json'},
            'Exact independent reference bundle inventory required')
        for name in names | {'seal.json'}:
            reader._member(parent, name)
        seal, raw = reader._seal_content(parent, names)
        snapshots.append((parent, names, seal, data.text_hash(raw.decode('utf-8')), []))
    # Both complete inventories have been admitted before any reference payload hash.
    for snapshot in snapshots:
        reader._verify_observation_snapshot(*snapshot)
    return snapshots


def _references(prepared, design, truth):
    originals = prepared['originals']
    rows = design.get('records') if isinstance(design, dict) else None
    ids = [o['sample']['id'] for o in originals]
    _require(isinstance(design, dict) and design.get('schema') == 'h007-burden-controls-plan-v1'
        and isinstance(rows, list) and len(rows) == 48 and [r.get('id') for r in rows] == ids
        and len(truth) == 48 and [r.get('id') for r in truth] == ids,
        'Complete ordered original48 design and construction truth required')
    references, counts, source_groups, cells = defaultdict(set), defaultdict(int), defaultdict(list), set()
    for original, row, label in zip(originals, rows, truth, strict=True):
        sample, alignment = original['sample'], original['alignment']
        cell = row['factor_cell']
        key = tuple(cell[n] for n in ('relation_family', 'vocabulary', 'source_direction', 'response_direction', 'distractor_count'))
        _require(key not in cells, 'Duplicate construction cell'); cells.add(key)
        _require(row['source_id'] == sample['source_id'] and row['input_text_sha256'] == original['input_text_sha256']
            and row['target_span'] == alignment['target_span'] and row['target_text'] == alignment['target_text']
            and row['whole_span'] == [0, len(sample['response'])]
            and cell['relation_family'] == prepared['sources'][original['source_key']]['relation_family']
            and label.get('source_id') == sample['source_id']
            and label.get('source_sha256') == original['input_text_sha256']['source']
            and label.get('response_sha256') == original['input_text_sha256']['response']
            and label.get('target_span') == alignment['target_span']
            and label.get('target_outcome') in ('supported', 'contradicted')
            and label.get('whole_outcome') == label['target_outcome']
            and label.get('filler_outcomes') == ['supported'] * cell['distractor_count']
            and label.get('target_benchmark_labels_used') is False
            and label.get('construction_rule') == 'entity_event_bijections_equal_iff_source_direction_equals_response_direction'
            and (label['target_outcome'] == 'supported') == (cell['source_direction'] == cell['response_direction']),
            'Reference source/response/target/filler/factor identity changed')
        source_groups[original['source_key']].append((original, row, label))
        if label['target_outcome'] == 'supported':
            for slot in alignment['slots']:
                references[slot['query_key']].add(slot['text']); counts[slot['query_key']] += 1
    expected_cells = {(f, v, s, r, b) for f in ('winner', 'sole_signer') for v in ('lex1', 'lex2')
                      for s in (0, 1) for r in (0, 1) for b in (0, 4, 12)}
    _require(cells == expected_cells and set(references) == set(prepared['queries'])
        and all(len(values) == 1 and counts[key] == 3 for key, values in references.items()),
        'Supported references must agree across all three burdens for every query')
    references = {key: next(iter(values)) for key, values in references.items()}
    source_metadata = {}
    for source_key, group in source_groups.items():
        factors = {(r['factor_cell']['relation_family'], r['factor_cell']['vocabulary'],
                    r['factor_cell']['source_direction']) for _, r, _ in group}
        _require(len(group) == 6 and len(factors) == 1, 'Source factor ownership changed')
        family, vocabulary, direction = next(iter(factors))
        query_keys = [key for key, query in prepared['queries'].items()
                      if data.text_hash(query['source']) == source_key]
        people = prepared['sources'][source_key]['people']
        _require(len(query_keys) == 2 and {references[key] for key in query_keys} == set(people),
                 'Every source must reference both distinct source people across its two events')
        source_metadata[source_key] = dict(relation_family=family, vocabulary=vocabulary,
                                          source_direction=direction, query_keys=query_keys)
        for original, row, label in group:
            if label['target_outcome'] == 'contradicted':
                _require(all(slot['text'] != references[slot['query_key']] for slot in original['alignment']['slots']),
                         'Both wrong-response slots must disagree with their supported references')
    return rows, truth, references, source_metadata


def _flip_pair(first, second, kind, fields):
    _require(first['candidates'] == second['candidates']
        and first['reference_candidate'] != second['reference_candidate'], 'Flip candidate/reference bijection changed')
    orientation = 1 if first['reference_candidate'] == first['candidates'][0] else -1
    a, b = first['raw_canonical_log_odds'], second['raw_canonical_log_odds']
    return dict(kind=kind, **fields, query_keys=[first['query_key'], second['query_key']],
        raw_canonical_log_odds=[a, b], expected_first_sign=orientation,
        oriented_raw_interaction=orientation * (a - b) / 2,
        positive_source_or_event_response=orientation * (a - b) > 0,
        correctly_reversed=orientation * a > 0 and orientation * b < 0,
        both_queries_correct=first['greedy_correct'] and second['greedy_correct'],
        tied_queries=int(first['tie']) + int(second['tie']))


def _analysis(prepared, executions, responses, complete, design, truth):
    rows, truth, references, source_metadata = _references(prepared, design, truth)
    queries = []
    for key, query in prepared['queries'].items():
        source_key = data.text_hash(query['source'])
        distribution = complete['queries'][key]['distribution']
        values = distribution['sequence_logprobs']; candidates = query['candidates']
        winners = [candidate for candidate in candidates if values[candidate] == max(values.values())]
        greedy = winners[0] if len(winners) == 1 else None
        log_mass = distribution['log_candidate_normalizer']; mass = math.exp(log_mass)
        queries.append(dict(query_key=key, source_key=source_key,
            **{k: v for k, v in source_metadata[source_key].items() if k != 'query_keys'},
            event=query['event'], question=query['question'], candidates=candidates,
            reference_candidate=references[key], greedy_candidate=greedy, tie=greedy is None,
            greedy_correct=greedy == references[key], sequence_logprobs=values,
            raw_canonical_log_odds=values[candidates[0]] - values[candidates[1]],
            raw_reference_margin=values[references[key]] - values[next(c for c in candidates if c != references[key])],
            conditional_pool_probabilities=distribution['probabilities'], log_candidate_mass=log_mass,
            candidate_mass=mass, candidate_mass_underflow=mass == 0,
            candidate_token_counts=complete['queries'][key]['candidate_token_counts'],
            candidate_mass_is_not_truth_probability=True, name_length_or_frequency_bias_removed=False))
    lookup = {query['query_key']: query for query in queries}
    scenario = defaultdict(dict)
    for source_key, metadata in source_metadata.items():
        key = metadata['relation_family'], metadata['vocabulary']
        scenario[key][metadata['source_direction']] = sorted(
            (lookup[q] for q in metadata['query_keys']), key=lambda q: q['event'])
    _require(len(scenario) == 4 and all(set(v) == {0, 1} for v in scenario.values()), 'All four sourceflip scenarios required')
    sourceflip, eventflip = [], []
    for (family, vocabulary), directions in scenario.items():
        _require([q['event'] for q in directions[0]] == [q['event'] for q in directions[1]], 'Sourceflip events changed')
        for first, second in zip(directions[0], directions[1], strict=True):
            sourceflip.append(_flip_pair(first, second, 'sourceflip',
                dict(relation_family=family, vocabulary=vocabulary, event=first['event'])))
        for direction, pair in directions.items():
            eventflip.append(_flip_pair(*pair, 'eventflip',
                dict(relation_family=family, vocabulary=vocabulary, source_direction=direction,
                     events=[q['event'] for q in pair])))
    _require(len(sourceflip) == len(eventflip) == 8, 'All8 sourceflip and8 eventflip pairs required')
    cases, outcomes, risks = [], [], []
    for row, label, record in zip(rows, truth, complete['records'], strict=True):
        _require(record['id'] == row['id'], 'Case score/reference order changed')
        outcome = int(label['target_outcome'] == 'contradicted')
        outcomes.append(outcome); risks.append(record['case_max_slot_risk'])
        cases.append(dict(record, factor_cell=row['factor_cell'], construction_binary_truth=outcome,
                          primary_evidence_is_query_level=True))
    def flip_summary(records):
        return dict(pairs=len(records), correctly_reversed_pairs=sum(r['correctly_reversed'] for r in records),
            positive_raw_interactions=sum(r['positive_source_or_event_response'] for r in records),
            records=records, positive_interaction_alone_is_not_correct_reconstruction=True)
    summary = dict(schema='source-reconstruction-diagnosis-v1', original_cases=48,
        source_qa=dict(unique_sources=8, unique_queries=16, correct=sum(q['greedy_correct'] for q in queries),
                      ties=sum(q['tie'] for q in queries), wrong=sum(not q['greedy_correct'] and not q['tie'] for q in queries)),
        sourceflip=flip_summary(sourceflip), eventflip=flip_summary(eventflip),
        secondary_case_metrics=prior._metrics(outcomes, risks, [r['unknown_fallback'] for r in complete['records']]),
        secondary_case_risk='maximum_of_two_frozen_slot_risks', costs=complete['costs'], coverage=complete['coverage'],
        model_calls=0, tokenizer_only_validation=True, completion_metadata_rebuilt=32,
        threshold_search_performed=False, target_label_selection_performed=False,
        candidate_length_bias_removed=False, method_promoted=False, natural_task_performance_claim=False,
        limitations=['Previously exposed48-case toy; only8 sources/16 unique QA observations, not48 independent reconstructions.',
            'Finite source grammar already carries toy semantics; no natural coverage is established.',
            'Raw joint sequence likelihood retains name length/frequency priors; positive interaction alone is not correct ranking.',
            'Two-candidate conditional mass and max-slot risk are not calibrated truth or joint factuality probabilities.'])
    raw = [dict(query_key=row['query_key'], candidate=row['candidate'], key=row['key'],
                response=responses[row['key']]) for row in executions]
    return summary, queries, cases, raw


def evaluate(inputs, design, construction, observations, model_path, output, protocol):
    paths = _paths(inputs, design, construction, observations, model_path, output, protocol)
    root, output = paths['root'], paths['output']
    inventory = _inventory(root); _snapshot(root, inventory)
    frozen = data.read_json(paths['protocol'])
    _require(isinstance(frozen, dict) and not set(frozen) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(frozen.get(k)) is type(v) and frozen[k] == v for k, v in DEFAULT_PROTOCOL.items()),
        'Prespecified source-reconstruction evaluation protocol required')
    protocol_hash, source = data.file_hash(paths['protocol']), research_identity()
    prepared, executions, responses, complete, plan, model_guard = _validate(paths['inputs'], root, paths['model'], source)
    cpu_plan = dict(schema='source-reconstruction-evaluation-plan-v1', state='frozen_before_reference_release',
        original_inputs_sha256=plan['original_inputs_sha256'], observation_root=str(root),
        observation_initial_seal_sha256=inventory[2], original_observation_protocol_sha256=plan['protocol_sha256'],
        evaluator_protocol_sha256=protocol_hash, reference_design_path=str(paths['design']),
        construction_truth_path=str(paths['construction']), source_identity_sha256=observer._digest(source),
        protocol=frozen, construction_truth_accessed=False, reference_design_accessed=False,
        model_calls=0, tokenizer_only_validation=True, completion_metadata_rebuilt=32)
    cpu_freeze = _freeze_cpu(output, cpu_plan, frozen, source)
    cpu_members = set(_CPU_FROZEN)
    result_hashes = {}
    cpu_seal_hash = None

    def unchanged():
        _check_cpu_freeze(output, cpu_freeze, cpu_members, sealed=cpu_seal_hash is not None)
        _require(all(data.file_hash(output / name) == digest for name, digest in result_hashes.items()),
                 'Previously written CPU reconstruction results changed')
        if cpu_seal_hash is not None:
            _require(data.file_hash(output / 'seal.json') == cpu_seal_hash, 'Initial CPU review seal changed')
        _snapshot(root, inventory); completed_identity(source)
        _require(data.file_hash(paths['inputs']) == plan['original_inputs_sha256']
            and data.file_hash(paths['protocol']) == protocol_hash, 'Original inputs/evaluator protocol changed')
        model_guard()
    unchanged()
    references = _reference_snapshot(paths, cpu_freeze)
    design = data.read_json(paths['design']); truth = list(read_jsonl(paths['construction']))
    summary, queries, cases, raw = _analysis(prepared, executions, responses, complete, design, truth)
    release = dict(reference_material_accessed_after_cpu_freeze=True,
        inference_seal_sha256=references[0][3], construction_seal_sha256=references[1][3],
        design_sha256=data.file_hash(paths['design']), construction_truth_sha256=data.file_hash(paths['construction']))

    def retain(name, value):
        data.write_json(output / name, value); cpu_members.add(name)
        _cpu_inventory(output, cpu_members)
        reader._equal(data.read_json(output / name), value, 'Written CPU result differs from reconstructed value')
        result_hashes[name] = data.file_hash(output / name)

    for name, value in [('reference-release.json', release), ('summary.json', summary), ('queries.json', queries), ('cases.json', cases),
                         ('candidate-observations.json', raw)]:
        retain(name, value)
    def final_check():
        unchanged()
        for snapshot in references:
            parent, names = snapshot[:2]
            _require({p.name for p in parent.iterdir()} == names | {'seal.json'},
                     'Reference bundle inventory changed after release')
            reader._verify_observation_snapshot(*snapshot)
    final_check()
    try:
        retain('complete.json', dict(schema='source-reconstruction-evaluation-complete-v1',
            state='sealed_source_reconstruction_diagnosis', cases=48, unique_queries=16, unique_completions=32,
            sourceflip_pairs=8, eventflip_pairs=8, model_calls=0, tokenizer_only_validation=True,
            evaluation_only_after_predictions_frozen=True, natural_task_performance_claim=False,
            observation_initial_seal_sha256=inventory[2]))
        final_check()
        expected_hashes = dict(cpu_freeze[0], **{'freeze.json': cpu_freeze[1]}, **result_hashes)
        _seal_cpu(output, cpu_members, expected_hashes)
        _cpu_inventory(output, cpu_members, sealed=True)
        reader._equal(data.read_json(output / 'seal.json'), expected_hashes, 'Final CPU seal differs from retained results')
        cpu_seal_hash = data.file_hash(output / 'seal.json')
        data.verify_seal(output, expected=cpu_members); final_check()
    except BaseException:
        (output / 'complete.json').unlink(missing_ok=True); (output / 'seal.json').unlink(missing_ok=True)
        raise
    return summary
