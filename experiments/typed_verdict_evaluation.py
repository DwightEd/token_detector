"""CPU-only, whole-population H004 review after every observation is verified."""

from collections import Counter
from pathlib import Path

from . import data, observation_reuse as reuse, verdict_evaluation as reader
from . import typed_verdict_qualification as typed, verdict_qualification as shared
from .datasets import read_jsonl, read_samples
from .provenance import research_identity, completed_identity
from .research import semantic_verdict as verdict


DEFAULT_PROTOCOL = dict(schema='typed-verdict-evaluation-v1', purpose='posthoc_review_only',
    required_cases=34, natural_cases=18, constructed_cases=16, conditions=['A','B','C','D'],
    permutations=6, whole_case_population=True, model_calls_allowed=False,
    tokenizer_calls_allowed=False, token_score_projection_allowed=False, auroc_allowed=False,
    automatic_natural_accuracy_allowed=False, target_label_selection_allowed=False,
    threshold_sweeps_allowed=False, fitted_calibration_allowed=False)
_HELPERS = ('experiments/typed_verdict_qualification.py', 'experiments.research/typed_source.py',
            'experiments.research/semantic_verdict.py', 'experiments/semantic_qualification.py',
            'experiments/verdict_qualification.py', 'experiments/observation_reuse.py')
_TRUTH = {'supported': 'supported', 'conflict': 'contradicted', 'not_determined': 'unresolved'}
_PANEL_FILES = {'inputs.jsonl', 'manifest.json', 'evaluation_only.jsonl'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inventory(root):
    expected = {'plan.json', 'complete.json', 'seal.json', *typed.CONDITIONS,
                *(arm+'-protocol.json' for arm in typed.CONDITIONS)}
    _require(root.resolve() == root and root.is_dir() and
             {p.name for p in root.iterdir()} == expected, 'Incomplete or aliased factorial archive')
    members, children = {'plan.json','complete.json'}, []
    for name in ('plan.json', 'complete.json', 'seal.json'):
        reader._member(root, name)
    for arm in typed.CONDITIONS:
        protocol = reader._member(root, arm+'-protocol.json')
        members.add(str(protocol.relative_to(root)))
        child = reader._member(root, arm, directory=True)
        submembers, seal, sealbytes, grandchildren = reader._observation_inventory(child)
        children.append((child, submembers, seal, data.text_hash(sealbytes.decode('utf-8')), grandchildren))
        members.update(str((child/name).relative_to(root)) for name in submembers | {'seal.json'})
    seal, sealbytes = reader._seal_content(root, members)
    return members, seal, data.text_hash(sealbytes.decode('utf-8')), children


def _snapshot(root, inventory):
    members, seal, digest, children = inventory
    reader._verify_observation_snapshot(root, members, seal, digest, [])
    for child, submembers, subseal, subhash, grandchildren in children:
        reader._verify_observation_snapshot(child, submembers, subseal, subhash, grandchildren)


def _identity(sample):
    return dict(id=sample.id, source_id=sample.source_id, task=sample.task,
                generator=sample.generator, inputs=sample.inputs)


def _validate(inputs, root, current):
    plan = data.read_json(root/'plan.json')
    protocol, source, model = plan['protocol'], plan['source_identity'], plan['model_identity']
    _require(plan.get('schema') == 'typed-verdict-factorial-plan-v1'
             and plan.get('evaluation_material_accessed') is False
             and reader._sha(plan.get('protocol_sha256'))
             and all(type(protocol.get(k)) is type(v) and protocol[k] == v
                     for k,v in typed.BASE_PROTOCOL.items())
             and not set(protocol)-set(typed.BASE_PROTOCOL)-{'notes','hypothesis_id'}, 'Invalid factorial plan/protocol')
    reader._equal(plan['effective_model_identity'], reuse.effective_model_identity(model), 'Effective model identity changed')
    for name in _HELPERS:
        _require(source.get('source_sha256',{}).get(name) == current.get('source_sha256',{}).get(name)
                 and reader._sha(source.get('source_sha256',{}).get(name)),
                 'Incompatible archived observation helper: '+name)
    _require(data.file_hash(inputs) == plan['inputs_sha256'], 'Combined original inputs changed')
    samples = list(read_samples(inputs))
    _require(len(samples) == 34 and len({s.id for s in samples}) == 34, 'Whole unique 34-case population required')
    donors, originals = {}, []
    _require(len(plan['donors']) == 2, 'Both H003 donor populations required')
    for binding, expected in zip(plan['donors'], protocol['donor_inputs_sha256'], strict=True):
        donor_root = Path(binding['path']).absolute()
        _require(binding['inputs_sha256'] == expected, 'Donor population changed')
        index = reuse.load_donor(donor_root, model, expected)
        _require(data.file_hash(donor_root/'seal.json') == binding['seal_sha256'], 'Donor seal changed')
        donor_plan = data.read_json(donor_root/'plan.json')
        originals.extend(read_samples(Path(donor_plan['inputs_path'])))
        for key, item in index.items():
            _require(key not in donors or donors[key]['response'] == item['response'], 'Conflicting donor observations')
            donors.setdefault(key,item)
    reader._equal([_identity(s) for s in samples], [_identity(s) for s in originals], 'Original donor roster changed')
    cases, root_rows, novel, observed = {}, [], set(), {}
    for sample in samples:
        context, transformation = shared._context(sample.inputs)
        row = dict(id=sample.id,source_id=sample.source_id,
            input_text_sha256={k:data.text_hash(v) for k,v in sample.inputs.items()},
            context_transformation=transformation,source_transformation=typed.typed_source.prepare_source(sample.source),conditions={})
        root_rows.append(row)
    for arm in typed.CONDITIONS:
        child = root/arm
        child_protocol_hash = data.file_hash(root/(arm+'-protocol.json'))
        child_protocol = dict(protocol,condition=arm,expected_cases=34,max_forward_passes=204)
        reader._equal(data.read_json(root/(arm+'-protocol.json')), child_protocol, 'Arm protocol changed')
        child_plan = data.read_json(child/'plan.json')
        reader._equal(data.read_json(child/'protocol.json'), child_protocol, 'Archived arm protocol changed')
        reader._equal(data.read_json(child/'source_identity.json'), source, 'Arm source identity changed')
        reader._equal(data.read_json(child/'model_identity.json'), model, 'Arm model identity changed')
        rows, executions = [], {}
        for sample, root_row in zip(samples, root_rows, strict=True):
            context, transformation = shared._context(sample.inputs)
            requests, logical = [], []
            for assignment in verdict.label_assignments():
                request = typed.build_request(sample.source,sample.response,context,assignment,arm)
                execution = dict(messages=request['messages'],labels=list(verdict.LABELS),model_identity=model,
                    protocol_sha256=child_protocol_hash,source_identity_sha256=shared._digest(source),
                    scoring=dict(use_cache=False,logits_to_keep=1,full_prefill=True,output_tokens=0),
                    admission=dict(max_input_tokens=4096))
                key, equivalent = shared._digest(execution), reuse.equivalence_key(execution)
                executions[key] = execution
                requests.append(dict(request_key=key,label_assignment=assignment))
                logical.append(dict(equivalence_key=equivalent,label_assignment=assignment,
                    origin='H003_donor' if equivalent in donors else 'new_or_exact_within_run_reuse'))
                if equivalent not in donors: novel.add(equivalent)
                _require(arm != 'A' or equivalent in donors, 'A requires exact H003 donor observations')
            root_row['conditions'][arm] = logical
            rows.append(dict(id=sample.id,source_id=sample.source_id,input_text_sha256=root_row['input_text_sha256'],
                pre_generation_context_sha256=data.text_hash(context),context_transformation=transformation,requests=requests))
        expected_plan = dict(schema='verdict-qualification-plan-v1',state='observations_only',inputs_path=str(inputs),
            inputs_sha256=plan['inputs_sha256'],protocol_sha256=child_protocol_hash,
            protocol=child_protocol,model_identity_sha256=shared._digest(model),source_identity_sha256=shared._digest(source),
            records=rows,evaluation_material_accessed=False,case_ids_are_model_inputs=False,engineering_failure_is_semantic_refutation=False)
        reader._equal(child_plan, expected_plan, 'Arm full-input/request plan changed')
        _require({p.name for p in (child/'requests').iterdir()} == set(executions)
                 and {p.name for p in (child/'cases').iterdir()} == {data.text_hash(s.id) for s in samples}, 'Arm population changed')
        responses = {key:shared._cached(child/'requests',key,execution) for key,execution in executions.items()}
        for key, response in responses.items():
            equivalent = reuse.equivalence_key(executions[key])
            if response['status'] != 'scored':
                _require(arm != 'A', 'A cannot replace a donor with input rejection')
                continue
            origin = response.get('observation_origin',{})
            bare = {k:v for k,v in response.items() if k != 'observation_origin'}
            mode = origin.get('mode')
            if mode == 'donor_reuse':
                _require(equivalent in donors, 'Unknown imported donor')
                reader._equal(bare, donors[equivalent]['response'], 'Imported donor response changed')
                reader._equal(origin,dict(mode=mode,equivalence_key=equivalent,donor=donors[equivalent]['provenance']), 'Donor origin changed')
            else:
                _require(arm != 'A' and equivalent not in donors, 'Required donor observation was rerun')
                reader._equal(origin,dict(mode=mode,equivalence_key=equivalent), 'Observation origin changed')
                if mode == 'observed':
                    _require(equivalent not in observed, 'Equivalent request was observed twice')
                    observed[equivalent] = bare
                else:
                    _require(mode == 'in_run_reuse' and equivalent in observed, 'Missing original in-run observation')
                    reader._equal(bare, observed[equivalent], 'In-run reused observation changed')
        arm_cases, records = [], []
        for sample,row in zip(samples,rows,strict=True):
            arm_cases.append(reader._case(sample,row,child,responses,child_protocol))
            path = child/'cases'/data.text_hash(sample.id)/'case.json'
            records.append(dict(id=sample.id,file=str(path.relative_to(child)),sha256=data.file_hash(path)))
        reader._equal(data.read_json(child/'complete.json'),reader._completion(arm_cases,responses,records,child_protocol), 'Arm completion changed')
        cases[arm] = arm_cases
    reader._equal(plan['records'],root_rows,'Factorial original source/context/condition roster changed')
    _require(plan['expected_new_unique_requests'] == len(novel) <= protocol['max_new_forward_passes'], 'Novel observation budget changed')
    expected_complete = dict(schema='typed-verdict-factorial-complete-v1',state='observations_sealed_not_evaluated',cases=34,
        conditions=list(typed.CONDITIONS),costs=typed._costs(root),source_identity_sha256=shared._digest(source),
        evaluation_material_accessed=False,auroc_computed=False)
    reader._equal(data.read_json(root/'complete.json'),expected_complete,'Factorial completion/costs changed')
    return samples, cases, plan


def _prediction(case):
    scores = case['verdict'].get('class_scores')
    if scores is None:return 'unknown'
    winners = [k for k,v in scores.items() if v == max(scores.values())]
    return winners[0] if len(winners) == 1 else 'unknown'


def _controls(panel, samples, seal):
    manifest = data.read_json(panel/'manifest.json')
    _require(manifest.get('schema') == 'constructed-observation-controls-v1'
        and manifest.get('counts') == {'inputs':16,'pairs':8,'exact_word_bag_pairs':6}
        and manifest.get('ids') == [s.id for s in samples]
        and manifest.get('inputs_sha256') == seal['inputs.jsonl']
        and manifest.get('evaluation_only_sha256') == seal['evaluation_only.jsonl'], 'Invalid constructed manifest')
    released = list(read_jsonl(panel/'evaluation_only.jsonl'))
    _require([r.get('id') for r in released] == [s.id for s in samples], 'Constructed label roster changed')
    for row,sample in zip(released,samples,strict=True):
        _require(row.get('construction_logical_outcome') in _TRUTH
            and row.get('source_sha256') == data.text_hash(sample.source)
            and row.get('response_sha256') == data.text_hash(sample.response)
            and ('source_id' not in row or row['source_id'] == sample.source_id), 'Constructed source/response join failed')
    pairs = manifest.get('pairs',[])
    index = {r['id']:r for r in released}
    _require(len(pairs) == 8 and len({p['pair_id'] for p in pairs}) == 8
        and Counter(i for p in pairs for i in p['ids']) == Counter(index.keys())
        and all(len(p['ids']) == 2 and all(index[i]['pair_id'] == p['pair_id'] for i in p['ids']) for p in pairs)
        and sum(p['bag_equality_required'] is True for p in pairs) == 6, 'Invalid constructed pair ownership')
    return manifest,released


def _control_summary(cases, labels, pairs):
    classes = verdict.SEMANTIC_CLASSES
    confusion = {truth:{pred:0 for pred in (*classes,'unknown')} for truth in classes}
    index, pair_rows = {}, []
    for case,label in zip(cases,labels,strict=True):
        truth = _TRUTH[label['construction_logical_outcome']]
        prediction = _prediction(case)
        confusion[truth][prediction] += 1
        index[case['id']] = (truth,case['verdict'].get('class_scores'))
    for pair in pairs:
        rows = [index[i] for i in pair['ids']]
        truths = [r[0] for r in rows]
        margin = None
        if truths.count('supported') == 1:
            si = truths.index('supported')
            if all(r[1] is not None for r in rows):margin = rows[si][1]['supported']-rows[1-si][1]['supported']
        else:_require(truths == ['supported','supported'], 'Unexpected constructed pair composition')
        pair_rows.append(dict(pair_id=pair['pair_id'],ids=pair['ids'],construction_classes=truths,
            q_supported={i:(index[i][1]['supported'] if index[i][1] is not None else None) for i in pair['ids']},
            margin=margin,strictly_positive=None if margin is None else margin>0,
            both_supported=truths == ['supported','supported'],same_word_bag=pair['bag_equality_required']))
    return dict(denominator=16,confusion=confusion,
        correct_fixed_argmax=sum(_prediction(c) == _TRUTH[r['construction_logical_outcome']] for c,r in zip(cases,labels,strict=True)),
        unknown_predictions=sum(_prediction(c) == 'unknown' for c in cases),pairs=pair_rows)


def _effects(cases, start, end):
    comparisons = {'B_minus_A':('B','A'),'C_minus_A':('C','A'),
                   'D_minus_C':('D','C'),'D_minus_B':('D','B')}
    rows = []
    for i in range(start,end):
        q = {arm:cases[arm][i]['verdict'].get('class_scores') for arm in typed.CONDITIONS}
        q = {arm:(value['supported'] if value is not None else None) for arm,value in q.items()}
        diffs = {name:(q[a]-q[b] if q[a] is not None and q[b] is not None else None) for name,(a,b) in comparisons.items()}
        diffs['interaction_D_minus_C_minus_B_plus_A'] = q['D']-q['C']-q['B']+q['A'] if all(v is not None for v in q.values()) else None
        rows.append(dict(id=cases['A'][i]['id'],q_supported=q,differences=diffs))
    return dict(denominator=end-start,rows=rows,descriptive_only=True,method_selected=False)


def evaluate(inputs, observations, natural_panel, controls_panel, output, protocol):
    """Verify 816 logical observations before releasing either evaluation panel."""
    inputs,root,natural,controls,output,protocol = (Path(p).absolute() for p in
        (inputs,observations,natural_panel,controls_panel,output,protocol))
    _require(all(p.resolve() == p for p in (inputs,root,natural,controls,output,protocol))
        and inputs.name == 'inputs.jsonl' and protocol.name.casefold() not in shared.ANNOTATION_NAMES,
        'Evaluation paths must be unaliased original inputs and independent protocol')
    trees = (root,natural,controls)
    _require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for i,a in enumerate(trees) for b in trees[i+1:])
        and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in trees)
        and all(not protocol.is_relative_to(p) for p in (*trees,output)), 'Evaluation artifact trees must be disjoint')
    if output.exists():raise FileExistsError('Retained evaluation output must not be overwritten')
    inventory = _inventory(root)  # Path/seal admission precedes every file hash.
    _snapshot(root,inventory)
    frozen = data.read_json(protocol)
    _require(isinstance(frozen,dict) and not set(frozen)-set(DEFAULT_PROTOCOL)-{'notes','hypothesis_id'}
        and all(type(frozen.get(k)) is type(v) and frozen[k] == v for k,v in DEFAULT_PROTOCOL.items()),
        'Prespecified whole-factorial evaluation protocol required')
    source, protocol_hash = research_identity(), data.file_hash(protocol)
    samples,cases,observer_plan = _validate(inputs,root,source)
    panels = []
    for panel,subset,expected_hash in ((natural,samples[:18],typed.BASE_PROTOCOL['donor_inputs_sha256'][0]),
                                      (controls,samples[18:],typed.BASE_PROTOCOL['donor_inputs_sha256'][1])):
        _require({p.name for p in panel.iterdir()} == _PANEL_FILES|{'seal.json'}, 'Panel requires exactly four original files')
        seal,sealbytes = reader._seal_content(panel,_PANEL_FILES)
        _require(seal['inputs.jsonl'] == expected_hash == data.file_hash(panel/'inputs.jsonl'), 'Panel original input identity changed')
        reader._equal([_identity(s) for s in read_samples(panel/'inputs.jsonl')],[_identity(s) for s in subset], 'Panel original roster changed')
        panels.append((panel,seal,data.text_hash(sealbytes.decode('utf-8'))))
    _snapshot(root,inventory)
    output.mkdir(parents=True)
    data.write_json(output/'plan.json',dict(schema='typed-verdict-evaluation-plan-v1',state='evaluator_frozen_before_release',
        inputs_sha256=observer_plan['inputs_sha256'],observation_root=str(root),observation_seal_sha256=inventory[2],
        panel_seals=[dict(path=str(p),seal_sha256=h,seal=s) for p,s,h in panels],
        original_ids=[s.id for s in samples],protocol_sha256=protocol_hash,
        evaluator_source_identity_sha256=shared._digest(source),evaluation_material_accessed=False))
    data.write_json(output/'protocol.json',frozen)
    data.write_json(output/'source_identity.json',source)
    frozen_files = {p.name:data.file_hash(p) for p in output.iterdir()}
    completed_identity(source)
    _snapshot(root,inventory)
    # RELEASE: all four arms, their donors and original populations now passed.
    for panel,seal,sealhash in panels:
        _require(data.file_hash(panel/'seal.json') == sealhash,'Panel resealed before release')
        reader._equal(data.verify_seal(panel,expected=_PANEL_FILES),seal,'Panel changed at release')
    natural_manifest = reader._manifest(natural,samples[:18],panels[0][1])
    natural_labels = reader._released_rows(natural,samples[:18],natural_manifest)
    control_manifest, control_labels = _controls(controls,samples[18:],panels[1][1])
    packets = output/'cases';packets.mkdir()
    records = []
    for arm in typed.CONDITIONS:
        for i,(sample,case) in enumerate(zip(samples,cases[arm],strict=True)):
            if i < 18:packet = reader._review(sample,case,natural_labels[i])
            else:packet = dict(id=sample.id,source_id=sample.source_id,original_inputs=sample.inputs,
                observations=case['observations'],verdict=case['verdict'],unscored_unknown=case['verdict'].get('status') == 'unknown',
                construction_class=_TRUTH[control_labels[i-18]['construction_logical_outcome']],
                released_evaluation_record=control_labels[i-18])
            packet.update(condition=arm,source_transformation=observer_plan['records'][i]['source_transformation'])
            path=packets/(data.text_hash(sample.id)+'-'+arm+'.json');data.write_json(path,packet)
            records.append(dict(id=sample.id,condition=arm,file=str(path.relative_to(output)),sha256=data.file_hash(path)))
    summary = dict(whole_case_denominator=34,conditions=list(typed.CONDITIONS),
        natural={arm:reader._summary(cases[arm][:18],natural_labels) for arm in typed.CONDITIONS},
        controls={arm:_control_summary(cases[arm][18:],control_labels,control_manifest['pairs']) for arm in typed.CONDITIONS},
        factorial_effects=dict(natural=_effects(cases,0,18),controls=_effects(cases,18,34)),
        model_calls=0,tokenizer_calls=0,auroc_computed=False,automatic_natural_accuracy_created=False,
        target_label_selection_performed=False,threshold_sweeps=0)
    data.write_json(output/'summary.json',summary)
    def unchanged():
        _snapshot(root,inventory)
        completed_identity(source)
        _require(data.file_hash(inputs) == observer_plan['inputs_sha256'] and data.file_hash(protocol) == protocol_hash,
                 'Original inputs/evaluator protocol changed during export')
        for name,digest in frozen_files.items():_require(data.file_hash(output/name) == digest,'Frozen evaluator metadata changed')
        for panel,seal,sealhash in panels:
            _require(data.file_hash(panel/'seal.json') == sealhash,'Panel resealed during export')
            reader._equal(data.verify_seal(panel,expected=_PANEL_FILES),seal,'Panel changed during export')
    unchanged()
    result=dict(schema='typed-verdict-evaluation-complete-v1',state='sealed_posthoc_review_human_natural_review_pending',
        cases=34,conditions=list(typed.CONDITIONS),packets=136,records=records,evaluation_material_accessed=True,
        model_calls=0,tokenizer_calls=0,auroc_computed=False,observation_root_seal_sha256=inventory[2])
    data.write_json(output/'complete.json',result)
    shared._seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
    try:
        data.verify_seal(output)
        unchanged()
    except Exception:
        (output/'complete.json').unlink(missing_ok=True)
        (output/'seal.json').unlink(missing_ok=True)
        raise
    return result
