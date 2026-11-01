"""A sealed source-certificate interaction probe, not a financial benchmark result."""

from collections import Counter
from pathlib import Path
import re
import time

import numpy as np

from experiments import data
from experiments.datasets import read_jsonl
from experiments.evaluate import token_labels
from experiments.fit_ablation import verify_seal
from experiments.native_dataset import read_native
from experiments.provenance import completed_identity, research_identity
from experiments.research.constraint_interaction import (
    belief_loo, belief_model, belief_scores, belief_worlds, interaction_states,
    normalized, numbers, prototype_scores, representations, witnesses,
)
from experiments.research.binding_witnesses import binding_pairs, binding_probes, binding_worlds

CASES = ['fred:finqa:test:53', 'fred:finqa:test:59', 'fred:tatqa:test:3',
         'fred:tatqa:test:50', 'fred:tatqa:test:51', 'fred:tatqa:test:54']


def variants(sample, factorial=False):
    text = sample.response
    if sample.id == 'fred:finqa:test:53':
        if text.count('decrease') != 1 or text.count('$8 million') != 1:
            raise ValueError('The declared Fin53 mechanism case changed')
        return [{'name': f'{direction}_{value}', 'text': text.replace('decrease', direction).replace(
            '$8 million', f'${value} million'), 'original': direction == 'decrease' and value == '8'}
            for direction in ['decrease', 'increase'] for value in ['8', '94']]
    if sample.id == 'fred:tatqa:test:3':
        if text.count('80.8') != 2 or text.count('USD million') != 1:
            raise ValueError('The declared TAT3 mechanism case changed')
        if factorial:
            pieces = text.split('80.8')
            return [{'name': f'p{prior}_s{summary}_{unit}', 'text':
                (pieces[0]+prior+pieces[1]+summary+pieces[2]).replace('USD million', unit),
                'original': prior == summary == '80.8' and unit == 'USD million'}
                for prior in ['80.8', '80.9'] for summary in ['80.8', '80.9'] for unit in ['USD million', 'USDm']]
        return [{'name': f'{value}_{unit}', 'text': text.replace('80.8', value).replace('USD million', unit),
                 'original': value == '80.8' and unit == 'USD million'}
                for value in ['80.8', '80.9'] for unit in ['USD million', 'USDm']]
    return [{'name': 'original', 'text': text, 'original': True}]


def rhs_rows(encoded, spans):
    offsets = np.asarray(encoded['offsets'])
    return np.flatnonzero(np.any([(offsets[:, 0] < b) & (offsets[:, 1] > a)
                                 for a, b in spans], axis=0))


def check_inputs(native, protocol, metadata):
    read_native(native)
    if (data.file_hash(native/'seal.json') != metadata['native_seal_sha256']
            or data.file_hash(protocol) != metadata['protocol_sha256']):
        raise ValueError('Constraint pilot dependencies changed')


def score_or_none(values, positive, negative):
    try:
        return prototype_scores(values, positive, negative)
    except ValueError as error:
        if str(error) != 'Constraint direction is unavailable':
            raise
        return None


def numeric_reading(row, value, occurrence=0, keys=('interaction', 'causal', 'raw')):
    if row is None:
        return {'tokens': [], 'means': dict.fromkeys(keys), 'available': False}
    match = list(re.finditer(r'(?<![\d.])'+re.escape(value)+r'(?![\d.])', row['response']))[occurrence]
    tokens = [t for t in row['tokens'] if t['start'] < match.end() and t['end'] > match.start()]
    return {'tokens': tokens, 'means': {k: None if any(t['scores'][k] is None for t in tokens) else
        float(np.mean([t['scores'][k] for t in tokens])) for k in keys}}


def mechanism_checks(rows):
    by_id = {(r['id'], r['variant']): r for r in rows if r.get('available', True)}
    comparisons = []
    for direction in ['decrease', 'increase']:
        wrong = numeric_reading(by_id.get(('fred:finqa:test:53', f'{direction}_8')), '8')
        right = numeric_reading(by_id.get(('fred:finqa:test:53', f'{direction}_94')), '94')
        comparisons.append({'id': f'fin53_{direction}', 'scope': 'numeric magnitude only; decrease94 remains directionally false',
                            'wrong': wrong, 'right': right})
    for unit in ['USD million', 'USDm']:
        for occurrence in range(2):
            wrong = numeric_reading(by_id.get(('fred:tatqa:test:3', f'80.9_{unit}')), '80.9', occurrence)
            right = numeric_reading(by_id.get(('fred:tatqa:test:3', f'80.8_{unit}')), '80.8', occurrence)
            comparisons.append({'id': f'tat3_{unit}_{occurrence}', 'scope': 'formula and downstream conclusion both retain the declared full history',
                                'wrong': wrong, 'right': right})
    for pair in comparisons:
        pair['margins'] = {k: None if pair['wrong']['means'][k] is None or pair['right']['means'][k] is None else
            pair['wrong']['means'][k]-pair['right']['means'][k] for k in ['interaction', 'causal', 'raw']}
    loo = {sid: next((r['loo']['interaction']['pair_accuracy'] for r in rows
                     if r['id'] == sid and r.get('available', True)), None)
           for sid in ['fred:finqa:test:53', 'fred:tatqa:test:3']}
    passed = (all(v is not None and v >= .75 for v in loo.values()) and
              all(p['margins']['interaction'] is not None and p['margins']['interaction'] > 0 for p in comparisons))
    return {'primary': 'interaction', 'primary_loo': loo, 'comparisons': comparisons,
            'mechanism_passed': passed, 'benchmark_target_met': False,
            'scope': 'post-seal declared diagnostic contrasts, not token repair of every subtoken or calibrated cross-source risk'}


def belief_checks(rows):
    by_id = {(r['id'], r['variant']): r for r in rows if r.get('available', True)}
    keys, comparisons = ('fact', 'coherence', 'binary'), []
    for direction in ['decrease', 'increase']:
        wrong = numeric_reading(by_id.get(('fred:finqa:test:53', f'{direction}_8')), '8', keys=keys)
        right = numeric_reading(by_id.get(('fred:finqa:test:53', f'{direction}_94')), '94', keys=keys)
        comparisons.append({'id': f'fin53_{direction}', 'wrong': wrong, 'right': right,
            'margin': None if wrong['means']['fact'] is None or right['means']['fact'] is None else
            wrong['means']['fact']-right['means']['fact']})
    truth_order = []
    for unit in ['USD million', 'USDm']:
        states = []
        for prior in ['80.8', '80.9']:
            for value in ['80.8', '80.9']:
                row = by_id.get(('fred:tatqa:test:3', f'p{prior}_s{value}_{unit}'))
                reading = numeric_reading(row, value, -1, keys)
                states.append({'prior': prior, 'value': value, 'risk': int(value == '80.9'),
                    'coherence': int(prior == value), 'reading': reading})
        valid = [s['reading']['means']['fact'] for s in states if s['risk'] == 0]
        invalid = [s['reading']['means']['fact'] for s in states if s['risk'] == 1]
        margin = None if any(v is None for v in valid+invalid) else min(invalid)-max(valid)
        truth_order.append({'unit': unit, 'states': states, 'worst_case_truth_margin': margin})
    loo = {sid: next((r['loo']['fact']['pair_accuracy'] for r in rows
                     if r['id'] == sid and r.get('available', True)), None)
           for sid in ['fred:finqa:test:53', 'fred:tatqa:test:3']}
    passed = (all(v is not None and v >= .75 for v in loo.values())
        and all(c['margin'] is not None and c['margin'] > 0 for c in comparisons)
        and all(c['worst_case_truth_margin'] is not None and c['worst_case_truth_margin'] > 0 for c in truth_order))
    return {'primary': 'fact', 'primary_loo': loo, 'numeric_contrasts': comparisons,
            'factorial_truth_order': truth_order, 'mechanism_passed': passed,
            'benchmark_target_met': False, 'scope': 'source-conditional numerical mechanism only; no natural benchmark claim'}


def binding_checks(rows):
    selected = {r['variant']: r for r in rows if r.get('id') == 'fred:tatqa:test:50' and r.get('probe_spans')}
    keys, comparisons = ('fact', 'coherence', 'binary'), []
    for order in [0, 1]:
        for slot in [0, 1]:
            states = []
            for prior in [0, 1]:
                for risk in [0, 1]:
                    row = selected.get(f'order{order}_p{prior}_b{risk}')
                    reading = {'means': dict.fromkeys(keys), 'tokens': []}
                    if row is not None:
                        a, b = row['probe_spans'][slot]
                        tokens = [t for t in row['tokens'] if t['start'] < b and t['end'] > a]
                        reading = {'tokens': tokens, 'means': {k: None if not tokens or
                            any(t['scores'][k] is None for t in tokens) else
                            float(np.mean([t['scores'][k] for t in tokens])) for k in keys}}
                    states.append({'prior_risk': prior, 'risk': risk, 'reading': reading})
            valid = [s['reading']['means']['fact'] for s in states if s['risk'] == 0]
            invalid = [s['reading']['means']['fact'] for s in states if s['risk'] == 1]
            margin = None if any(v is None for v in valid+invalid) else min(invalid)-max(valid)
            comparisons.append({'order': order, 'slot': slot, 'states': states, 'worst_case_truth_margin': margin})
    accuracy = next((r['loo']['fact']['pair_accuracy'] for r in rows if r['id'] == 'fred:tatqa:test:50'
                     and r.get('available', True)), None)
    return {'primary': 'fact', 'source_loo': accuracy, 'comparisons': comparisons,
        'mechanism_passed': accuracy is not None and accuracy >= .75 and
            all(c['worst_case_truth_margin'] is not None and c['worst_case_truth_margin'] > 0 for c in comparisons),
        'benchmark_target_met': False, 'scope': 'one released source, eight certified paraphrase probes; no independent dataset claim'}


def run(native, output, protocol, factorial=False, binding=False):
    import torch
    from transformers import AutoTokenizer

    from myproject.inputs import encode_response, encode_source
    from myproject.model import load_model, prefill

    factorial = factorial or binding
    plan, samples, original = read_native(native)
    selected = [(g, s, r) for (g, s), r in zip(samples, original, strict=True) if s.id in CASES]
    if {s.id for _, s, _ in selected} != set(CASES):
        raise ValueError('Require all six declared original mechanism cases')
    output.mkdir(parents=True, exist_ok=False)
    observation_root = output/'observations'; observation_root.mkdir()
    identity = research_identity()
    hashes = {p: data.file_hash(Path(p)) for p in plan['model_files_sha256']}
    if hashes != plan['model_files_sha256']:
        raise ValueError('Frozen model changed')
    metadata = {'native': str(native.resolve()), 'native_seal_sha256': data.file_hash(native/'seal.json'),
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'source': identity, 'model_files_sha256': hashes, 'annotation_reads': 0,
        'mode': 'binding' if binding else 'belief' if factorial else 'interaction',
        'scope': 'six selected failures plus declared diagnostic edits; no benchmark AUROC/AP claim'}
    data.write_json(output/'plan.json', metadata)
    prepared = []
    for group, sample, record in selected:
        bank = binding_pairs(sample.source) if binding else None
        queries = binding_probes(sample, bank) if binding else variants(sample, factorial)
        exclusion = ' '.join(q['text'] for q in queries)
        bank = bank if binding else witnesses(sample.source, exclusion)
        prepared.append({'id': sample.id, 'group': group, 'queries': queries, 'witnesses': bank,
                         'excluded_values': [] if binding else sorted(str(v) for v in numbers(exclusion)),
                         'available': len(bank) >= 3})
    data.write_json(output/'inputs.json', {'cases': prepared, 'witness_supervision':
        'source row/year/value contradictions' if binding else
        'Decimal-certified mathematical identities, not human hallucination labels or financial relevance'})
    torch.manual_seed(42)
    model = load_model(Path(plan['model']), 'cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], use_fast=True, local_files_only=True)
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    records, start = [], time.monotonic()
    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        for case, (_, sample, original_record) in zip(prepared, selected, strict=True):
            if not case['available']:
                records.append({'id': sample.id, 'available': False}); continue
            prompt = encode_source(tokenizer, sample.prompt, sample.source)['prompt_with_source']
            with torch.inference_mode():
                cache = prefill(model, prompt)
            modes = ('causal',) if factorial else ('bidirectional', 'independent', 'causal')
            bank_states = {k: [] for k in modes}
            bank_pair, bank_sign, bank_coherence = [], [], []
            for index, pair in enumerate(case['witnesses']):
                views = (binding_worlds(pair) if binding else belief_worlds(pair) if factorial else
                         [pair[side] | {'risk': sign} for sign, side in enumerate(['positive', 'negative'])])
                encoded_pair = [encode_response(tokenizer, view['text']) for view in views]
                if any(Counter(e['answer_ids']) != Counter(encoded_pair[0]['answer_ids']) for e in encoded_pair[1:]):
                    raise ValueError('Certificate pair tokenizer multisets differ')
                for view, encoded in zip(views, encoded_pair, strict=True):
                    ids = encoded['answer_ids']
                    rows = rhs_rows(encoded, view['rhs_spans'])
                    units = encoded['units'] if factorial else [{'start': 0, 'stop': len(ids)}]
                    states = interaction_states(model, prompt, ids, units, cache, modes)
                    for key in states:
                        bank_states[key].append(states[key][rows])
                    bank_pair.extend([index]*len(rows)); bank_sign.extend([view['risk']]*len(rows))
                    if factorial:
                        bank_coherence.extend([view['coherence']]*len(rows))
            file = observation_root/f'{len(records)}_anchors.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, **{k: np.concatenate(v) for k, v in bank_states.items()},
                                    pair_index=bank_pair, sign=bank_sign,
                                    **({'coherence': bank_coherence} if factorial else {}))
            entry = {'id': sample.id, 'available': True, 'anchors': str(file.relative_to(output)),
                     'anchors_sha256': data.file_hash(file), 'queries': []}
            for index, query in enumerate(case['queries']):
                encoded = encode_response(tokenizer, query['text'])
                ids, offsets = np.asarray(encoded['answer_ids']), np.asarray(encoded['offsets'])
                target = np.flatnonzero((offsets[:, 1] > offsets[:, 0]) & ~np.isin(ids, encoded['special_ids']))
                unit_index = np.repeat(np.arange(len(encoded['units'])),
                                       [u['stop']-u['start'] for u in encoded['units']])[target]
                if query['original']:
                    with np.load(native/original_record['file']) as raw:
                        if (not np.array_equal(ids, raw['full_token_id'])
                                or not np.array_equal(target, raw['target'])
                                or not np.array_equal(unit_index, raw['unit_index'])
                                or not np.array_equal(offsets, raw['offsets'])):
                            raise ValueError('Original response axis changed')
                states = interaction_states(model, prompt, ids.tolist(), encoded['units'], cache, modes)
                file = observation_root/f'{len(records)}_query{index}.npz'
                with file.open('xb') as stream:
                    np.savez_compressed(stream, **{k: v[target] for k, v in states.items()},
                                        target=target, token_id=ids[target], offsets=offsets[target], unit_index=unit_index)
                entry['queries'].append({'name': query['name'], 'original': query['original'],
                                        'file': str(file.relative_to(output)), 'sha256': data.file_hash(file)})
            records.append(entry)
            print('COLLECTED', len(records), len(selected), sample.id, len(case['witnesses']), flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    check_inputs(native, protocol, metadata)
    if any(data.file_hash(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Frozen model changed during observation')
    data.write_json(output/'collection.json', {'records': records, 'source': completed_identity(identity),
        'counts': counts, 'seconds': time.monotonic()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {n: data.file_hash(output/n) for n in ['plan.json', 'inputs.json', 'collection.json']})
    score(output)
    evaluate(output)


def score(root):
    verify_seal(root)
    identity = research_identity()
    if identity != data.read_json(root/'plan.json')['source']:
        raise ValueError('Pilot source changed before scoring')
    destination = root/'scores'; destination.mkdir(exist_ok=False)
    records = data.read_json(root/'collection.json')['records']
    factorial = data.read_json(root/'plan.json').get('mode') in {'belief', 'binding'}
    scored = []
    for entry in records:
        if not entry['available']:
            scored.append(entry); continue
        if data.file_hash(root/entry['anchors']) != entry['anchors_sha256']:
            raise ValueError('Witness states changed')
        with np.load(root/entry['anchors']) as raw:
            count = int(raw['pair_index'].max())+1
            if factorial:
                features = normalized(raw['causal'])
                model = belief_model(features, raw['pair_index'], raw['sign'], raw['coherence'])
                loo = belief_loo(features, raw['pair_index'], raw['sign'], raw['coherence'])
            else:
                features = representations(raw)
                banks = {key: [np.array([value[(raw['pair_index'] == i) & (raw['sign'] == sign)].mean(0)
                                         for i in range(count)]) for sign in [0, 1]]
                         for key, value in features.items()}
        if factorial:
            available = dict.fromkeys(['fact', 'coherence', 'binary'], True)
        else:
            loo = {}
            for key, (positive, negative) in banks.items():
                margins = []
                for i in range(count):
                    keep = np.arange(count) != i
                    values = score_or_none(np.stack([positive[i], negative[i]]), positive[keep], negative[keep])
                    margins.append(float(values[1]-values[0]) if values is not None else None)
                loo[key] = {'margins': margins, 'available_pairs': sum(v is not None for v in margins),
                            'pair_accuracy': None if any(v is None for v in margins) else float(np.mean(np.sign(margins)/2+.5))}
            available = {k: score_or_none(p[:1], p, n) is not None for k, (p, n) in banks.items()}
        output = {'id': entry['id'], 'available': True, 'readouts': available, 'loo': loo, 'queries': []}
        for query in entry['queries']:
            if data.file_hash(root/query['file']) != query['sha256']:
                raise ValueError('Query states changed')
            with np.load(root/query['file']) as raw:
                if factorial:
                    values = belief_scores(normalized(raw['causal']), model)
                else:
                    features = representations(raw)
                    values = {key: prototype_scores(value, *banks[key]) for key, value in features.items() if available[key]}
                file = destination/f'{len(scored)}_{len(output["queries"])}.npz'
                with file.open('xb') as stream:
                    np.savez_compressed(stream, **values, **{k: raw[k] for k in ['target', 'token_id', 'offsets', 'unit_index']})
            output['queries'].append(query | {'file': str(file.relative_to(root)), 'sha256': data.file_hash(file)})
        scored.append(output)
    data.write_json(destination/'complete.json', {'records': scored, 'source': completed_identity(identity), 'annotation_reads': 0})
    data.write_json(destination/'seal.json', {'complete.json': data.file_hash(destination/'complete.json')})


def evaluate(root):
    verify_seal(root); verify_seal(root/'scores')
    plan = data.read_json(root/'plan.json')
    native = Path(plan['native']); check_inputs(native, Path(plan['protocol']), plan)
    original_plan, _, _ = read_native(native)
    annotations = {}
    for group, spec in enumerate(original_plan['groups']):
        if spec['role'] == 'test':
            prepared = Path(spec['path']); manifest = data.read_json(prepared/'manifest.json')
            if data.file_hash(prepared/'annotations.jsonl') != manifest['annotations_sha256']:
                raise ValueError('Original annotations changed')
            annotations.update({a['id']: a for a in read_jsonl(prepared/'annotations.jsonl') if a['id'] in CASES})
    inputs = {c['id']: c for c in data.read_json(root/'inputs.json')['cases']}
    rows = []
    for entry in data.read_json(root/'scores/complete.json')['records']:
        if not entry['available']:
            rows.append(entry); continue
        queries = {q['name']: q for q in inputs[entry['id']]['queries']}
        for query in entry['queries']:
            if data.file_hash(root/query['file']) != query['sha256']:
                raise ValueError('Sealed scores changed')
            text = queries[query['name']]['text']
            if query['original'] and annotations[entry['id']]['response_sha256'] != data.text_hash(text):
                raise ValueError('Original annotation/response binding changed')
            with np.load(root/query['file']) as raw:
                y = (token_labels({'labels': annotations[entry['id']]['spans']}, {'offsets': raw['offsets']})
                     if query['original'] else None)
                tokens = [{'text': text[a:b], 'start': int(a), 'end': int(b),
                           'label': int(y[i]) if y is not None else None, 'unit_index': int(raw['unit_index'][i]),
                           'scores': {k: float(raw[k][i]) if k in raw else None for k in entry['readouts']}}
                          for i, (a, b) in enumerate(raw['offsets'])]
            rows.append({'id': entry['id'], 'variant': query['name'], 'original': query['original'], 'response': text,
                         'probe_spans': queries[query['name']].get('probe_spans'),
                         'readouts': entry['readouts'], 'loo': entry['loo'], 'tokens': tokens})
    check_inputs(native, Path(plan['protocol']), plan)
    verify_seal(root); verify_seal(root/'scores')
    checks = (binding_checks(rows) if plan.get('mode') == 'binding' else
              belief_checks(rows) if plan.get('mode') == 'belief' else mechanism_checks(rows))
    data.write_json(root/'evaluation.json', {'cases': rows, 'checks': checks,
        'scope': 'post-seal mechanism inspection only; diagnostic edits have no benchmark labels',
        'source': completed_identity(plan['source'])})
