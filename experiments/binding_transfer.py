"""Full-population financial evaluation of relation-conditioned shared inference."""

from pathlib import Path
from collections import Counter
import sys
import time

import numpy as np

from experiments import data
from experiments.datasets import _check_spans, read_jsonl
from experiments.evaluate import paired_source_differences, ranking, token_labels
from experiments.fit_ablation import verify_seal
from experiments.native_dataset import FIELDS, read_native
from experiments.provenance import completed_identity, research_identity
from experiments.research.binding_dependency import dependency_readout, source_cells
from experiments.research.relational_shared import grouped_relational_scores, relation_factors
from myproject.measure import unit_ranges
from experiments.research.readouts import SharedDetector


SYSTEMS = ('full_gap', 'r4_frozen', 'r4_adapted', 'shared', 'relational_shared')


def run(native, scores, output, protocol):
    import torch
    from transformers import AutoTokenizer

    from experiments.research.attention import attention_profiles
    from myproject.inputs import encode_response, encode_source
    from myproject.model import load_model

    plan, samples, originals = read_native(native)
    verify_seal(scores)
    old_plan = data.read_json(scores/'plan.json')
    if old_plan['native_seal_sha256'] != data.file_hash(native/'seal.json'):
        raise ValueError('Historical scores must use exactly the original native observations')
    if not {'frozen_score', 'adapted_score'} <= set(old_plan['systems']):
        raise ValueError('Historical frozen/adapted R4 systems required')
    old_rows = data.read_json(scores/'complete.json')['records']
    old_records = {r['id']: r for r in old_rows}
    selected = [(g, s, r) for (g, s), r in zip(samples, originals, strict=True)
                if plan['groups'][g]['role'] == 'test']
    if (len(old_rows) != len(old_records) or len(old_records) != len(selected) or
            set(old_records) != {s.id for _, s, _ in selected} or
            any(old_records[s.id]['group'] != g for g, s, _ in selected)):
        raise ValueError('All original released test responses are required')
    population = Counter(Path(plan['groups'][g]['path']).name for g, _, _ in selected)
    if population != {'test_finqa': 162, 'test_tatqa': 130}:
        raise ValueError('This frozen financial protocol requires FinQA162 and TATQA130')
    output.mkdir(parents=True, exist_ok=False)
    identity = research_identity()
    hashes = {p: data.file_hash(Path(p)) for p in plan['model_files_sha256']}
    if hashes != plan['model_files_sha256']:
        raise ValueError('Frozen model changed')
    metadata = {'source': identity, 'observation_source': plan['source'],
        'native': str(native.resolve()), 'native_seal_sha256': data.file_hash(native/'seal.json'),
        'historical_scores': str(scores.resolve()), 'score_seal_sha256': data.file_hash(scores/'seal.json'),
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'model_files_sha256': hashes, 'command': sys.argv, 'annotation_reads': 0,
        'systems': list(SYSTEMS), 'scope': 'complete released FRED silver test; exposed development data',
        'max_clauses': 12, 'prior': [1., 1.], 'unknown_policy': 'neutral; all tokens retained'}
    data.write_json(output/'plan.json', metadata)
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True, use_fast=True)
    encoded, preflight = {}, []
    for group, sample, original in selected:
        response = encode_response(tokenizer, sample.response)
        offsets = np.asarray(response['offsets'])
        ids = np.asarray(response['answer_ids'])
        target = np.flatnonzero((offsets[:, 1] > offsets[:, 0]) & ~np.isin(ids, response['special_ids']))
        unit = np.repeat(np.arange(len(response['units'])),
                         [u['stop']-u['start'] for u in response['units']])[target]
        with np.load(native/original['file']) as raw:
            if (not np.array_equal(ids, raw['full_token_id']) or not np.array_equal(offsets, raw['offsets'])
                    or not np.array_equal(target, raw['target']) or not np.array_equal(unit, raw['unit_index'])):
                raise ValueError('Original token or fixed span identity changed')
            old = old_records[sample.id]
            if data.file_hash(scores/old['file']) != old['sha256']:
                raise ValueError('Historical score bytes changed before collection')
            with np.load(scores/old['file']) as previous:
                for axis in ['target', 'token_id', 'answer_index', 'unit_index', 'offsets']:
                    if not np.array_equal(raw[axis], previous[axis]):
                        raise ValueError('Historical score axes changed before collection')
        # Uniform positive history bounds all eligible clauses before GPU execution.
        _, edges = dependency_readout(sample.response, offsets, response['units'],
                                      np.ones((len(ids), len(ids))), sample.source)
        factors, exclusions = relation_factors(sample.response, edges, target, unit)
        starts, stops = unit_ranges(np.zeros(len(target), int), unit)
        maximum = max((sum(len(f) for f in factors if all(a <= t < b for c in f for t in c['tokens']))
                       for a, b in zip(starts, stops, strict=True)), default=0)
        if maximum > metadata['max_clauses']:
            raise ValueError(f'Exact DP cap exceeded before collection: {sample.id} {maximum}')
        encoded[sample.id] = response
        preflight.append({'id': sample.id, 'group': group, 'tokens': len(target),
            'source_cells': len(source_cells(sample.source)), 'potential_factors': len(factors),
            'max_span_clauses': maximum, 'exclusions': exclusions})
    if sum(r['tokens'] for r in preflight) != 23794:
        raise ValueError('Frozen financial token population changed')
    data.write_json(output/'preflight.json', {'records': preflight, 'annotation_reads': 0})
    torch.manual_seed(42)
    model = load_model(Path(plan['model']), 'cuda:0')
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    destination = output/'predictions'; destination.mkdir()
    attention_root = output/'attention'; attention_root.mkdir()
    start, records = time.monotonic(), []
    torch.cuda.reset_peak_memory_stats()
    try:
        for i, (group, sample, original) in enumerate(selected):
            response = encoded[sample.id]
            ids, offsets = np.asarray(response['answer_ids']), np.asarray(response['offsets'])
            attention = None
            if preflight[i]['source_cells']:
                prompt = encode_source(tokenizer, sample.prompt, sample.source)['prompt_with_source']
                observed = attention_profiles(model, prompt, response['answer_ids'])
                p, n = observed['prompt_tokens'], len(ids)
                history = (observed['fit'][:, p:p+n]+observed['hold'][:, p:p+n])/2
                attention = attention_root/f'{i}.npz'
                with attention.open('xb') as stream:
                    np.savez_compressed(stream, history=history, token_id=ids, offsets=offsets)
            else:
                history = np.zeros((len(ids), len(ids)))
            year_values, edges = dependency_readout(sample.response, offsets, response['units'], history, sample.source)
            old = old_records[sample.id]
            if data.file_hash(scores/old['file']) != old['sha256']:
                raise ValueError('Historical score bytes changed')
            with np.load(native/original['file']) as raw, np.load(scores/old['file']) as previous:
                for axis in ['target', 'token_id', 'answer_index', 'unit_index', 'offsets']:
                    if not np.array_equal(raw[axis], previous[axis]):
                        raise ValueError('Historical score axes changed')
                pack = {k: raw[k] for k in FIELDS}
                factors, exclusions = relation_factors(sample.response, edges, raw['target'], raw['unit_index'])
                shared = SharedDetector(plan['task']).score(pack)
                combined = grouped_relational_scores(raw['delta'][:, 1], raw['answer_index'], raw['unit_index'], factors)
                file = destination/f'{i}.npz'
                with file.open('xb') as stream:
                    np.savez_compressed(stream, full_gap=raw['delta'][:, 1], shared=shared,
                        r4_frozen=previous['frozen_score'], r4_adapted=previous['adapted_score'],
                        relational_shared=combined, year_only=year_values['dependency'][raw['target']],
                        **{a: raw[a] for a in ['target', 'token_id', 'answer_index', 'unit_index', 'offsets']})
            records.append({'id': sample.id, 'group': group, 'file': str(file.relative_to(output)),
                'sha256': data.file_hash(file), 'attention_file': str(attention.relative_to(output)) if attention else None,
                'attention_sha256': data.file_hash(attention) if attention else None,
                'edges': edges, 'factors': factors, 'exclusions': exclusions,
                'tokens_changed': int(np.count_nonzero(combined != shared))})
            print('SCORED', i+1, len(selected), sample.id, 'factors', len(factors), flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    read_native(native); verify_seal(scores)
    if (data.file_hash(native/'seal.json') != metadata['native_seal_sha256'] or
            data.file_hash(scores/'seal.json') != metadata['score_seal_sha256'] or
            data.file_hash(protocol) != metadata['protocol_sha256'] or
            any(data.file_hash(Path(p)) != h for p, h in hashes.items())):
        raise ValueError('A frozen experiment dependency changed')
    data.write_json(output/'complete.json', {'records': records, 'source': completed_identity(identity),
        'annotation_reads': 0, 'counts': counts, 'seconds': time.monotonic()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
    data.write_json(output/'seal.json', {f: data.file_hash(output/f)
        for f in ['plan.json', 'preflight.json', 'complete.json']})
    evaluate(output)


def evaluate(root):
    """All labels enter only after complete, sealed full-population predictions."""
    from sklearn.metrics import auc, precision_recall_curve

    verify_seal(root)
    metadata = data.read_json(root/'plan.json')
    native = Path(metadata['native'])
    plan, samples, _ = read_native(native)
    records = data.read_json(root/'complete.json')['records']
    selected = {s.id: (g, s) for g, s in samples if plan['groups'][g]['role'] == 'test'}
    if len(records) != len(selected) or {r['id'] for r in records} != set(selected):
        raise ValueError('Full population score join required')
    results, examples = {}, []
    for group, spec in enumerate(plan['groups']):
        if spec['role'] != 'test':
            continue
        prepared = Path(spec['path'])
        manifest = data.read_json(prepared/'manifest.json')
        if data.file_hash(prepared/'annotations.jsonl') != manifest['annotations_sha256']:
            raise ValueError('Annotation bytes changed')
        rows = list(read_jsonl(prepared/'annotations.jsonl'))
        annotations = {r['id']: r for r in rows}
        if len(rows) != len(annotations) or set(annotations) != {s.id for g, s in samples if g == group}:
            raise ValueError('Exact annotation population required')
        labels, source_ids = [], []
        arrays = {key: [] for key in SYSTEMS+('year_only',)}
        coverage = {'responses': 0, 'table_responses': 0, 'relation_responses': 0,
                    'factor_responses': 0, 'factors': 0, 'tokens_changed': 0}
        for record in records:
            if record['group'] != group:
                continue
            _, sample = selected[record['id']]
            gold = annotations[sample.id]
            if (gold['granularity'] != 'token' or gold['coverage'] != 'complete' or
                    gold['response_sha256'] != data.text_hash(sample.response)):
                raise ValueError('Original complete token annotations required')
            _check_spans(sample.response, gold['spans'])
            if any(s['label'] != 1 for s in gold['spans']):
                raise ValueError('Expected positive spans with complete coverage')
            if data.file_hash(root/record['file']) != record['sha256']:
                raise ValueError('Sealed scores changed')
            if record['attention_file'] and data.file_hash(root/record['attention_file']) != record['attention_sha256']:
                raise ValueError('Physical attention changed')
            with np.load(root/record['file']) as raw:
                y = token_labels({'labels': gold['spans']}, {'offsets': raw['offsets']})[raw['target']]
                labels.append(y)
                source_ids.extend([sample.source_id]*len(y))
                for key in arrays:
                    arrays[key].append(raw[key])
                changed = np.flatnonzero(raw['shared'] != raw['relational_shared'])
                if len(changed):
                    examples.append({'id': sample.id, 'text': sample.response, 'gold_spans': gold['spans'],
                        'factors': record['factors'], 'edges': record['edges'],
                        'changed': [{'token': int(t), 'text': sample.response[a:b], 'label': bool(y[t]),
                                     'shared': float(raw['shared'][t]), 'relational': float(raw['relational_shared'][t])}
                            for t in changed for a, b in [raw['offsets'][raw['target'][t]]]]})
            coverage['responses'] += 1
            coverage['table_responses'] += record['attention_file'] is not None
            coverage['relation_responses'] += bool(record['edges'])
            coverage['factor_responses'] += bool(record['factors'])
            coverage['factors'] += len(record['factors'])
            coverage['tokens_changed'] += record['tokens_changed']
        labels = np.concatenate(labels)
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        metrics = {}
        for key in SYSTEMS:
            precision, recall, _ = precision_recall_curve(labels, arrays[key])
            metrics[key] = ranking(labels, arrays[key]) | {'aupr_trapezoid': float(auc(recall, precision))}
        covered = np.isfinite(arrays['year_only'])
        coverage.update(year_tokens=int(covered.sum()), year_positive_tokens=int(labels[covered].sum()))
        results[prepared.name] = {'tokens': len(labels), 'positives': int(labels.sum()),
            'coverage': coverage, 'metrics': metrics,
            'year_only_conditional': ranking(labels[covered], arrays['year_only'][covered]) if covered.any() else None,
            'paired_difference': paired_source_differences(labels, arrays['shared'], arrays['relational_shared'], np.asarray(source_ids)),
            'annotation_sha256': manifest['annotations_sha256'],
            'label_provenance': sorted({r.get('label_provenance', '') for r in rows})}
        print('RESULT', prepared.name, {k: (v['auroc'], v['ap']) for k, v in metrics.items()}, coverage, flush=True)
    data.write_json(root/'evaluation.json', {'groups': results, 'score_seal_sha256': data.file_hash(root/'seal.json'),
        'source': completed_identity(metadata['source']), 'benchmark_target_met': all(
            g['metrics']['relational_shared']['auroc'] >= .8 and
            g['metrics']['relational_shared']['ap'] > g['metrics']['shared']['ap'] for g in results.values()),
        'promoted': False, 'scope': metadata['scope']})
    data.write_json(root/'changed_examples.json', {'records': examples})
