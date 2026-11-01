"""Source-only temporal interventions through the public response-replay API."""

from pathlib import Path
import re
import time

import numpy as np

from experiments import data
from experiments.native_dataset import read_native
from experiments.provenance import completed_identity, research_identity
from myproject.counterfactual import repair_support

YEAR = re.compile(r'\b(?:19|20)\d{2}\b')


def year_repairs(source, response):
    """All other literal source years, without certificate or label selection."""
    years = sorted(set(YEAR.findall(source)))
    variants, blocks = [response], []
    for match in YEAR.finditer(response):
        if match[0] not in years:
            continue
        candidates = [year for year in years if year != match[0]]
        indices = list(range(len(variants), len(variants)+len(candidates)))
        variants.extend(response[:match.start()]+year+response[match.end():] for year in candidates)
        blocks.append({'start': match.start(), 'end': match.end(), 'original': match[0],
                       'candidates': candidates, 'variant_indices': indices})
    return variants, blocks


def future_region(encoded, block):
    """Unchanged raw future positions through the original prechosen span end."""
    offsets = np.asarray(encoded['offsets'])
    condition = np.flatnonzero((offsets[:, 0] < block['end']) & (offsets[:, 1] > block['start']))
    if not len(condition) or offsets[condition[-1], 1] != block['end']:
        return condition, np.empty(0, int), 'condition_not_on_token_boundary'
    unit = next((u for u in encoded['units']
                 if u['start'] <= condition[0] and condition[-1] < u['stop']), None)
    if unit is None:
        return condition, np.empty(0, int), 'condition_crosses_span'
    future = np.arange(condition[-1]+1, unit['stop'])
    return condition, future, None if len(future) else 'empty_future'


def run(native, output, protocol, ids):
    import torch
    from myproject.api import HallucinationDetector

    plan, samples, originals = read_native(native)
    selected = [(s, r) for (_, s), r in zip(samples, originals, strict=True) if s.id in ids]
    if len(set(ids)) != len(ids) or {s.id for s, _ in selected} != set(ids):
        raise ValueError('Require exact unique existing response IDs')
    output.mkdir(parents=True, exist_ok=False)
    destination = output/'observations'; destination.mkdir()
    identity = research_identity()
    model = Path(plan['model'])
    hashes = {str(p.resolve()): data.file_hash(p) for p in model.iterdir()
              if p.is_file() and (p.suffix in {'.json', '.bin', '.safetensors'} or p.name == 'tokenizer.model')}
    if any(hashes.get(p) != h for p, h in plan['model_files_sha256'].items()):
        raise ValueError('Native parent model files changed')
    inventory = {s.id: year_repairs(s.source, s.response) for s, _ in selected}
    protocol_hash = data.file_hash(protocol)
    data.write_json(output/'plan.json', {'native': str(native.resolve()),
        'native_seal_sha256': data.file_hash(native/'seal.json'),
        'protocol': str(protocol.resolve()), 'protocol_sha256': protocol_hash,
        'model': str(model), 'model_files_sha256': hashes, 'source': identity,
        'ids': ids, 'annotation_reads': 0, 'backend': 'matched SDPA response replay in both source conditions',
        'inventory': {key: {'blocks': blocks, 'response_variants': len(variants)}
                      for key, (variants, blocks) in inventory.items()},
        'scope': 'exposed diagnostic cases; temporal conditional compatibility, not a token detector benchmark'})
    torch.manual_seed(42)
    detector = HallucinationDetector.from_pretrained(model, task=plan['task'], method='witness')
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = detector.model.model.register_forward_pre_hook(count, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    start, records = time.monotonic(), []
    try:
        for sample, original in selected:
            variants, blocks = inventory[sample.id]
            likelihoods, encodings = detector.measure_responses(sample.prompt, variants, sample.source)
            if any(not np.isfinite(values).all() for values in likelihoods):
                raise ValueError('Nonfinite response-replay observation')
            encoded = encodings[0]
            with np.load(native/original['file']) as raw:
                unit_index = np.empty(len(encoded['answer_ids']), dtype=int)
                for index, unit in enumerate(encoded['units']):
                    unit_index[unit['start']:unit['stop']] = index
                if (not np.array_equal(encoded['answer_ids'], raw['full_token_id'])
                        or not np.array_equal(encoded['offsets'], raw['offsets'])
                        or not np.array_equal(unit_index[raw['target']], raw['unit_index'])):
                    raise ValueError('Original response axis changed')
                arrays = {key: raw[key] for key in ['target', 'token_id', 'unit_index', 'answer_index', 'offsets']}
            arrays.update({f'logp_{i}': values for i, values in enumerate(likelihoods)})
            arrays.update({f'ids_{i}': np.asarray(row['answer_ids']) for i, row in enumerate(encodings)})
            results = []
            for b, block in enumerate(blocks):
                condition, future, reason = future_region(encoded, block)
                row = {**block, 'condition_targets': condition.tolist(), 'future_targets': future.tolist()}
                if not block['candidates'] or reason:
                    results.append({**row, 'status': reason or 'no_source_alternative'}); continue
                aligned = all(encodings[i]['offsets'] == encoded['offsets']
                    and np.array_equal(np.asarray(encodings[i]['answer_ids'])[future],
                                       np.asarray(encoded['answer_ids'])[future])
                    for i in block['variant_indices'])
                if not aligned:
                    results.append({**row, 'status': 'future_axis_mismatch'}); continue
                result = repair_support(likelihoods[0][future],
                    np.stack([likelihoods[i][future] for i in block['variant_indices']]))
                arrays[f'increments_{b}'] = result['increments']
                results.append({**row, 'status': 'measured', 'score': result['score'],
                    'candidate_gains': result['candidate_gains'].tolist(),
                    'absent_weights': result['absent_weights'].tolist()})
            file = data.prediction_path(destination, sample.id).with_suffix('.npz')
            with file.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            records.append({'id': sample.id, 'file': str(file.relative_to(output)),
                'sha256': data.file_hash(file), 'response_sha256': data.text_hash(sample.response),
                'variant_sha256': [data.text_hash(text) for text in variants], 'blocks': results})
            print('REPLAYED', len(records), len(selected), sample.id, len(variants), flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    read_native(native)
    if data.file_hash(protocol) != protocol_hash or any(data.file_hash(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Frozen protocol or model changed')
    data.write_json(output/'complete.json', {'records': records, 'counts': counts,
        'seconds': time.monotonic()-start, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'source': completed_identity(identity), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in ['plan.json', 'complete.json']})
