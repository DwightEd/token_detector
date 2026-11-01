"""Collect matched frozen-model likelihoods for source-only relational controls."""

from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.fit_ablation import verify_seal
from experiments.native_dataset import read_native
from experiments.provenance import completed_identity, research_identity
from experiments.research.relational import relational_sources


def read_worlds(root, native):
    verify_seal(root)
    plan = data.read_json(root/'plan.json')
    if (plan['source'] != research_identity()
            or plan['native_seal_sha256'] != data.file_hash(native/'seal.json')
            or data.file_hash(Path(plan['protocol'])) != plan['protocol_sha256']):
        raise ValueError('Relational observation dependencies changed')
    records = data.read_json(root/'collection.json')['records']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate relational observations')
    if any(data.file_hash(root/r['file']) != r['sha256'] for r in records):
        raise ValueError('Relational observations changed')
    return records


def collect(native, output, protocol, ids=None):
    import torch
    from transformers import AutoTokenizer

    from myproject.inputs import encode_response, encode_source
    from myproject.model import load_model, prefill, score_continuation

    plan, samples, observed = read_native(native)
    selected = [(g, s, r) for (g, s), r in zip(samples, observed, strict=True)
                if plan['groups'][g]['role'] == 'test' and (ids is None or s.id in ids)]
    if not selected or (ids is not None and set(ids) != {s.id for _, s, _ in selected}):
        raise ValueError('Require nonempty exact test sample identities')
    output.mkdir(parents=True, exist_ok=False)
    destination = output/'observations'; destination.mkdir()
    identity = research_identity()
    model_path = Path(plan['model'])
    hashes = {p: data.file_hash(Path(p)) for p in plan['model_files_sha256']}
    if hashes != plan['model_files_sha256']:
        raise ValueError('Parent frozen model changed')
    native_hash, protocol_hash = data.file_hash(native/'seal.json'), data.file_hash(protocol)
    data.write_json(output/'plan.json', {'native': str(native.resolve()),
        'native_seal_sha256': native_hash, 'source': identity,
        'protocol': str(protocol.resolve()), 'protocol_sha256': protocol_hash,
        'model_files_sha256': hashes, 'ids': ids, 'annotation_reads': 0,
        'backend': 'matched SDPA source/empty/relational; historical eager is a separate comparator'})
    torch.manual_seed(42)
    model = load_model(model_path, 'cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    records, start = [], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for group, sample, record in selected:
                response = encode_response(tokenizer, sample.response)
                with np.load(native/record['file']) as raw:
                    if (not np.array_equal(response['answer_ids'], raw['full_token_id'])
                            or not np.array_equal(response['offsets'], raw['offsets'])):
                        raise ValueError('Response tokenization changed')
                    axis = {k: raw[k] for k in ['target', 'token_id', 'answer_index', 'unit_index']}
                original = encode_source(tokenizer, sample.prompt, sample.source)
                worlds = [{'name': 'empty', 'prompt': original['prompt_without_source']}]
                for world in relational_sources(sample.source):
                    prompt = sample.prompt.replace(sample.source, world['source'])
                    worlds.append({'name': world['name'], 'prompt':
                        encode_source(tokenizer, prompt, world['source'])['prompt_with_source']})
                likelihoods = []
                for prompt in [original['prompt_with_source'], *(w['prompt'] for w in worlds)]:
                    cache = prefill(model, prompt[:-1])
                    likelihoods.append(score_continuation(model, cache, prompt[-1], response['answer_ids']))
                    del cache
                logp = np.array(likelihoods, dtype=float)[:, axis['target']].T
                if not np.isfinite(logp).all():
                    raise ValueError('Nonfinite relational likelihood')
                file = data.prediction_path(destination, sample.id).with_suffix('.npz')
                with file.open('xb') as stream:
                    np.savez_compressed(stream, logp=logp, **axis)
                records.append({'id': sample.id, 'group': group, 'file': str(file.relative_to(output)),
                    'sha256': data.file_hash(file), 'worlds': [w['name'] for w in worlds],
                    'prompt_sha256': [data.text_hash(str(original['prompt_with_source'])),
                                      *(data.text_hash(str(w['prompt'])) for w in worlds)]})
                print('COLLECTED', len(records), len(selected), sample.id, len(worlds), flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    read_native(native)
    if data.file_hash(native/'seal.json') != native_hash or data.file_hash(protocol) != protocol_hash:
        raise ValueError('Native source or relational protocol changed during collection')
    if any(data.file_hash(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Frozen model changed during collection')
    data.write_json(output/'collection.json', {'records': records, 'source': completed_identity(identity),
        'counts': counts, 'seconds': time.monotonic()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {n: data.file_hash(output/n) for n in ['plan.json', 'collection.json']})
