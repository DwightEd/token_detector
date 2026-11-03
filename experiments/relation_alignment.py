"""Align sealed source occurrences to qualified original token axes; no scores."""

import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from myproject.inputs import encode_response, encode_source
from .data import file_hash, read_json, text_hash, verify_seal, write_json
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .research.relation_inventory import CONTRACT, align_source_spans


TOKENIZER_METADATA = ('config.json', 'tokenizer_config.json', 'tokenizer.json',
                      'special_tokens_map.json', 'model.safetensors.index.json')
AXIS_FIELDS = ('token_ids', 'answer_ids', 'source_mask', 'query_positions')


def _contained(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('Alignment input escapes its declared root')
    return candidate


def _original_axes(path):
    """Decode only original integer/Boolean axes, never Q/K/V or hidden states."""
    with np.load(path, allow_pickle=False) as archive:
        axes = {name: archive[name] for name in AXIS_FIELDS}
    if any(axes[name].ndim != 1 for name in AXIS_FIELDS):
        raise ValueError('Captured alignment axes must be one-dimensional')
    for name in ('token_ids', 'answer_ids', 'query_positions'):
        if axes[name].dtype != np.int64:
            raise ValueError('Captured integer token/position axis changed')
    if axes['source_mask'].dtype != np.bool_:
        raise ValueError('Captured source mask must remain Boolean')
    return {name: value.tolist() for name, value in axes.items()}


def _exact_offsets(coordinates, characters, shift=0):
    result = []
    for coordinate in coordinates:
        if (len(coordinate) != 2 or any(not isinstance(value, Integral) or isinstance(value, bool)
                                        for value in coordinate)
                or not 0 <= coordinate[0] <= coordinate[1] <= characters):
            raise ValueError('Tokenizer character offsets must be exact non-Boolean integers')
        result.append([shift + int(value) for value in coordinate])
    return result


def reconstruct_alignment(sample, tokenizer, axes, inventory):
    """Use mainline encoders and exact separate prompt/response tokenization.

    Supplemental offsets are qualified against both the mainline encoder and the
    original captured IDs/mask. No text normalization or joint retokenization is
    used to repair a mismatch. The tokenizer performs no model inference.
    """
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Exact original character offsets require a Fast tokenizer')
    source_pack = encode_source(tokenizer, sample.prompt, sample.source)
    response_pack = encode_response(tokenizer, sample.response)
    chat = tokenizer.apply_chat_template([dict(role='user', content=sample.prompt)],
                                         tokenize=False, add_generation_prompt=True)
    prompt = sample.prompt if sample.prompt in chat else sample.prompt.strip()
    if not prompt or chat.count(prompt) != 1 or prompt.count(sample.source) != 1:
        raise ValueError('Rendered chat has ambiguous or altered original source/prompt')
    source_start = chat.index(prompt) + prompt.index(sample.source)
    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids, answer_ids = source_pack['prompt_with_source'], response_pack['answer_ids']
    if encoded['input_ids'] != prompt_ids:
        raise ValueError('Supplemental chat offsets differ from mainline prompt token IDs')
    if (not isinstance(prompt_ids, list) or not isinstance(answer_ids, list)
            or not prompt_ids or not answer_ids):
        raise ValueError('Nonempty original prompt and answer axes required')
    full_ids = prompt_ids + answer_ids
    full_mask = source_pack['source_mask'] + [False] * len(answer_ids)
    positions = list(range(len(prompt_ids) - 1, len(full_ids)))
    if (axes['token_ids'] != full_ids or axes['answer_ids'] != answer_ids
            or axes['source_mask'] != full_mask or axes['query_positions'] != positions):
        raise ValueError('Current tokenizer does not reproduce original full token/mask/T+1 axes')
    offsets = _exact_offsets(encoded['offset_mapping'], len(chat))
    offsets.extend(_exact_offsets(response_pack['offsets'], len(sample.response), len(chat)))
    mapping = align_source_spans(inventory, sample.source, chat + sample.response,
        source_start, offsets, full_ids, axes['token_ids'], full_mask)
    return dict(status='original_axes_matched', alignment=mapping,
                prompt_tokens=len(prompt_ids), answer_tokens=len(answer_ids),
                full_tokens=len(full_ids), query_rows=len(positions),
                prompt_character_count=len(chat), response_sha256=text_hash(sample.response),
                source_semantic_status='unknown_not_inferred',
                tokenizer_identity_scope='Operational equality on this case; original tokenizer body hash unattested',
                model_forwards=0, detector_scores=0)


def _tokenizer_assets(tokenizer_root, data_root, manifest_path, capture_root):
    manifest = read_json(manifest_path)
    environment_path = capture_root / 'MODEL_ENVIRONMENT.json'
    if manifest.get('source_model_environment_sha256') != file_hash(environment_path):
        raise ValueError('Asset manifest is not bound to the original capture environment')
    environment = read_json(environment_path)
    if not tokenizer_root.is_relative_to(data_root):
        raise ValueError('Tokenizer root escapes the declared data root')
    rows = {row['relative_path']: row for row in manifest['files']}
    bindings, descriptions = {}, []
    for name in TOKENIZER_METADATA:
        path = tokenizer_root / name
        relative = path.relative_to(data_root).as_posix()
        if relative not in rows:
            raise ValueError('Tokenizer metadata is absent from the pinned asset manifest')
        row, digest = rows[relative], file_hash(path)
        if digest != row['sha256'] or path.stat().st_size != row['size']:
            raise ValueError('Tokenizer metadata differs from the witnessed asset manifest')
        if name in ('config.json', 'tokenizer_config.json'):
            if digest != environment['files'][name]['sha256']:
                raise ValueError('Tokenizer/config metadata differs from original capture hashes')
        bindings[str(path)] = digest
        descriptions.append(dict(name=name, sha256=digest, identity_status=row['identity_status']))
    return bindings, descriptions


def run(args):
    """Run a memory-claimed engineering alignment on qualified cases only.

    Required: capture_root, prepared, inventory_root, qualification_root,
    tokenizer, asset_manifest, data_root, output. Optional ids is an explicit
    list or JSON file (array or {"ids": array}); default is every qualified case
    in the supplied qualification root. Passing a later disjoint M1 run extends
    coverage without retokenizing previously aligned canaries.
    """
    started = time.perf_counter()
    root, prepared, inventory_root, qualification_root, tokenizer_root, data_root, output = (
        Path(getattr(args, name)).resolve() for name in
        ('capture_root', 'prepared', 'inventory_root', 'qualification_root', 'tokenizer', 'data_root', 'output'))
    manifest_path = Path(args.asset_manifest).resolve()
    if output.exists():
        raise FileExistsError('Alignment output already exists')
    inventory_seal = verify_seal(inventory_root, required=('plan.json', 'population.json', 'occurrences.jsonl'))
    qualification_seal = verify_seal(qualification_root, required=('plan.json', 'summary.json'))
    bindings = {str(manifest_path): file_hash(manifest_path),
                str(inventory_root / 'seal.json'): file_hash(inventory_root / 'seal.json'),
                str(qualification_root / 'seal.json'): file_hash(qualification_root / 'seal.json')}
    asset_bindings, asset_descriptions = _tokenizer_assets(tokenizer_root, data_root, manifest_path, root)
    bindings.update(asset_bindings)
    observer = capture_identity(root, tokenizer_root / 'config.json')
    input_path, prepared_manifest = prepared / 'inputs.jsonl', prepared / 'manifest.json'
    bindings[str(input_path)], bindings[str(prepared_manifest)] = file_hash(input_path), file_hash(prepared_manifest)
    if read_json(prepared_manifest).get('inputs_sha256') != bindings[str(input_path)]:
        raise ValueError('Prepared input bytes differ from the released manifest')
    m0_plan, m1_plan = read_json(inventory_root / 'plan.json'), read_json(qualification_root / 'plan.json')
    if m1_plan.get('observer') != observer:
        raise ValueError('Original observer source/config differs from the M1 qualification')
    if bindings[str(input_path)] not in m0_plan['input_bindings'].values():
        raise ValueError('Prepared QA input is not part of the sealed M0 population')
    summary = read_json(qualification_root / 'summary.json')
    qualified = {str(row['id']): row for row in summary['cases'] if row['status'] == 'qualified'}
    selected = getattr(args, 'ids', None)
    if isinstance(selected, (str, Path)):
        selection_path = Path(selected).resolve()
        bindings[str(selection_path)] = file_hash(selection_path)
        selected = read_json(selection_path)
        selected = selected.get('ids') if isinstance(selected, dict) else selected
    if selected is None:
        selected = list(qualified)
    if (not isinstance(selected, list) or not selected or any(not isinstance(value, str) for value in selected)
            or len(set(selected)) != len(selected) or not set(selected).issubset(qualified)):
        raise ValueError('Unique explicit qualified answer IDs are required')
    witnesses = {str(row['id']): row for row in m1_plan['inputs']}
    population = {row['id']: row for row in read_json(inventory_root / 'population.json')['records']}
    samples = {sample.id: sample for sample in read_samples(input_path)}
    sources_needed = set()
    for answer_id in selected:
        if answer_id not in samples or answer_id not in population or answer_id not in witnesses:
            raise ValueError('Selected case is outside the original prepared/M0/M1 population')
        sample, original, witness = samples[answer_id], population[answer_id], witnesses[answer_id]
        if (sample.task != 'QA' or sample.split != 'test' or original['task'] != 'QA'
                or sample.source_id != original['source_id'] or sample.source_id != witness['source_id']
                or sample.source_id != qualified[answer_id]['source_id']
                or text_hash(sample.source) != original['source_sha256']):
            raise ValueError('Original answer/source identity changed across prepared/M0/M1')
        sources_needed.add(sample.source_id)
        directory = _contained(root / 'capture', answer_id)
        for name in ('arrays.npz', 'record.json'):
            path = directory / name
            digest = file_hash(path)
            if digest != witness['files_sha256'][name]:
                raise ValueError('Original capture differs from its qualified M1 witness')
            bindings[str(path)] = digest
    inventories = {}
    with (inventory_root / 'occurrences.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['task'] == 'QA' and row['source_id'] in sources_needed:
                if row['source_id'] in inventories or row.get('schema') != CONTRACT:
                    raise ValueError('Duplicate or unsupported M0 source occurrence record')
                inventories[row['source_id']] = row
    if set(inventories) != sources_needed:
        raise ValueError('Selected original sources are missing from sealed M0 inventory')
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', dict(schema='h017-relation-alignment-plan-v1', ids=selected,
        source=identity, input_bindings=bindings, observer=observer, tokenizer_assets=asset_descriptions,
        inventory_seal=inventory_seal, qualification_seal=qualification_seal,
        scope='Exact source positions only; semantic relationships and risk remain unassessed',
        annotation_reads=0, model_forwards=0))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, use_fast=True, local_files_only=True,
                                               trust_remote_code=False)
    results = []
    for answer_id in selected:
        sample = samples[answer_id]
        directory = _contained(root / 'capture', answer_id)
        record = read_json(directory / 'record.json')
        if any(str(record[key]) != str(getattr(sample, key)) for key in
               ('id', 'source_id', 'task', 'split', 'generator')):
            raise ValueError('Capture record and original inference input differ')
        try:
            result = reconstruct_alignment(sample, tokenizer, _original_axes(directory / 'arrays.npz'),
                                           inventories[sample.source_id])
        except ValueError as error:
            result = dict(status='axis_mismatch_unknown', reason=str(error),
                          source_semantic_status='unknown_not_inferred', detector_scores=0, model_forwards=0)
        result.update(id=answer_id, source_id=sample.source_id)
        write_json(output / f'answer-{answer_id}.json', result)
        results.append(dict(id=answer_id, source_id=sample.source_id, status=result['status']))
    if any(file_hash(Path(path)) != digest for path, digest in bindings.items()):
        raise RuntimeError('Alignment inputs changed during execution')
    if verify_seal(inventory_root) != inventory_seal or verify_seal(qualification_root) != qualification_seal:
        raise RuntimeError('Upstream sealed results changed during alignment')
    if capture_identity(root, tokenizer_root / 'config.json') != observer:
        raise RuntimeError('Original observer source/config identity changed')
    completed_identity(identity)
    report = dict(schema='h017-relation-alignment-result-v1', cases=results,
        status_counts={status: sum(row['status'] == status for row in results)
                       for status in sorted({row['status'] for row in results})},
        elapsed_seconds=time.perf_counter() - started, semantic_relations_assessed=False,
        original_tokenizer_body_identity_attested=False, annotation_reads=0,
        model_forwards=0, detector_scores=0)
    write_json(output / 'summary.json', report)
    write_json(output / 'seal.json', {path.name: file_hash(path) for path in sorted(output.glob('*.json'))})
    return report
