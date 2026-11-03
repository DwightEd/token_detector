"""Source-only H017 occurrence inventory using the integrated dataset workflow."""

from collections import Counter
import json
from pathlib import Path
import time

from .data import file_hash, read_json, text_hash, write_json
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .research.relation_inventory import source_inventory, structural_cohorts


def inventory(args):
    """Seal lexical/serialized occurrences, never a semantic verdict or risk."""
    output = Path(args.output).resolve()
    protocol_path = Path(args.protocol).resolve()
    protocol_digest = file_hash(protocol_path)
    protocol = read_json(protocol_path)
    if (protocol.get('stage') != 'h017_source_occurrence_inventory'
            or protocol.get('annotation_reads') != 0):
        raise ValueError('Expected the admitted label-free source inventory protocol')
    if output.exists():
        raise FileExistsError('Inventory output already exists')
    started = time.perf_counter()
    identity = research_identity()
    inputs, sources, population, counts = {}, {}, [], Counter()
    for supplied in args.prepared:
        root = Path(supplied).resolve()
        for name in ('inputs.jsonl', 'manifest.json'):
            inputs[str(root / name)] = file_hash(root / name)
        manifest = read_json(root / 'manifest.json')
        if manifest.get('inputs_sha256') != inputs[str(root / 'inputs.jsonl')]:
            raise ValueError('Prepared input bytes differ from the released manifest')
        root_count = 0
        for sample in read_samples(root / 'inputs.jsonl'):
            root_count += 1
            if sample.split != 'test':
                raise ValueError('This protocol admits only the declared TEST inputs')
            counts[sample.task] += 1
            key = (sample.task, sample.source_id)
            if key in sources and sources[key] != sample.source:
                raise ValueError('A source ID has inconsistent original bytes')
            sources[key] = sample.source
            population.append({'id': sample.id, 'task': sample.task,
                               'source_id': sample.source_id,
                               'source_sha256': text_hash(sample.source)})
        if root_count != manifest.get('samples'):
            raise ValueError('Prepared sample count differs from the released manifest')
    if len({r['id'] for r in population}) != len(population):
        raise ValueError('Duplicate original answer identities')
    if dict(counts) != protocol['expected_task_counts']:
        raise ValueError('Original answer population differs from the protocol')
    input_bindings = {**inputs, str(protocol_path): protocol_digest}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', {'stage': protocol['stage'], 'source': identity,
        'input_bindings': input_bindings, 'protocol': protocol,
        'scope': 'Exact source occurrences and structural paths only; no semantic relation truth',
        'annotation_reads': 0, 'model_forwards': 0})
    totals, source_records = Counter(), []
    with (output / 'occurrences.jsonl').open('x', encoding='utf-8') as stream:
        for (task, source_id), source in sorted(sources.items()):
            record = source_inventory(source, source_id)
            cohorts = structural_cohorts(record)
            record.update(task=task, structural_cohorts=cohorts)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            status = record['status']
            totals[f'{task}:{status}:sources'] += 1
            totals[f'{task}:fields'] += len(record['fields'])
            totals[f'{task}:lexical_occurrences'] += len(record['lexical_occurrences'])
            totals[f'{task}:potential_structural_cohorts'] += len(cohorts)
            source_records.append({'task': task, 'source_id': source_id,
                'source_sha256': record['source_sha256'], 'status': status,
                'fields': len(record['fields']), 'lexical_occurrences': len(record['lexical_occurrences'])})
    if any(file_hash(Path(path)) != digest for path, digest in input_bindings.items()):
        raise ValueError('Source inventory inputs changed while running')
    completed_identity(identity)
    summary = {'answers': len(population), 'task_counts': dict(counts),
        'sources': len(sources), 'source_records': source_records,
        'counts': dict(totals), 'elapsed_seconds': time.perf_counter() - started,
        'semantic_relations_assessed': False, 'token_character_alignment_performed': False,
        'annotation_reads': 0, 'model_forwards': 0, 'detector_scores': 0,
        'scope': 'Occurrence plumbing, not a repeat of prior semantic-frontend experiments'}
    write_json(output / 'population.json', {'records': population})
    write_json(output / 'summary.json', summary)
    write_json(output / 'seal.json', {p.name: file_hash(p) for p in sorted(output.iterdir())})
    return summary
