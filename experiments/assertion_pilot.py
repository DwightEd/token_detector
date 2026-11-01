"""Source-independent dependency structure qualification on original answers."""

from collections import Counter
from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

import numpy as np

from . import data
from .datasets import read_samples
from .fit_ablation import verify_seal
from .native_dataset import read_native
from .provenance import completed_identity, research_identity
from .research.assertions import numeric_assertions, parse_conllu
from .research.table_bindings import YEAR, number, table_regions, literal_links
from myproject.inputs import encode_response


def load_parser(model_path):
    """Load the optional frozen syntactic instrument; never download resources."""
    from ufal.udpipe import Model, Pipeline, ProcessingError

    model = Model.load(str(model_path))
    if model is None:
        raise ValueError(f'Cannot load UDPipe model: {model_path}')
    def parse(text):
        # Pipeline borrows Model's C++ pointer. Capture the model itself and
        # create each document pipeline while that owning reference is alive.
        pipeline = Pipeline(model, 'tokenizer=ranges', Pipeline.DEFAULT,
                            Pipeline.DEFAULT, 'conllu')
        error = ProcessingError()
        conllu = pipeline.process(text, error)
        if error.occurred():
            return {'conllu': conllu, 'tokens': [], 'assertions': [], 'parse_error': error.message}
        try:
            tokens = parse_conllu(text, conllu)
        except ValueError as exc:
            return {'conllu': conllu, 'tokens': [], 'assertions': [], 'parse_error': str(exc)}
        return {'conllu': conllu, 'tokens': tokens,
                'assertions': numeric_assertions(text, tokens), 'parse_error': None}

    return parse


def control_payload(samples):
    """Choose syntax literals from sources before examining any parse or labels."""
    ordered = sorted(samples, key=lambda sample: (data.text_hash(sample.source), sample.id))
    for sample in ordered:
        for _, _, table in table_regions(sample.source):
            header = next(({j: str(value).strip() for j, value in enumerate(row[1:], 1)
                            if re.fullmatch(YEAR, str(value).strip())}
                           for row in table if isinstance(row, list) and
                           sum(bool(re.fullmatch(YEAR, str(c).strip())) for c in row[1:]) >= 2), None)
            if not header:
                continue
            for row in table:
                if not row or not re.fullmatch(r'[A-Za-z][A-Za-z -]{2,60}', str(row[0])):
                    continue
                cells = [(year, number(row[j])) for j, year in header.items() if j < len(row)]
                cells = [(year, value) for year, value in cells if value is not None]
                if (len(cells) < 2 or cells[0][0] == cells[1][0]
                        or Decimal(cells[0][1]) == Decimal(cells[1][1])):
                    continue
                if any(Decimal(value) == Decimal(year) for _, value in cells[:2]
                       for year, _ in cells[:2]):
                    continue
                return {'source_id': sample.source_id, 'sample_id': sample.id,
                        'source_sha256': data.text_hash(sample.source),
                        'source_subject': str(row[0]), 'subject': 'the value for '+str(row[0]),
                        'year0': cells[0][0], 'year1': cells[1][0],
                        'value0': format(Decimal(cells[0][1]), 'f'),
                        'value1': format(Decimal(cells[1][1]), 'f')}
    raise ValueError('No source-derived two-year control payload available')


def structural_controls(payload):
    """Known attachment contrasts; no natural factual-error labels are created."""
    subject = payload['subject']
    y0, y1 = payload['year0'], payload['year1']
    v0, v1 = payload['value0'], payload['value1']
    difference = str(Decimal(v1)-Decimal(v0))
    direct = [
        ('year_first', f'In {y0}, {subject} was {v0}.', [(y0, v0)]),
        ('year_last', f'{subject} was {v0} in {y0}.', [(y0, v0)]),
        ('separate', f'In {y0}, {subject} was {v0}. In {y1}, {subject} was {v1}.',
         [(y0, v0), (y1, v1)]),
        ('prior_distractor', f'The previously reported value was {v0}. In {y1}, {subject} was {v1}.',
         [(y1, v1)]),
    ]
    harder = [
        ('coordination', f'In {y0}, {subject} was {v0}, and in {y1}, {subject} was {v1}.',
         [(y0, v0), (y1, v1)]),
        ('pronoun_coordination', f'In {y0}, {subject} was {v0}, while in {y1} it was {v1}.',
         [(y0, v0), (y1, v1)]),
        ('from_to', f'{subject} increased from {v0} in {y0} to {v1} in {y1}.',
         [(y0, v0), (y1, v1)]),
    ]
    guards = [
        ('arithmetic', f'The difference is {v1} - ({v0}) = {difference}.'),
        ('negation', f'It is not true that in {y0}, {subject} was {v1}.'),
        ('hypothesis', f'If in {y0} {subject} were {v1}, the conclusion would change.'),
        ('quoted_rejection', f'The claim "In {y0}, {subject} was {v1}" is false.'),
    ]
    result = []
    for name, text, pairs in direct+harder:
        expected = []
        subject_matches = list(re.finditer(re.escape(subject), text))
        for pair_index, (year, value) in enumerate(pairs):
            # Unique template coordinates are synthetic structural ground truth.
            def span(literal):
                matches = list(re.finditer(r'(?<![\w.,])'+re.escape(literal)+r'(?!\d|,\d|\.\d)', text))
                if len(matches) != 1:
                    raise ValueError('Ambiguous control literal coordinates')
                return list(matches[0].span())
            if name == 'pronoun_coordination' and pair_index == 1:
                mention = re.search(r'\bit\b', text)
            else:
                mention = subject_matches[min(pair_index, len(subject_matches)-1)]
            expected.append({'year': year, 'value': value,
                             'year_span': span(year), 'value_span': span(value),
                             'subject_text': mention.group(), 'subject_span': list(mention.span())})
        result.append({'name': name, 'text': text, 'expected': expected,
                       'simple': name in {item[0] for item in direct}, 'guard': False})
    return result + [{'name': name, 'text': text, 'expected': [],
                      'simple': False, 'guard': True} for name, text in guards]


def check_control(control, parsed):
    active = [r for r in parsed['assertions']
              if r['status'] == 'candidate' and r['kind'] == 'assertion']
    observed = Counter((r['condition_year']['text'], number(r['amount']['text']),
                        tuple(r['condition_year']['span']), tuple(r['amount']['span']),
                        r['subject']['text'], tuple(r['subject']['span']))
                       for r in active if r.get('condition_year')
                       and r.get('amount') and r.get('subject'))
    expected = Counter((item['year'], number(item['value']), tuple(item['year_span']),
                        tuple(item['value_span']), item['subject_text'], tuple(item['subject_span']))
                       for item in control['expected'])
    wrong = observed-expected
    missing = expected-observed
    passed = not active if control['guard'] else not wrong and (not missing or not control['simple'])
    passed = passed and not parsed.get('parse_error')
    return {'passed': bool(passed), 'observed_pairs': list(observed.elements()),
            'missing_pairs': list(missing.elements()), 'wrong_pairs': list(wrong.elements()),
            'active_assertions': len(active),
            'criterion': 'no active direct assertions' if control['guard'] else
                         ('exact attachment recovery' if control['simple'] else 'no incorrect pair; abstention allowed')}


def project_roles(text, records, offsets):
    """Map character roles to unchanged scored-token indices, retaining duplicates."""
    offsets = np.asarray(offsets)
    projected = []
    for record in records:
        roles = {}
        for name in ('subject', 'condition_year', 'amount', 'predicate'):
            role = record.get(name)
            if role is None:
                roles[name] = []
                continue
            a, b = role['span']
            if not 0 <= a < b <= len(text) or text[a:b] != role['text']:
                raise ValueError(f'Invalid original character role: {name}')
            roles[name] = np.flatnonzero((offsets[:, 0] < b) & (offsets[:, 1] > a)).tolist()
        projected.append(roles)
    return projected


def prepared_inputs(prepared):
    """Read a whole inference-only split, including hidden/unalignable labels.

    Text parsing also accepts structured-source claims. That does not make their
    sources supported by the text detector or supply native token annotations.
    """
    manifest = data.read_json(prepared/'manifest.json')
    if manifest['selection'].get('limit') is not None or manifest['selection'].get('ids'):
        raise ValueError('Assertion coverage requires an unfiltered prepared split')
    if data.file_hash(prepared/'inputs.jsonl') != manifest['inputs_sha256']:
        raise ValueError('Prepared assertion inputs changed')
    for name in ('raw_files', 'provenance_files', 'evidence_files'):
        for path, expected in manifest.get(name, {}).items():
            if data.file_hash(Path(path)) != expected:
                raise ValueError(f'Prepared assertion dependency changed: {path}')
    samples = list(read_samples(prepared/'inputs.jsonl'))
    if len(samples) != manifest['samples'] or not samples:
        raise ValueError('Prepared assertion population changed')
    if any(not isinstance(s.response, str) or not isinstance(s.source, str) for s in samples):
        raise ValueError('Assertion inputs must retain original text')
    return manifest, samples


def reference_controls(reference, model_hash, identity):
    """Reuse one sealed instrument panel; no table requirement on new data."""
    verify_seal(reference)
    plan = data.read_json(reference/'plan.json')
    if data.file_hash(Path(plan['protocol'])) != plan['protocol_sha256']:
        raise ValueError('Control reference protocol changed')
    if (plan['model_sha256'] != model_hash or
            plan['source']['source_sha256']['experiments.research/assertions.py'] !=
            identity['source_sha256']['experiments.research/assertions.py']):
        raise ValueError('Control reference uses a different model or role adapter')
    records = data.read_json(reference/'controls.json')['records']
    if any(json.dumps(check_control(r, r), sort_keys=True) !=
           json.dumps(r['check'], sort_keys=True) for r in records):
        raise ValueError('Control reference checks do not recompute')
    return records


def response_axis(tokenizer, response):
    """Use the public observer's exact tokenization rule, without observer calls."""
    encoded = encode_response(tokenizer, response)
    ids, offsets = np.asarray(encoded['answer_ids']), np.asarray(encoded['offsets'])
    if not len(ids):
        return np.empty((0, 2), dtype=int), []
    target = ((offsets[:, 1] > offsets[:, 0]) & ~np.isin(ids, encoded['special_ids']))
    return offsets[target], ids[target].tolist()


def tokenizer_files(path):
    """Snapshot the exact tokenizer/config file set without hashing LM weights."""
    weights = {'.safetensors', '.bin', '.pt', '.h5', '.msgpack'}
    return {str(p): data.file_hash(p) for p in path.iterdir()
            if p.is_file() and p.suffix not in weights}


def run(native, model_path, protocol, output, *, prepared=None, tokenizer_path=None,
        control_reference=None):
    """Measure full natural coverage and fixed controls without detector labels."""
    model_path, protocol, output = [Path(p).resolve() for p in (model_path, protocol, output)]
    if (native is None) == (prepared is None):
        raise ValueError('Provide exactly one native or prepared input')
    tokenizer, tokenizer_hashes = None, {}
    if native is not None:
        native = Path(native).resolve()
        plan, all_samples, records = read_native(native)
        pairs = [(sample, record) for (group, sample), record in zip(all_samples, records, strict=True)
                 if plan['groups'][group]['role'] == 'test']
        samples = [sample for sample, _ in pairs]
        input_identity = {'native': str(native), 'native_seal_sha256': data.file_hash(native/'seal.json'),
                          'axis_origin': 'unchanged native observer tokens'}
    else:
        from transformers import AutoTokenizer

        prepared = Path(prepared).resolve()
        if tokenizer_path is None:
            raise ValueError('Prepared assertion inputs require the explicit observer tokenizer')
        tokenizer_path = Path(tokenizer_path).resolve()
        manifest, samples = prepared_inputs(prepared)
        pairs = [(sample, None) for sample in samples]
        tokenizer_hashes = tokenizer_files(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
        if not tokenizer.is_fast:
            raise ValueError('Original character mapping requires a fast tokenizer')
        input_identity = {'prepared': str(prepared), 'manifest_sha256': data.file_hash(prepared/'manifest.json'),
                          'inputs_sha256': manifest['inputs_sha256'], 'dataset': manifest['dataset'],
                          'split': manifest['split'], 'source_modality': manifest['modality'],
                          'native_label_granularities': manifest['granularities'],
                          'tokenizer': str(tokenizer_path), 'tokenizer_files_sha256': tokenizer_hashes,
                          'axis_origin': 'public encode_response tokenization; no observer scores collected'}
    if not samples:
        raise ValueError('No evaluation responses in the original input')
    output.mkdir(parents=True, exist_ok=False)
    (output/'records').mkdir()
    identity, model_hash, protocol_hash = research_identity(), data.file_hash(model_path), data.file_hash(protocol)
    metadata = {**input_identity, 'model': str(model_path), 'model_sha256': model_hash, 'udpipe_version': version('ufal.udpipe'),
                'protocol': str(protocol), 'protocol_sha256': protocol_hash, 'source': identity,
                'annotation_reads': 0, 'new_lm_calls': 0, 'external_syntax_supervision': True}
    payloads, controls, remaining = [], [], list(samples)
    if control_reference is None:
        for subject_index in range(4):
            payload = control_payload(remaining)
            payloads.append(payload)
            for control in structural_controls(payload):
                controls.append(control | {'name': f'{subject_index}:{control["name"]}',
                                           'source_subject_index': subject_index})
            remaining = [s for s in remaining if data.text_hash(s.source) != payload['source_sha256']]
    else:
        control_reference = Path(control_reference).resolve()
        control_results = reference_controls(control_reference, model_hash, identity)
        metadata.update(control_reference=str(control_reference),
                        control_reference_seal_sha256=data.file_hash(control_reference/'seal.json'),
                        controls_scope='sealed instrument panel, not target-dataset relation accuracy')
    data.write_json(output/'plan.json', metadata | {'control_payloads': payloads, 'controls': controls})
    parse, started = load_parser(model_path), time.monotonic()
    if control_reference is None:
        control_results = []
        for control in controls:
            parsed = parse(control['text'])
            control_results.append(control | parsed | {'check': check_control(control, parsed)})
    data.write_json(output/'controls.json', {'records': control_results})
    summary, index, reasons = Counter(), [], Counter()
    for i, (sample, record) in enumerate(pairs, 1):
        parsed = parse(sample.response)
        if native is not None:
            with np.load(native/record['file'], allow_pickle=False) as archive:
                offsets = archive['offsets'][archive['target']]
                token_ids = archive['token_id'].tolist()
        else:
            offsets, token_ids = response_axis(tokenizer, sample.response)
        projections = project_roles(sample.response, parsed['assertions'], offsets)
        candidate = [r for r in parsed['assertions'] if r['status'] == 'candidate' and
                     r['kind'] == 'assertion' and r.get('subject') and r.get('condition_year') and r.get('amount')]
        reasons.update(r['reason'] for r in parsed['assertions'])
        summary.update({'responses': 1, 'parser_tokens': len(parsed['tokens']),
                        'numeric_records': len(parsed['assertions']), 'candidate_records': len(candidate),
                        'candidate_responses': int(bool(candidate)),
                        'parser_failures': int(parsed['parse_error'] is not None),
                        'responses_without_text_source': int(not sample.source.strip()),
                        'responses_without_numeric_records': int(not parsed['assertions'])})
        row = {'id': sample.id, 'source_id': sample.source_id, 'response': sample.response,
               'source_sha256': data.text_hash(sample.source), 'response_sha256': data.text_hash(sample.response),
               'scored_token_ids': token_ids, 'scored_offsets': offsets.tolist(),
               'projected_roles': projections, 'legacy_literal_links': literal_links(sample.response), **parsed}
        path = data.prediction_path(output/'records', sample.id)
        data.write_json(path, row)
        index.append({'id': sample.id, 'file': str(path.relative_to(output)), 'sha256': data.file_hash(path),
                      'candidate_records': len(candidate)})
        print('PARSED', i, len(samples), sample.id, len(candidate), flush=True)
    gate1 = all(r['check']['passed'] for r in control_results)
    gate2 = summary['candidate_responses'] >= 10
    report = {'counts': dict(summary), 'reasons': dict(reasons), 'structural_controls_passed': gate1,
              'natural_coverage_gate_passed': gate2, 'structure_instrument_gate_passed': gate1 and gate2,
              'detector_scoring_permitted': False, 'annotation_reads': 0, 'new_lm_calls': 0,
              'seconds': time.monotonic()-started, 'records': index, 'source': completed_identity(identity)}
    if data.file_hash(model_path) != model_hash or data.file_hash(protocol) != protocol_hash:
        raise ValueError('Parser model/protocol changed during qualification')
    if native is not None:
        read_native(native)
        if data.file_hash(native/'seal.json') != input_identity['native_seal_sha256']:
            raise ValueError('Native assertion input was resealed during measurement')
    else:
        prepared_inputs(prepared)
        if data.file_hash(prepared/'manifest.json') != input_identity['manifest_sha256']:
            raise ValueError('Prepared manifest changed during assertion measurement')
        if tokenizer_files(tokenizer_path) != tokenizer_hashes:
            raise ValueError('Observer tokenizer changed during assertion measurement')
    if control_reference is not None:
        if data.file_hash(control_reference/'seal.json') != metadata['control_reference_seal_sha256']:
            raise ValueError('Control reference seal changed during measurement')
        reference_controls(control_reference, model_hash, identity)
    data.write_json(output/'summary.json', report)
    data.write_json(output/'seal.json', {name: data.file_hash(output/name)
                                       for name in ('plan.json', 'controls.json', 'summary.json')})
    return report
