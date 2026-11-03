"""Known affirmative-conjunction compiler; a narrow H013 interface diagnostic.

The question template, system and source pool are unchanged from H011. Only the
masked-statement payload becomes the selected single conjunct. This removes a
bundle of context and is not a unique internal-mechanism intervention.
"""
from copy import deepcopy
import hashlib
import json

from . import masked_reconstruction as masked

CONTRACT_VERSION = 'known-affirmative-one-conjunct-reconstruction-v1'


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def prepare_atomic_queries(samples):
    """Retain all original slots, including unknowns, without model/file access."""
    samples = list(samples)
    source = masked.prepare_masked_queries(samples)
    by_id = {s.get('id') if isinstance(s, dict) else s.id: s for s in samples}
    result = dict(schema=CONTRACT_VERSION, sources=deepcopy(source['sources']),
                  originals=deepcopy(source['originals']), queries={}, requests={})
    for original in result['originals']:
        if original['status'] != 'ok':
            continue
        sample = by_id[original['id']]
        response = sample['response'] if isinstance(sample, dict) else sample.response
        begin, end = original['target_span']
        compound = response[begin:end]
        if compound.count(', and ') != 1 or not compound.endswith('. '):
            raise ValueError('Admitted affirmative two-conjunct toy framing changed')
        delimiter = compound.index(', and ')
        for slot in original['occurrences']:
            old_query = source['queries'][slot['query_key']]
            other = next(name for name in old_query['candidates'] if name != slot['text'])
            erased = masked._mask_person_values(compound, slot['text'], other)
            if erased['status'] != 'ok':
                raise ValueError('An admitted compound lost its exact person erasure')
            pieces = erased['masked_response'].split(', and ')
            atom = pieces[0] + '. ' if slot['index'] == 0 else pieces[1]
            if atom.count('[TARGET]') != 1 or '[OTHER]' in atom or not atom.endswith('. '):
                raise ValueError('Known target conjunct must contain one target and no other-person slot')
            question = masked.QUESTION_TEMPLATE.format(masked_response=atom)
            query_key = _key(dict(contract=CONTRACT_VERSION, source=old_query['source'], question=question))
            messages = [deepcopy(old_query['messages'][0]), dict(role='user', content=json.dumps(
                dict(source=old_query['source'], question=question), ensure_ascii=False, separators=(',', ':')))]
            query = dict(query_key=query_key, source_sha256=old_query['source_sha256'],
                source=old_query['source'], event=slot['event'], masked_response=atom,
                question=question, candidates=deepcopy(old_query['candidates']),
                messages=messages, eos_policy=deepcopy(old_query['eos_policy']),
                dangling_other_instruction_retained=True)
            if query_key in result['queries'] and result['queries'][query_key] != query:
                raise ValueError('Conflicting atomic question identity')
            result['queries'][query_key] = query
            observation_keys = []
            for candidate in query['candidates']:
                observation_key = _key(dict(query_key=query_key, candidate=candidate, eos_policy=masked.EOS_POLICY))
                result['requests'][observation_key] = dict(observation_key=observation_key, query_key=query_key,
                    messages=deepcopy(messages), candidate=candidate, eos_policy=deepcopy(masked.EOS_POLICY))
                observation_keys.append(observation_key)
            slot.update(donor_H011_query_key=slot['query_key'], query_key=query_key,
                observation_keys=observation_keys,
                atomic_original_span=([begin, begin + delimiter] if slot['index'] == 0 else
                                      [begin + delimiter + len(', and '), end]),
                atomic_terminal_added='. ' if slot['index'] == 0 else '',
                compound_erasure_map=erased['mask_spans'],
                transformation='select_actual_R_conjunct_and_canonical_period_space')
    slots = [slot for original in result['originals'] for slot in original['occurrences']]
    result['counts'] = dict(originals=len(result['originals']), sources=len(result['sources']),
        logical_queries=len(slots), unique_queries=len(result['queries']),
        logical_requests=sum(len(slot['observation_keys']) for slot in slots),
        unique_requests=len(result['requests']), unknown_occurrences=sum(slot['status'] != 'ok' for slot in slots))
    return result
