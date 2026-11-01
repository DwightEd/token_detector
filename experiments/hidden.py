"""Frozen source/answer states with original token and character alignment."""

import numpy as np
import torch

from myproject.inputs import encode_response, encode_source
from myproject.model import input_ids


@torch.no_grad()
def hidden_observation(model, tokenizer, source_text, answer_text, prompt=None):
    """Source states from full pass; answer states from physical source deletion.

    States belong to the actual input token (post-token final norm). Prediction
    rows are separately shifted by one for the matched full-history source gap.
    This log-probability comparator is not the complete frozen R4 recipe.
    """
    if prompt is None:
        prompt = f'Summarize the following passage faithfully:\n{source_text}\n\nSummary:'
    source = encode_source(tokenizer, prompt, source_text)
    response = encode_response(tokenizer, answer_text)
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != source['prompt_with_source']:
        raise ValueError('Prompt tokenization changed during offset alignment')
    beginning = chat.index(prompt) + prompt.index(source_text)
    offsets = np.asarray(encoded['offset_mapping'])
    source_rows = np.flatnonzero(source['source_mask'])
    source_offsets = offsets[source_rows] - beginning
    answer_ids = response['answer_ids']
    answer_hidden = source_hidden = None
    log_probabilities = []
    for present in [True, False]:
        prefix = source['prompt_with_source' if present else 'prompt_without_source']
        hidden = model.model(input_ids=input_ids(model, prefix + answer_ids),
                             use_cache=False, return_dict=True).last_hidden_state[0]
        if present:
            source_hidden = hidden[source_rows].float().cpu().numpy()
        else:
            answer_hidden = hidden[len(prefix):].float().cpu().numpy()
        parts = []
        for first in range(0, len(answer_ids), 16):
            last = min(first + 16, len(answer_ids))
            logits = model.lm_head(hidden[len(prefix) - 1 + first:len(prefix) - 1 + last]).float()
            targets = input_ids(model, answer_ids[first:last])[0]
            parts.append((logits.gather(1, targets[:, None])[:, 0]
                          - logits.logsumexp(-1)).cpu().numpy())
        log_probabilities.append(np.concatenate(parts).astype(np.float64))
        del hidden
    return {
        'source_hidden': source_hidden, 'source_offsets': source_offsets,
        'answer_hidden': answer_hidden, 'answer_offsets': np.asarray(response['offsets']),
        'answer_ids': np.asarray(answer_ids),
        'source_ids': np.asarray(source['prompt_with_source'])[source_rows],
        'source_gap': log_probabilities[1] - log_probabilities[0],
        'source_logp': log_probabilities[0], 'absent_logp': log_probabilities[1],
        'prompt_tokens': len(source['prompt_with_source']),
        'absent_prompt_tokens': len(source['prompt_without_source']),
    }
