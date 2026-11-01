"""Matched-backend prediction traces and actual post-token final hidden states."""

import numpy as np
import torch

from myproject.model import QUERY_CHUNK, attention_backend, input_ids, prefill


@torch.no_grad()
def hidden_trace(model, prompt, answer):
    if not prompt or not answer:
        raise ValueError('Prompt and answer must be nonempty')
    cache = prefill(model, prompt[:-1])
    queries = [prompt[-1], *answer[:-1]]
    states, likelihoods, entropy, ranks = [], [], [], []
    with attention_backend(model, 'sdpa'):
        for start in range(0, len(answer), QUERY_CHUNK):
            hidden = model.model(input_ids=input_ids(model, queries[start:start+QUERY_CHUNK]),
                                 past_key_values=cache, use_cache=True, return_dict=True).last_hidden_state[0]
            logits = model.lm_head(hidden).float()
            logp = logits.log_softmax(-1)
            targets = input_ids(model, answer[start:start+QUERY_CHUNK])[0]
            actual = logp.gather(1, targets[:, None])[:, 0]
            states.append(hidden.float().cpu().numpy())
            likelihoods.append(actual.cpu().numpy())
            entropy.append(-(logp.exp()*logp).sum(-1).cpu().numpy())
            ranks.append((logp > actual[:, None]).sum(-1).cpu().numpy())
        last = model.model(input_ids=input_ids(model, [answer[-1]]), past_key_values=cache,
                           use_cache=True, return_dict=True).last_hidden_state[0]
    prediction = np.concatenate(states)
    post = np.concatenate([prediction[1:], last.float().cpu().numpy()])
    return {'hidden': post, 'logp': np.concatenate(likelihoods),
            'entropy': np.concatenate(entropy), 'rank': np.concatenate(ranks)}
