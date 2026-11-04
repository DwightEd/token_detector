"""Original-prompt, post-token attention routing; no graph walks or retokenization."""

import numpy as np
import torch

from myproject.model import QUERY_CHUNK, attention_backend, input_ids, prefill


@torch.no_grad()
def attention_profiles(model, prompt, answer):
    """Keep prompt identities, past answer identities and one self coordinate.

    Returned head-parity views are correlated diagnostics. Segmentation rows P+t
    belong to actual y_t states, distinct from P+t-1 token probability rows.
    """
    if not prompt or not answer:
        raise ValueError('Prompt and answer must be nonempty')
    p, n = len(prompt), len(answer)
    result = torch.zeros((2, n, p + n + 1), device=model.device, dtype=torch.float32)
    cache = prefill(model, prompt)
    for first in range(0, n, QUERY_CHUNK):
        last = min(first + QUERY_CHUNK, n)
        handles = []

        def record(_module, _args, output):
            attention = output[1][0].detach().float()
            if attention.shape[0] < 2:
                raise ValueError('Two head-parity views require at least two heads')
            for group in [0, 1]:
                weights = attention[group::2].mean(0)
                result[group, first:last, :p] += weights[:, :p]
                history = weights[:, p:p + last].clone()
                rows = torch.arange(last - first, device=model.device)
                self_weights = history[rows, torch.arange(first, last, device=model.device)].clone()
                history *= (torch.arange(last, device=model.device)[None, :]
                            < torch.arange(first, last, device=model.device)[:, None])
                result[group, first:last, p:p + last] += history
                result[group, first:last, -1] += self_weights

        try:
            for layer in model.model.layers:
                handles.append(layer.self_attn.register_forward_hook(record))
            with attention_backend(model, 'eager'):
                model.model(input_ids=input_ids(model, answer[first:last]),
                            past_key_values=cache, use_cache=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
    values = result.cpu().numpy().astype(np.float64) / len(model.model.layers)
    # BF16 attention rounding can change the sum slightly; retain raw masses too.
    mass = values.sum(-1)
    values /= mass[:, :, None]
    available = np.ones((n, p + n + 1), bool)
    available[:, p:p+n] = np.tril(np.ones((n, n), bool), -1)
    return {'fit': values[0], 'hold': values[1], 'available': available, 'raw_mass': mass,
            'prompt_tokens': p, 'answer_ids': np.asarray(answer)}
