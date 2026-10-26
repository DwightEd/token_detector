"""Present/absent source likelihoods with full and unit-local histories."""

import numpy as np

from .model import (
    QUERY_CHUNK,
    attention_backend,
    input_ids,
    prefill,
    score_continuation,
)
from .routing import observe_route


def source_full(model, cache, prompt, answer, source_mask, grams):
    queries = [prompt[-1], *answer[:-1]]
    likelihoods = []
    routes = []
    energies = {}
    with attention_backend(model, "eager"):
        for start in range(0, len(answer), QUERY_CHUNK):
            with observe_route(model, cache, grams, energies, len(prompt), source_mask, start) as layers:
                hidden = model.model(
                    input_ids=input_ids(model, queries[start:start + QUERY_CHUNK]),
                    past_key_values=cache, use_cache=True, return_dict=True,
                ).last_hidden_state[0]
            logits = model.lm_head(hidden).float().log_softmax(-1)
            targets = input_ids(model, answer[start:start + QUERY_CHUNK])[0]
            likelihoods.append(logits.gather(1, targets[:, None])[:, 0].cpu().numpy())
            source = np.stack([layers[index]["norm_source"] for index in layers], axis=1)
            history = np.stack([layers[index]["norm_history"] for index in layers], axis=1)
            total = np.stack([layers[index]["norm_total"] for index in layers], axis=1)
            ratio = (history - source).sum(-1) / np.maximum(total.sum(-1), 1e-12)
            routes.append(ratio.mean(-1))
    return np.concatenate(likelihoods), np.concatenate(routes)


def source_condition(model, prompt, answer, units, source_mask=None, grams=None):
    """Full history and independent unit-local histories share one prompt prefill."""
    cache = prefill(model, prompt[:-1])
    if source_mask is None:
        full = score_continuation(model, cache, prompt[-1], answer)
        route = None
    else:
        full, route = source_full(model, cache, prompt, answer, source_mask, grams)
    local = np.empty_like(full)
    for unit in units:
        cache.crop(len(prompt) - 1)
        start, stop = unit["start"], unit["stop"]
        local[start:stop] = score_continuation(model, cache, prompt[-1], answer[start:stop])
    return local, full, route
