"""Frozen Llama loading, KV prefill and token likelihoods."""

from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


PREFILL_CHUNK = 256
QUERY_CHUNK = 16


def load_model(path, device):
    """Load the source-observation model with its original native BF16 arithmetic."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    return model


def input_ids(model, tokens):
    return torch.tensor([tokens], device=model.device, dtype=torch.long)


@contextmanager
def attention_backend(model, implementation):
    """Restore the caller's attention backend even when a native forward fails."""
    previous = model.config._attn_implementation
    model.set_attn_implementation(implementation)
    try:
        yield
    finally:
        model.set_attn_implementation(previous)


def prefill(model, tokens):
    cache = DynamicCache()
    with attention_backend(model, "sdpa"):
        for start in range(0, len(tokens), PREFILL_CHUNK):
            model.model(
                input_ids=input_ids(model, tokens[start:start + PREFILL_CHUNK]),
                past_key_values=cache, use_cache=True, return_dict=True,
            )
    return cache


def score_continuation(model, cache, last_prompt_token, answer):
    """Query P+t-1 predicts y_t; local callers reset the prefix before a unit."""
    queries = [last_prompt_token, *answer[:-1]]
    likelihoods = []
    with attention_backend(model, "sdpa"):
        for start in range(0, len(answer), QUERY_CHUNK):
            hidden = model.model(
                input_ids=input_ids(model, queries[start:start + QUERY_CHUNK]),
                past_key_values=cache, use_cache=True, return_dict=True,
            ).last_hidden_state[0]
            logits = model.lm_head(hidden).float()
            targets = input_ids(model, answer[start:start + QUERY_CHUNK])[0]
            actual = logits.gather(1, targets[:, None])[:, 0]
            likelihoods.append((actual - logits.logsumexp(-1)).cpu().numpy())
    return np.concatenate(likelihoods)
