"""Frozen Llama loading, KV prefill and token likelihoods."""

from contextlib import contextmanager
import warnings

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


def continuation_logits(model, cache, last_prompt_token, answer):
    """Yield native logits in query chunks; the caller selects the attention backend."""
    queries = [last_prompt_token, *answer[:-1]]
    for start in range(0, len(answer), QUERY_CHUNK):
        hidden = model.model(
            input_ids=input_ids(model, queries[start:start + QUERY_CHUNK]),
            past_key_values=cache, use_cache=True, return_dict=True,
        ).last_hidden_state[0]
        yield start, model.lm_head(hidden)


def score_continuation(model, cache, last_prompt_token, answer, on_logits=None, *, backend="sdpa"):
    """Query P+t-1 predicts y_t; preserve each backend's existing arithmetic."""
    likelihoods = []
    with attention_backend(model, backend):
        for start, native in continuation_logits(model, cache, last_prompt_token, answer):
            if on_logits is not None:
                on_logits(start, native)
            logits = native.float()
            targets = input_ids(model, answer[start:start + QUERY_CHUNK])[0]
            if backend == "eager":
                actual = logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]
            else:
                actual = logits.gather(1, targets[:, None])[:, 0] - logits.logsumexp(-1)
            likelihoods.append(actual.cpu().numpy())
    return np.concatenate(likelihoods)


def load_pretrained(model_path, device):
    """Load local weights and tokenizer quietly, restoring caller logging state."""
    from huggingface_hub.utils.tqdm import progress_bar_states
    from transformers.utils import logging

    verbosity = logging.get_verbosity()
    progress_enabled = logging.is_progress_bar_enabled()
    hub_progress = dict(progress_bar_states)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"(transformers|huggingface_hub)(\..*)?")
        try:
            logging.set_verbosity_error()
            logging.disable_progress_bar()
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, local_files_only=True,
            )
            model = load_model(model_path, device)
        finally:
            logging.set_verbosity(verbosity)
            if progress_enabled:
                logging.enable_progress_bar()
            else:
                logging.disable_progress_bar()
            progress_bar_states.clear()
            progress_bar_states.update(hub_progress)
    return model, tokenizer
