"""Extract source contrasts and score existing responses."""

from pathlib import Path

import numpy as np
import torch

from .inputs import encode_response, encode_source
from .measure import answer_axis, canonical_delta, token_scores
from .model import load_pretrained, prefill, score_continuation
from .observations import Observations
from .witness import WitnessDetector


class HallucinationDetector:
    """A frozen causal model and tokenizer, with the V3 witness readout."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device: str = "cuda:0"):
        model, tokenizer = load_pretrained(model_path, device)
        return cls(model, tokenizer)

    @torch.no_grad()
    def extract(self, prompt: str, response: str, source: str,
                answer_index: int = 0) -> Observations:
        """Measure full-history source deletion on the original response axis.

        Source must occur exactly once in prompt; response must tokenize to at
        least one token. This method neither reads labels nor writes files.
        """
        encoded_source = encode_source(self.tokenizer, prompt, source)
        encoded = encode_response(self.tokenizer, response)
        answer = encoded["answer_ids"]
        if not answer:
            raise ValueError("response must contain at least one token")
        likelihoods = []
        for name, backend in (("prompt_with_source", "eager"),
                              ("prompt_without_source", "sdpa")):
            tokens = encoded_source[name]
            cache = prefill(self.model, tokens[:-1])
            likelihoods.append(score_continuation(
                self.model, cache, tokens[-1], answer, backend=backend))
        gap = likelihoods[1].astype(np.float64) - likelihoods[0]
        # Preserve the original readout's unit-centered floating-point order.
        gap = canonical_delta(gap[:, None], encoded["units"])[:, 0]
        axis = answer_axis(encoded, answer_index)
        return Observations({"full_gap": gap[axis["target"]], **axis}, encoded)

    @staticmethod
    def score(observed: Observations) -> list[dict]:
        """Score extracted data without loading or running a language model."""
        scores = WitnessDetector().score(observed.pack)
        return token_scores(observed.pack, observed.response, scores)

    def predict(self, prompt: str, response: str, source: str) -> list[dict]:
        if not response:
            return []
        return self.score(self.extract(prompt, response, source))
