"""Native source deletion and prediction-row message measurements."""

import numpy as np
import torch

from .routing import projection_grams
from .source import source_condition
from .distribution import SourceDistributions


class Observer:
    """Keep one frozen model and reuse output-projection Gram matrices."""

    def __init__(self, model):
        self.model = model
        self.grams = projection_grams(model)

    @torch.no_grad()
    def measure(self, source: dict, response: dict, *, full_distribution=False) -> dict:
        """Measure source-present/absent × full/local histories for actual tokens."""
        prompt = source["prompt_with_source"]
        answer = response["answer_ids"]
        distributions = SourceDistributions(answer) if full_distribution else None
        local, full, route = source_condition(
            self.model, prompt, answer, response["units"], source["source_mask"], self.grams,
            on_logits=distributions.present if distributions else None,
        )
        absent_local, absent_full, _ = source_condition(
            self.model, source["prompt_without_source"], answer, response["units"],
            on_logits=distributions.absent if distributions else None,
        )
        result = {
            "delta": np.column_stack((absent_local.astype(np.float64) - local,
                                      absent_full.astype(np.float64) - full)),
            "route": route, "token_id": np.asarray(answer, dtype=np.int64),
        }
        if distributions is not None:
            result.update(distributions.finish())
        return result
