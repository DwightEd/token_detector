"""Alignment of additional observations used by research readouts."""

import numpy as np
from myproject.measure import answer_axis, canonical_delta
from .distribution import DISTRIBUTION_FIELDS


def window_mean(values: np.ndarray, window: int = 16) -> np.ndarray:
    """Historical offline route window: seven past and eight future positions."""
    values = np.asarray(values, dtype=np.float64)
    target = np.arange(len(values))
    past = (window - 1) // 2
    left = np.maximum(0, target - past)
    right = np.minimum(len(values), target + window - past)
    cumulative = np.r_[0., np.cumsum(values)]
    return (cumulative[right] - cumulative[left]) / (right - left)



def answer_rows(observed: dict, response: dict, answer_index: int) -> dict:
    """Align measurements, excluding only empty offsets and special tokens."""
    if not np.array_equal(observed["token_id"], response['answer_ids']):
        raise ValueError("Measurement and prepared response token IDs differ")
    axis = answer_axis(response, answer_index)
    target = axis['target']
    return {"delta": canonical_delta(observed["delta"], response["units"])[target],
            "route": window_mean(observed["route"])[target], **axis,
            **{key: observed[key][target] for key in DISTRIBUTION_FIELDS if key in observed}}
