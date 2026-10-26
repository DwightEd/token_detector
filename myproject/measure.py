"""In-memory measurement alignment on the original response token positions."""

import numpy as np


def window_mean(values: np.ndarray, window: int = 16) -> np.ndarray:
    """Historical offline route window: seven past and eight future positions."""
    values = np.asarray(values, dtype=np.float64)
    target = np.arange(len(values))
    past = (window - 1) // 2
    left = np.maximum(0, target - past)
    right = np.minimum(len(values), target + window - past)
    cumulative = np.r_[0., np.cumsum(values)]
    return (cumulative[right] - cumulative[left]) / (right - left)


def canonical_delta(values: np.ndarray, units: list[dict]) -> np.ndarray:
    """Preserve the archived unit-mean/deviation floating-point arithmetic."""
    result = np.empty(values.shape, dtype=np.float64)
    for unit in units:
        region = slice(unit["start"], unit["stop"])
        weights = np.ones(unit["stop"] - unit["start"])
        for column in range(2):
            # The archived vectors were contiguous; BLAS otherwise changes
            # a handful of unit means by one ULP on a strided column.
            part = values[region, column].copy()
            mean = np.dot(part, weights) / weights.sum()
            result[region, column] = mean + (part - mean)
    return result


def answer_rows(observed: dict, response: dict, answer_index: int) -> dict:
    """Align measurements, excluding only empty offsets and special tokens."""
    token_id = np.asarray(response["answer_ids"])
    if not np.array_equal(observed["token_id"], token_id):
        raise ValueError("Measurement and prepared response token IDs differ")
    offsets = np.asarray(response["offsets"])
    valid = ((offsets[:, 1] > offsets[:, 0])
             & ~np.isin(token_id, response["special_ids"]))
    unit_index = np.empty(len(token_id), dtype=np.int32)
    for index, unit in enumerate(response["units"]):
        unit_index[unit["start"]:unit["stop"]] = index
    count = int(valid.sum())
    return {
        "delta": canonical_delta(observed["delta"], response["units"])[valid],
        "route": window_mean(observed["route"])[valid],
        "target": np.flatnonzero(valid), "token_id": token_id[valid],
        "answer_index": np.full(count, answer_index, dtype=np.int32),
        "unit_index": unit_index[valid],
    }
