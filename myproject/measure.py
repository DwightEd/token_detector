"""In-memory measurement alignment on the original response token positions."""

import numpy as np


def token_scores(pack: dict, response: dict, scores: np.ndarray) -> list[dict]:
    """Render scores on the existing original token/character axis, without I/O."""
    result = []
    for target, token_id, score in zip(pack['target'], pack['token_id'], scores, strict=True):
        start, end = map(int, response['offsets'][target])
        result.append({'token_id': int(token_id), 'text': response['text'][start:end],
                       'start': start, 'end': end, 'score': float(score)})
    return result


def canonical_delta(values: np.ndarray, units: list[dict]) -> np.ndarray:
    """Preserve the archived unit-mean/deviation floating-point arithmetic."""
    result = np.empty(values.shape, dtype=np.float64)
    for unit in units:
        region = slice(unit["start"], unit["stop"])
        weights = np.ones(unit["stop"] - unit["start"])
        for column in range(values.shape[1]):
            # The archived vectors were contiguous; BLAS otherwise changes
            # a handful of unit means by one ULP on a strided column.
            part = values[region, column].copy()
            mean = np.dot(part, weights) / weights.sum()
            result[region, column] = mean + (part - mean)
    return result


def answer_axis(response: dict, answer_index: int = 0) -> dict:
    """Original valid-token/unit axes, independent of which signals were measured."""
    token_id = np.asarray(response["answer_ids"])
    offsets = np.asarray(response["offsets"]).reshape(-1, 2)
    if offsets.shape != (len(token_id), 2):
        raise ValueError('Response offsets and tokens differ')
    valid = ((offsets[:, 1] > offsets[:, 0])
             & ~np.isin(token_id, response["special_ids"]))
    unit_index = np.empty(len(token_id), dtype=np.int32)
    for index, unit in enumerate(response["units"]):
        unit_index[unit["start"]:unit["stop"]] = index
    count = int(valid.sum())
    return {
        "target": np.flatnonzero(valid), "token_id": token_id[valid],
        "answer_index": np.full(count, answer_index, dtype=np.int32),
        "unit_index": unit_index[valid],
    }


def unit_ranges(answer_index: np.ndarray, unit_index: np.ndarray) -> tuple:
    """Return contiguous answer/unit ranges on the original packed row order."""
    changed = ((answer_index[1:] != answer_index[:-1])
               | (unit_index[1:] != unit_index[:-1]))
    starts = np.r_[0, np.flatnonzero(changed) + 1]
    return starts, np.r_[starts[1:], len(answer_index)]
