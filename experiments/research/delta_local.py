"""Source-response direction agreement in fixed semantic locals."""

import numpy as np

from myproject.pooling import population_spans
from myproject.detector import unit_ranges

EPSILON = 1e-12


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite token states')
    return values


def direction_scores(values, answer_index, unit_index, raw_nonzero=None):
    """Algebraically exclude own row; later causal states still contain its token."""
    values = _matrix(values)
    axes = [np.asarray(x) for x in [answer_index, unit_index]]
    if any(x.shape != (len(values),) for x in axes):
        raise ValueError('Metadata must match the original token rows')
    nonzero = np.linalg.norm(values, axis=1) > EPSILON
    if raw_nonzero is not None:
        raw_nonzero = np.asarray(raw_nonzero, bool)
        if raw_nonzero.shape != nonzero.shape:
            raise ValueError('Raw availability must match token rows')
        nonzero &= raw_nonzero
    norm = np.linalg.norm(values, axis=1)
    normalized = np.divide(values, norm[:, None], out=np.zeros_like(values),
                           where=nonzero[:, None])
    score = np.ones(len(values))
    available = np.zeros(len(values), bool)
    starts, stops = unit_ranges(*axes)
    for start, stop in zip(starts, stops, strict=True):
        part = normalized[start:stop]
        others = part.sum(0) - part
        length = np.linalg.norm(others, axis=1)
        valid = nonzero[start:stop] & (length > EPSILON)
        cosine = np.divide(np.sum(part * others, axis=1), length,
                           out=np.zeros(len(part)), where=valid)
        score[start:stop] = np.where(valid, 1-np.clip(cosine, -1., 1.), 1.)
        available[start:stop] = valid
    return {'score': score, 'available': available}


def fit_whitening(values, answer_index, unit_index, source_index):
    """Only FIT moments; reuse the existing source/answer/span weight contract."""
    values = _matrix(values)
    _, weights = population_spans(np.zeros(len(values)), answer_index, unit_index, source_index)
    starts, stops = unit_ranges(answer_index, unit_index)
    row_weights = np.repeat(weights/(stops-starts), stops-starts)
    row_weights /= row_weights.sum()
    mean = row_weights @ values
    varying = np.any(values != values[0], axis=0)
    mean[~varying] = values[0, ~varying]
    variance = row_weights @ (values-mean)**2
    scale = np.where(varying & (variance > 0), np.sqrt(variance), 1.)
    return {'mean': mean, 'scale': scale}


def whiten(values, parameters):
    values = _matrix(values)
    mean, scale = parameters['mean'], parameters['scale']
    if (mean.shape != (values.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('Whitening parameters must match state dimension')
    return (values-mean)/scale
