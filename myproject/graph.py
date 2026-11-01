"""Exact binary-chain messages on original answer positions."""

import numpy as np


def bounded_message(log_odds: float, penalty: float) -> float:
    """Return a same-state message bounded by the edge penalty."""
    return float(2 * np.arctanh(np.tanh(penalty / 2) * np.tanh(log_odds / 2)))


def chain_log_odds(emission: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """Solve sum(emission*z) - sum(penalty*[neighbor states differ])."""
    emission = np.asarray(emission, dtype=np.float64)
    penalties = np.asarray(penalties, dtype=np.float64)
    if penalties.shape != (len(emission) - 1,):
        raise ValueError("One penalty is required between consecutive packed rows")
    left = np.zeros(len(emission), dtype=np.float64)
    right = np.zeros_like(left)
    for row in range(1, len(emission)):
        left[row] = bounded_message(emission[row - 1] + left[row - 1], penalties[row - 1])
    for row in range(len(emission) - 2, -1, -1):
        right[row] = bounded_message(emission[row + 1] + right[row + 1], penalties[row])
    return emission + left + right


def smooth_scores(score: np.ndarray, target: np.ndarray,
                  answer_index: np.ndarray, strength: float = .5) -> np.ndarray:
    """Apply R4 fixed continuity; gaps weaken edges and answers never connect."""
    if strength == 0:
        return np.asarray(score, dtype=np.float64).copy()
    same_answer = answer_index[1:] == answer_index[:-1]
    gaps = np.diff(np.asarray(target, dtype=np.int64))
    if np.any(same_answer & (gaps <= 0)):
        raise ValueError("Original targets must increase within each answer")
    gate = np.zeros(len(target) - 1, dtype=np.float64)
    gate[same_answer] = 1 / gaps[same_answer]
    return chain_log_odds(2.0 * (score - .5), strength * gate)
