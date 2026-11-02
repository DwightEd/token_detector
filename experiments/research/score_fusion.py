"""Fixed, label-free batch rank fusion; no accuracy guarantee or fitting."""

import numpy as np
from scipy.stats import rankdata


def equal_rank_fusion(r4, witness):
    """Average ascending midranks on the complete declared scoring population.

    Unknown-label tokens must remain in the input. The result depends on the
    batch population and is a heuristic risk index, not a factual probability.
    """
    # Preserve integer precision: converting scores above 2**53 to float64
    # before ranking can turn distinct scores into ties.
    r4, witness = np.asarray(r4), np.asarray(witness)
    if (r4.ndim != 1 or witness.shape != r4.shape
            or r4.dtype.kind not in 'fiu' or witness.dtype.kind not in 'fiu'
            or not np.isfinite(r4).all() or not np.isfinite(witness).all()):
        raise ValueError('Matching finite one-dimensional score arrays required')
    if not len(r4):
        return np.empty(0)
    left = (rankdata(r4, method='average')-.5)/len(r4)
    right = (rankdata(witness, method='average')-.5)/len(witness)
    return .5*(left+right)
