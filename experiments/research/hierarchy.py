"""Fixed-concentration lexical hierarchy retaining every token likelihood factor."""

from functools import lru_cache

import numpy as np
from scipy.special import betaln, logsumexp

from myproject.measure import unit_ranges
from experiments.research.pooling import cavity_scores, forward
from experiments.research.segmentation import legal_atoms
from experiments.research.readouts import SharedDetector


@lru_cache(maxsize=128)
def quadrature(length):
    count = (length+2)//2
    nodes, weights = np.polynomial.legendre.leggauss(count)
    theta, log_weight = (nodes+1)/2, np.log(weights/2)
    theta.setflags(write=False); log_weight.setflags(write=False)
    return theta, log_weight


def hierarchical_scores(values, atom_index):
    """Integrate span theta and atom rho; own contrast is never averaged.

    theta~Beta(1,1), rho_a|theta~Beta(theta,1-theta). Gaussian integration
    is polynomial-exact at the selected order, up to floating-point error.
    Scores are source-expert responsibility odds, not semantic-error probabilities.
    """
    values, atom_index = np.asarray(values, float), np.asarray(atom_index)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or atom_index.shape != values.shape or atom_index.dtype.kind not in 'iu'):
        raise ValueError('Finite contrasts and matching integer atom indices required')
    starts = np.r_[0, np.flatnonzero(atom_index[1:] != atom_index[:-1])+1]
    stops = np.r_[starts[1:], len(values)]
    if len(np.unique(atom_index)) != len(starts):
        raise ValueError('Each lexical atom must be contiguous')
    theta, log_weight = quadrature(len(values))
    terminal, likelihoods = {}, []
    for a, b in zip(starts, stops, strict=True):
        size = b-a
        if size not in terminal:
            k = np.arange(size+1)
            terminal[size] = (betaln(theta[:, None]+k, 1-theta[:, None]+size-k)
                              - betaln(theta, 1-theta)[:, None])
        likelihoods.append(logsumexp(terminal[size]+forward(values[a:b])[-1], axis=1))
    likelihoods = np.asarray(likelihoods)
    total = likelihoods.sum(axis=0)
    result = np.empty_like(values)
    for atom, (a, b) in enumerate(zip(starts, stops, strict=True)):
        terms = terminal[b-a]
        other_atoms = log_weight+total-likelihoods[atom]
        for t in range(a, b):
            other = np.delete(values[a:b], t-a)
            coefficients = forward(other)[-1] if len(other) else np.array([0.])
            absent = logsumexp(terms[:, 1:]+coefficients, axis=1)
            present = logsumexp(terms[:, :-1]+coefficients, axis=1)
            result[t] = values[t]+logsumexp(other_atoms+absent)-logsumexp(other_atoms+present)
    return result


def shared_exploration(pack, text, offsets, task='QA'):
    """Match the primary flat readout and two prespecified lexical controls."""
    cuts, _ = legal_atoms({'text': text, 'offsets': offsets})
    atoms = np.searchsorted(cuts, pack['target'], side='right')-1
    full = pack['delta'][:, 1]
    hierarchical, atomic = np.empty_like(full), np.empty_like(full)
    starts, stops = unit_ranges(pack['answer_index'], pack['unit_index'])
    for a, b in zip(starts, stops, strict=True):
        local = atoms[a:b]
        hierarchical[a:b] = hierarchical_scores(full[a:b], local)
        boundaries = np.r_[0, np.flatnonzero(local[1:] != local[:-1])+1, b-a]
        sums = np.add.reduceat(full[a:b], boundaries[:-1])
        atomic[a:b] = np.repeat(cavity_scores(sums)['score'], np.diff(boundaries))
    return {'shared_score': SharedDetector(task).score(pack),
            'hierarchical_score': hierarchical, 'atomic_shared': atomic}
