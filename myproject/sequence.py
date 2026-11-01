"""Joint source-expert tendency and persistence, integrated without fitting.

The latent states allocate predictive experts; they are not factuality labels.
This readout retains the original token factor and span boundaries.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from .detector import unit_ranges


@lru_cache(maxsize=128)
def _grid(n, copying, prevalence='integrated'):
    # Polynomial degrees are <=n in pi and <=n-1 in c. Positive quadrature
    # therefore integrates all evidence and marginal numerators exactly.
    # The matched prior contributes 2*c, and the reported posterior c mean
    # contributes one more power. Include that diagnostic in the exact budget.
    count = 1 if prevalence == 'fixed' else (n+3)//2 if prevalence == 'matched' else n//2+1
    nodes, weights = leggauss(count)
    pi, weights = (nodes+1)/2, weights/2
    if prevalence == 'integrated':
        c, c_weights = (pi, weights) if copying else (np.array([0.]), np.array([1.]))
    elif prevalence in {'symmetric', 'matched'} and copying:
        c, c_weights = pi, weights
        if prevalence == 'matched':
            c_weights = c_weights*(2*c)
        pi, weights = np.array([.5]), np.array([1.])
    elif prevalence == 'fixed' and copying:
        c, c_weights = np.array([2/3]), np.array([1.])
    elif prevalence == 'fixed_persistence' and copying:
        c, c_weights = np.array([.5]), np.array([1.])
    else:
        raise ValueError('Unknown prevalence prior or fixed prevalence without copying')
    pi, c = np.meshgrid(pi, c, indexing='ij')
    pi, c = pi.ravel(), c.ravel()
    log_weights = np.log(np.multiply.outer(weights, c_weights).ravel())
    transition = np.array([(1-pi)+c*pi, (1-c)*pi,
                           (1-c)*(1-pi), c+(1-c)*pi])
    return pi, c, log_weights, np.log(transition)


def _check_prevalence(prevalence, *, witness, copying=True, onset=False):
    if prevalence not in {'integrated', 'symmetric', 'matched', 'fixed', 'fixed_persistence'}:
        raise ValueError('Unknown prevalence prior')
    if prevalence != 'integrated' and (not witness or not copying or onset):
        raise ValueError('Nondefault prior requires witness emissions and copying, without onset')


def witness_emissions(values):
    """Likelihood ratio of (p_source+p_absent)/2 relative to p_source.

    A risky region can still contain source-predictable tokens. This nested
    alternative adds no information and does not identify factual errors.
    """
    return np.logaddexp(0., np.asarray(values, float))-np.log(2.)


def responsibility_logits(values, state_scores):
    """Posterior absent-expert log odds inside the declared witness hierarchy.

    P(absent|d)=sigmoid(state_scores)*sigmoid(d). This is a predictive
    allocation diagnostic, not a probability of factual error.
    """
    values, state_scores = np.asarray(values, float), np.asarray(state_scores, float)
    if (values.ndim != 1 or state_scores.shape != values.shape
            or not np.isfinite(values).all() or not np.isfinite(state_scores).all()):
        raise ValueError('Matching finite contrast and state-score vectors required')
    return values+state_scores-np.logaddexp(0., np.logaddexp(values, state_scores))


def persistent_scores(values, *, copying=True, witness=False, onset=False,
                      prevalence='integrated'):
    """Exact two-state forward/backward marginalization under explicit priors.

    P(z_t=1|z_previous,pi,c)=c*z_previous+(1-c)*pi.
    copying=False fixes c=0, recovering exchangeable Beta(1,1) sharing.
    Integrate unnormalized joints, not uniformly averaged node posteriors.
    onset=True uses the absent expert at 0->1 entries (including initial 1),
    and the half-source mixture at 1->1 continuation. No gold onset is used.
    prevalence='symmetric' fixes pi=.5 with uniform c; 'matched' fixes pi=.5
    with c~Beta(2,1), matching the integrated model's adjacent agreement 5/6.
    prevalence='fixed' fixes pi=.5 and c=2/3: a standard fixed HMM with
    the same adjacent agreement 5/6. prevalence='fixed_persistence' retains
    uniform pi and fixes c=.5, also preserving adjacent agreement 5/6.
    All nondefault prevalence modes require
    witness emissions and copying; none introduces a factuality observation.
    """
    _check_prevalence(prevalence, witness=witness, copying=copying, onset=onset)
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite token contrasts required')
    if onset and witness:
        raise ValueError('Choose witness or onset emissions, not both')
    continuation = witness_emissions(values) if onset else values
    if witness:
        values = witness_emissions(values)
    n = len(values)
    pi, c, weights, transition = _grid(n, copying, prevalence)
    t00, t01, t10, t11 = transition
    forward = np.empty((n, len(pi), 2))
    forward[0, :, 0], forward[0, :, 1] = np.log1p(-pi), np.log(pi)+values[0]
    for t in range(1, n):
        previous = forward[t-1]
        forward[t, :, 0] = np.logaddexp(previous[:, 0]+t00, previous[:, 1]+t10)
        forward[t, :, 1] = (np.logaddexp(previous[:, 0]+t01+values[t],
                                        previous[:, 1]+t11+continuation[t]) if onset else
                            values[t]+np.logaddexp(previous[:, 0]+t01, previous[:, 1]+t11))
    node_evidence = np.logaddexp(forward[-1, :, 0], forward[-1, :, 1])
    evidence = float(logsumexp(weights+node_evidence))
    persistence = .5 if prevalence == 'fixed_persistence' else float(np.exp(weights+node_evidence-evidence) @ c)
    backward, score = np.zeros((len(pi), 2)), np.empty(n)
    entry_score = np.empty(n) if onset else None
    for t in range(n-1, -1, -1):
        joints = forward[t]+backward+weights[:, None]
        score[t] = logsumexp(joints[:, 1])-logsumexp(joints[:, 0])
        if onset:
            if t == 0:
                entry_score[t] = score[t]
            else:
                previous = forward[t-1]
                entry = previous[:, 0]+t01+values[t]+backward[:, 1]
                other = np.stack((previous[:, 0]+t00+backward[:, 0],
                                  previous[:, 1]+t10+backward[:, 0],
                                  previous[:, 1]+t11+continuation[t]+backward[:, 1]))
                entry_score[t] = logsumexp(weights+entry)-logsumexp(weights+other)
        next_zero = np.logaddexp(t00+backward[:, 0], t01+values[t]+backward[:, 1])
        next_one = np.logaddexp(t10+backward[:, 0],
                               t11+(continuation[t] if onset else values[t])+backward[:, 1])
        backward = np.column_stack([next_zero, next_one])
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite persistent-state responsibility')
    result = {'score': score, 'log_evidence': evidence, 'persistence_mean': persistence}
    result.update({'entry_score': entry_score} if onset else {'cavity_log_odds': score-values})
    return result


def token_state_scores(values):
    """Separate a persistent regime from individual states with a fixed channel.

    P(z=1|h=0)=1/4 and P(z=1|h=1)=3/4. Integrate z analytically, infer h
    with the existing kernel, then recover z from the leave-one-out message.
    Context log odds are bounded by +/-log(3); own witness evidence is retained.
    Neither state is a calibrated factuality label.
    """
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite token contrasts required')
    emission = witness_emissions(values)
    low, high = np.log(.25), np.log(.75)
    zero = np.logaddexp(high, low+emission)
    contrast = np.logaddexp(low-emission, high)-np.logaddexp(high-emission, low)
    result = persistent_scores(contrast)
    cavity = result['cavity_log_odds']
    context = np.logaddexp(low, high+cavity)-np.logaddexp(high, low+cavity)
    return {'score': emission+context, 'cavity_log_odds': context,
            'log_evidence': result['log_evidence']+float(zero.sum()),
            'persistence_mean': result['persistence_mean']}


def adaptive_token_scores(values):
    """Integrate equal prior mass over shared and token-deviation structures.

    Combine unnormalized state joints using each structure's evidence. The
    uniform-posterior mixture is returned only as a frozen deletion control.
    Structure evidence is predictive compatibility, not factual correctness.
    """
    shared = persistent_scores(values, witness=True)
    token = token_state_scores(values)
    scores = np.stack([shared['score'], token['score']])
    evidence = np.array([shared['log_evidence'], token['log_evidence']])
    zero, one = -np.logaddexp(0., scores), -np.logaddexp(0., -scores)
    score = logsumexp(evidence[:, None]+one, axis=0)-logsumexp(evidence[:, None]+zero, axis=0)
    probability = float(np.exp(evidence[1]-logsumexp(evidence)))
    return {'score': score, 'witness_score': shared['score'], 'token_state_score': token['score'],
            'uniform_model_score': logsumexp(one, axis=0)-logsumexp(zero, axis=0),
            'deviation_probability': probability,
            'cavity_log_odds': score-witness_emissions(values),
            'log_evidence': float(logsumexp(evidence)-np.log(2.)),
            'persistence_mean': (1-probability)*shared['persistence_mean']
                                +probability*token['persistence_mean']}


def grouped_persistent_scores(values, answer_index, unit_index, *, witness=False, onset=False,
                              token_state=False, adaptive=False, prevalence='integrated'):
    """Keep original row order and reset the latent state at each original span."""
    _check_prevalence(prevalence, witness=witness, onset=onset)
    if prevalence != 'integrated' and (token_state or adaptive):
        raise ValueError('Fixed prevalence does not define token-state or adaptive compositions')
    if token_state and (witness or onset):
        raise ValueError('Token-state inference defines its own witness emissions')
    if adaptive and (witness or onset or token_state):
        raise ValueError('Adaptive inference defines its own structures')
    values, answer_index, unit_index = map(np.asarray, (values, answer_index, unit_index))
    if values.ndim != 1 or answer_index.shape != values.shape or unit_index.shape != values.shape:
        raise ValueError('Original token axes must match contrasts')
    starts, stops = unit_ranges(answer_index, unit_index)
    score, persistence = np.empty(len(values)), []
    entry = np.empty(len(values)) if onset else None
    extras = {key: np.empty(len(values)) for key in ['witness_score', 'token_state_score',
              'uniform_model_score', 'deviation_probability']} if adaptive else {}
    for a, b in zip(starts, stops, strict=True):
        result = (adaptive_token_scores(values[a:b]) if adaptive else
                  token_state_scores(values[a:b]) if token_state else
                  persistent_scores(values[a:b], witness=witness, onset=onset,
                                    prevalence=prevalence))
        score[a:b] = result['score']
        persistence.append(result['persistence_mean'])
        if onset:
            entry[a:b] = result['entry_score']
        for key, output in extras.items():
            output[a:b] = result[key]
    return {'score': score, 'persistence_mean': np.asarray(persistence),
            **extras, **({'entry_score': entry} if onset else {})}

