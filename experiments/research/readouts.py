"""Research readouts assembled from shared observation and inference kernels."""

from pathlib import Path

import numpy as np

from experiments.research.r4 import RECIPES
from experiments.research.pooling import grouped_scores
from experiments.research.sequence import grouped_persistent_scores, responsibility_logits

SEQUENCE_METHODS = ('persistent', 'witness', 'responsibility', 'conditional', 'onset',
                    'token_state', 'adaptive_token', 'proof_revision',
                    'symmetric_local', 'matched_local', 'fixed_local', 'fixed_persistence')
WITNESS_METHODS = ('witness', 'responsibility')
LOCAL_METHODS = {'symmetric_local': 'symmetric', 'matched_local': 'matched',
                 'fixed_local': 'fixed', 'fixed_persistence': 'fixed_persistence'}
SHARED_METHODS = ('shared', *SEQUENCE_METHODS)
METHODS = (*SHARED_METHODS, 'r4')


class SharedDetector:
    """Share a latent tendency within a span while retaining each token's full gap.

    Tasks share fixed priors and the same full-history source contrast.
    Scores are expert log odds, or bounded diagnostic probabilities for proof_revision;
    neither is a calibrated factuality probability.
    """

    method = 'shared'

    def __init__(self, task='QA', *, segmentation='punct', method='shared'):
        if task not in RECIPES:
            raise ValueError(f'Unknown task: {task}')
        if segmentation not in {'punct', 'dependency', 'dependency_predictive', 'precomputed'}:
            raise ValueError('Unknown segmentation')
        if method not in SHARED_METHODS:
            raise ValueError(f'Unknown shared method: {method}')
        self.task, self.segmentation = task, segmentation
        self.method = method

    def score(self, pack):
        return self.score_components(pack)['score']

    def score_components(self, pack):
        """One implementation behind text prediction, cached runs and ablations."""
        if self.method == 'proof_revision':
            from experiments.research.revision import revise_scores

            proofs = pack.get('proofs', {'components': [], 'diagnostics': {'reason': 'missing_source_context'}})
            result = revise_scores(pack['delta'][:, 1], pack['answer_index'], pack['unit_index'],
                                   proofs['components'])
            result['proof_diagnostics'] = {'compiler': proofs['diagnostics'],
                                            'revision': result.pop('components')}
            return result
        if not len(pack['target']):
            return self._score_values(np.empty(0), None)
        values = (pack['calibrated_gap'] if self.method == 'conditional'
                  else pack['delta'][:, 1])
        return self._score_values(values, (pack['answer_index'], pack['unit_index']))

    def score_full_gap(self, values, answer_index, unit_index):
        """Score actual full-history contrasts without inventing local/route fields."""
        if self.method in {'conditional', 'proof_revision'}:
            raise ValueError('This method requires observations beyond a full source gap')
        values, answers, units = map(np.asarray, (values, answer_index, unit_index))
        if (values.ndim != 1 or answers.shape != values.shape or units.shape != values.shape
                or not np.isfinite(values).all()):
            raise ValueError('Matching finite full-gap and original token axes required')
        return self._score_values(values, (answers, units))['score']

    def _score_values(self, values, axes):
        if not len(values):
            values, result = np.empty(0), {'score': np.empty(0)}
        else:
            result = (grouped_scores(values, *axes, alpha=1., beta=1.) if self.method == 'shared'
                      else grouped_persistent_scores(values, *axes,
                                                     witness=self.method in WITNESS_METHODS or self.method in LOCAL_METHODS,
                                                     onset=self.method == 'onset',
                                                     token_state=self.method == 'token_state',
                                                     adaptive=self.method == 'adaptive_token',
                                                     prevalence=LOCAL_METHODS.get(self.method, 'integrated')))
        if self.method in WITNESS_METHODS:
            result['witness_score'] = result['score']
            result['responsibility_score'] = responsibility_logits(values, result['score'])
            if self.method == 'responsibility':
                result['score'] = result['responsibility_score']
        if self.method == 'onset' and not len(values):
            result['entry_score'] = np.empty(0)
        if self.method == 'adaptive_token' and not len(values):
            result.update({key: np.empty(0) for key in ['witness_score', 'token_state_score',
                           'uniform_model_score', 'deviation_probability']})
        return result

    def save(self, path: Path):
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, method=np.asarray(self.method), task=np.asarray(self.task),
                                segmentation=np.asarray(self.segmentation))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as saved:
            method = str(saved['method'])
            if method not in SHARED_METHODS:
                raise ValueError('Not a shared detector configuration')
            return cls(str(saved['task']), segmentation=str(saved['segmentation']), method=method)
