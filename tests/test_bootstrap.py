"""Compare source bootstrap intervals with explicit sklearn resampling."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from myproject_experiment.evaluate import _cluster_intervals


class BootstrapTests(unittest.TestCase):
    def test_matches_explicit_resampling(self):
        populations = [
            ([False, True], [np.iinfo(np.int64).min, 0], [0, 0]),
            ([False, True], np.array([0, 1], dtype=np.uint64), [0, 0]),
            ([False, True], [False, True], [0, 0]),
            ([False, True, False, True], [1., 1., 2., 2.], [0, 0, 1, 1]),
            ([True, True, False, False], [2., 1., 2., 0.], [0, 0, 1, 1]),
            ([True, False, False, True, False], [-2., 0., 1., 0., 1.], [0, 0, 0, 1, 2]),
            ([True, True], [0., 1.], [0, 1]),
            ([False, False], [0., 1.], [0, 1]),
            ([True], [0.], [0]),
        ]
        for labels, scores, groups in populations:
            with self.subTest(scores=scores):
                labels, scores, groups = map(np.asarray, (labels, scores, groups))
                actual = _cluster_intervals(labels, scores, groups, 40, 17)
                _, index = np.unique(groups, return_inverse=True)
                count = int(index.max()) + 1
                reference = {"auroc": [], "ap": []}
                draws = np.random.default_rng(17).integers(0, count, (40, count))
                for draw in draws:
                    weights = np.bincount(draw, minlength=count)[index]
                    selected = np.repeat(np.arange(len(labels)), weights)
                    target, score = labels[selected], scores[selected]
                    if target.any() and not target.all():
                        reference["auroc"].append(roc_auc_score(target, score))
                    if target.any():
                        reference["ap"].append(average_precision_score(target, score))
                for metric, values in reference.items():
                    self.assertEqual(actual[metric]["defined"], len(values))
                    self.assertEqual(actual[metric]["degenerate"], 40 - len(values))
                    if values:
                        np.testing.assert_allclose(actual[metric]["interval95"],
                                                   np.quantile(values, [.025, .975]),
                                                   rtol=0, atol=1e-14)
                    else:
                        self.assertIsNone(actual[metric]["interval95"])
