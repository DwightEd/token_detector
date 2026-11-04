"""R4 source calibration, token innovations and bounded continuity."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.research.chain import smooth_scores
from myproject.measure import unit_ranges

QUANTILES = np.linspace(0, 1, 10001)


@dataclass(frozen=True)
class Recipe:
    """Task-specific R4 parameters."""

    view: str
    tau: float
    radius: int
    innovation_weight: float
    route_weight: float


RECIPES = {
    "QA": Recipe("pair", .1, 2, .1, .25),
    "Summary": Recipe("full", .25, 4, .25, 0.),
    "Data2txt": Recipe("pair", .1, 4, .25, 0.),
}


def quantile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Historical 10001-quantile mid-CDF, preserving tied values."""
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2 * len(reference))


def source_values(delta: np.ndarray, view: str) -> np.ndarray:
    if view == "full":
        return delta[:, 1]
    if view == "pair":
        return .5 * (delta[:, 0] + delta[:, 1])
    raise ValueError(f"Unknown source view: {view}")


def tempered_prior(values: np.ndarray, starts: np.ndarray,
                   stops: np.ndarray, tau: float) -> np.ndarray:
    """Broadcast log(mean(exp(tau*delta)))/tau inside each text unit."""
    counts = stops - starts
    maxima = np.maximum.reduceat(values, starts)
    anchors = np.repeat(maxima, counts)
    totals = np.add.reduceat(np.exp(tau * (values - anchors)), starts)
    aggregate = maxima + np.log(totals / counts) / tau
    return np.repeat(aggregate, counts)


def target_window_mean(values: np.ndarray, target: np.ndarray,
                       starts: np.ndarray, stops: np.ndarray,
                       radius: int) -> np.ndarray:
    """Use raw target distance inside a unit; excluded tokens keep their gaps."""
    result = np.empty_like(values, dtype=np.float64)
    for start, stop in zip(starts, stops):
        positions = target[start:stop]
        left = np.searchsorted(positions, positions - radius, side="left")
        right = np.searchsorted(positions, positions + radius, side="right")
        for row, (first, last) in enumerate(zip(left, right), start=start):
            window = values[start + first:start + last]
            anchor = window[0]
            # Keep a fixed addition order for the calibrated reference scale.
            total = np.cumsum(window - anchor, axis=0)[-1]
            result[row] = anchor + total / len(window)
    return result


def center_units(values: np.ndarray, starts: np.ndarray,
                 stops: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for start, stop in zip(starts, stops):
        unit = values[start:stop]
        mean = unit[0] + (unit - unit[0]).mean()
        result[start:stop] = unit - mean
    return result


def raw_features(pack: dict, recipe: Recipe, innovation: str = "window") -> tuple:
    """Compute the source prior and local window used by the readout."""
    if innovation not in {"window", "point"}:
        raise ValueError("innovation must be window or point")
    starts, stops = unit_ranges(pack["answer_index"], pack["unit_index"])
    values = source_values(pack["delta"], recipe.view)
    prior = tempered_prior(values, starts, stops, recipe.tau)
    window = (values.copy() if innovation == "point" else
              target_window_mean(values, pack["target"], starts, stops, recipe.radius))
    return prior, window, starts, stops


class Detector:
    """Calibrate and score in-memory measurements for one task."""

    method = 'r4'

    def __init__(self, task: str, references: dict | None = None, *,
                 segmentation: str = "punct", innovation: str = "window",
                 calibration: str = "quantile"):
        if task not in RECIPES:
            raise ValueError(f"Unknown task: {task}. Choose from {tuple(RECIPES)}")
        self.task = task
        self.recipe = RECIPES[task]
        if segmentation not in {"punct", "dependency", "dependency_predictive", "precomputed"}:
            raise ValueError("Unknown segmentation")
        if innovation not in {"window", "point"}:
            raise ValueError("innovation must be window or point")
        if calibration not in {"quantile", "none"}:
            raise ValueError("calibration must be quantile or none")
        self.segmentation = segmentation
        self.innovation = innovation
        self.calibration = calibration
        if calibration == "quantile" and references is not None:
            if set(references) != {"prior", "window", "route"}:
                raise ValueError("References require prior, window and route arrays")
            references = {name: np.asarray(value, dtype=np.float64)
                          for name, value in references.items()}
            if any(values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                   or np.any(np.diff(values) < 0) for values in references.values()):
                raise ValueError("Reference arrays must be finite, sorted and nonempty")
        if calibration == "none" and references:
            raise ValueError("A no-FIT readout cannot use reference arrays")
        self.references = {} if calibration == "none" else references

    def calibrate(self, name: str, values: np.ndarray) -> np.ndarray:
        """Reference rank or a fixed bounded contrast; neither is a truth posterior."""
        if self.calibration == "quantile":
            return quantile_rank(values, self.references[name])
        if name == "route":
            return np.clip(.5 * (np.asarray(values) + 1.), 0., 1.)
        return np.exp(-np.logaddexp(0., -np.asarray(values)))

    def fit(self, pack: dict):
        """Fit references from the supplied unlabeled training measurements."""
        if not len(pack["target"]):
            raise ValueError("At least one training token is required")
        if self.calibration == "none":
            return self
        prior, window, _, _ = raw_features(pack, self.recipe, self.innovation)
        self.references = {
            "prior": np.quantile(prior, QUANTILES),
            "window": np.quantile(window, QUANTILES),
            "route": np.quantile(pack["route"], QUANTILES),
        }
        return self

    def score(self, pack: dict) -> np.ndarray:
        """Return final per-token risk scores without logging or saving features."""
        return self.score_components(pack)['final']

    def score_components(self, pack: dict) -> dict:
        """The actual R4 computation, exposed for diagnostic and ablation callers."""
        if self.references is None:
            raise ValueError("Fit or load reference distributions before scoring")
        if not len(pack["target"]):
            return {k: np.empty(0, dtype=np.float64) for k in
                    ['point', 'prior', 'source', 'window', 'window_rank', 'innovation', 'base', 'unary', 'final']}
        prior, window, starts, stops = raw_features(pack, self.recipe, self.innovation)
        source = self.calibrate("prior", prior)
        route = self.calibrate("route", pack["route"])
        rank = self.calibrate("window", window)
        innovation = center_units(rank, starts, stops)
        recipe = self.recipe
        base = (1 - recipe.route_weight) * source + recipe.route_weight * route
        unary = base + recipe.innovation_weight * innovation
        return {'point': source_values(pack['delta'], recipe.view),
                'prior': prior, 'source': source, 'window': window, 'window_rank': rank,
                'innovation': innovation, 'base': base, 'unary': unary,
                'final': smooth_scores(unary, pack['target'], pack['answer_index'])}

    def save(self, path: Path) -> None:
        """Save only the task and calibrated reference distributions."""
        if self.references is None:
            raise ValueError("Fit reference distributions before saving")
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, task=np.asarray(self.task),
                                segmentation=np.asarray(self.segmentation),
                                innovation=np.asarray(self.innovation),
                                calibration=np.asarray(self.calibration), **self.references)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as saved:
            settings = {key: str(saved[key]) for key in
                        ("segmentation", "innovation", "calibration") if key in saved}
            references = {key: saved[key] for key in ("prior", "window", "route") if key in saved}
            return cls(str(saved["task"]), references, **settings)
