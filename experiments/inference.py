"""Direct text interface: shared source readout by default, explicit legacy R4."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from experiments.research.r4 import RECIPES, Detector
from myproject.inputs import encode_response, encode_source
from experiments.research.measurement import answer_rows
from myproject.measure import token_scores
from myproject.model import load_pretrained
from myproject.api import HallucinationDetector as SourceDetector
from experiments.research.readouts import METHODS, SHARED_METHODS, SharedDetector


class HallucinationDetector:
    """Measure existing responses and return per-token source-risk scores."""

    def __init__(self, model, tokenizer, task: str = "QA",
                 readout: Detector | SharedDetector | None = None, *, method: str | None = None):
        if task not in RECIPES:
            raise ValueError(f"task must be one of {tuple(RECIPES)}")
        if readout is not None and readout.task != task:
            raise ValueError("Reference task differs from the requested task")
        if readout is not None and readout.segmentation == "precomputed":
            raise ValueError("Precomputed segmentation requires measurement packs, not text inputs")
        if method is not None and method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        if readout is not None and method is not None and readout.method != method:
            raise ValueError('Readout method differs from the requested method')
        self.model = model
        self.tokenizer = tokenizer
        self.readout = readout if readout is not None else (
            Detector(task) if method == 'r4' else SharedDetector(task, method=method or 'shared'))
        self._observer = None

    @classmethod
    def from_pretrained(cls, model_path: str | Path, task: str = "QA",
                        references: str | Path | None = None, device: str = "cuda:0",
                        readout: Detector | SharedDetector | None = None,
                        method: str | None = None):
        """Load a local model; shared needs no FIT, explicit references select legacy R4."""
        if task not in RECIPES:
            raise ValueError(f"task must be one of {tuple(RECIPES)}")
        if readout is not None and references is not None:
            raise ValueError("Use either readout or references")
        if method is not None and method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        if method in SHARED_METHODS and references is not None:
            raise ValueError('Shared detection does not use R4 reference distributions')
        if readout is None:
            selected = method or ('r4' if references is not None else 'shared')
            if selected in SHARED_METHODS:
                readout = SharedDetector(task, method=selected)
            else:
                reference_path = (Path(references) if references is not None
                                  else Path(__file__).parent / "research" / "references" / f"{task}.npz")
                readout = Detector.load(reference_path)
        if method is not None and readout.method != method:
            raise ValueError('Readout method differs from the requested method')
        if readout.task != task:
            raise ValueError("Reference task differs from the requested task")
        if readout.segmentation == "precomputed":
            raise ValueError("Precomputed segmentation requires measurement packs, not text inputs")
        model, tokenizer = load_pretrained(model_path, device)
        return cls(model, tokenizer, task, readout, method=method)

    def measure(self, prompt: str, response: str, source: str,
                 answer_index: int = 0) -> tuple[dict, dict]:
        """Return aligned native observations and original token metadata, without I/O."""
        encoded_source = encode_source(self.tokenizer, prompt, source)
        encoded_response = encode_response(self.tokenizer, response)
        if not encoded_response["answer_ids"]:
            raise ValueError("response must contain at least one token")
        if self.readout.segmentation != "punct":
            from experiments.research.attention import attention_profiles
            from experiments.research.segmentation import legal_atoms, predictive_segment, segment

            graph = attention_profiles(self.model, encoded_source["prompt_with_source"],
                                       encoded_response["answer_ids"])
            atoms, _ = legal_atoms(encoded_response)
            fitted = (predictive_segment(graph["fit"], graph["hold"], graph["available"], atoms)
                      if self.readout.segmentation == "dependency_predictive" else
                      segment(graph["fit"], graph["available"], atoms))
            cuts = fitted["cuts"]
            encoded_response["units"] = [{"start": int(a), "stop": int(b)}
                                         for a, b in zip(cuts[:-1], cuts[1:])]
        if self._observer is None:
            from experiments.research.observation import Observer

            self._observer = Observer(self.model)
        observed = self._observer.measure(encoded_source, encoded_response,
                                          full_distribution=self.readout.method == 'conditional')
        return answer_rows(observed, encoded_response, answer_index), encoded_response

    def fit(self, examples: Iterable[Mapping[str, str]]):
        """Fit references from unlabeled examples with prompt/response/source strings."""
        if isinstance(self.readout, SharedDetector):
            raise ValueError('The fixed shared method needs no FIT; select method=r4 to fit R4 references')
        batches = [self.measure(example["prompt"], example["response"], example["source"],
                                 index)[0]
                   for index, example in enumerate(examples)]
        if not batches:
            raise ValueError("fit requires at least one example")
        pack = {name: np.concatenate([batch[name] for batch in batches])
                for name in batches[0]}
        self.readout.fit(pack)
        return self

    def measure_responses(self, prompt: str, responses: Iterable[str], source: str):
        """Replay supplied response variants; return raw logp matrices and encodings.

        Both source conditions use matched SDPA. This observation API does not
        score factuality or select repairs, and preserves each raw response axis.
        """
        from experiments.research.counterfactual import observe_responses

        encoded_source = encode_source(self.tokenizer, prompt, source)
        encoded = [encode_response(self.tokenizer, text) for text in responses]
        return observe_responses(self.model, encoded_source,
                                 [row['answer_ids'] for row in encoded]), encoded

    def save(self, path: str | Path) -> None:
        """Save the shared configuration or the explicit legacy R4 references."""
        self.readout.save(Path(path))

    def predict(self, prompt: str, response: str, source: str) -> list[dict]:
        """Return token_id/text/start/end/score; offsets index response characters."""
        if isinstance(self.readout, Detector) and self.readout.references is None:
            raise ValueError("Fit reference examples or load references before predict")
        if not response:
            return []
        if self.readout.method in {"shared", "persistent", "witness"} and self.readout.segmentation == "punct":
            observed = SourceDetector(self.model, self.tokenizer).extract(prompt, response, source)
            pack = observed.pack
            scores = self.readout.score_full_gap(pack["full_gap"], pack["answer_index"], pack["unit_index"])
            return token_scores(pack, observed.response, scores)
        pack, encoded = self.measure(prompt, response, source)
        if self.readout.method == 'proof_revision':
            from experiments.research.revision import proof_pack

            pack = proof_pack(pack, source, response, encoded['offsets'])
        scores = self.readout.score(pack)
        return token_scores(pack, encoded, scores)
