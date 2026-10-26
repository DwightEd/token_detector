"""Direct text interface for the R4 detector."""

import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .detector import RECIPES, Detector
from .inputs import encode_response, encode_source
from .measure import answer_rows


class HallucinationDetector:
    """Measure existing responses and return their final per-token R4 scores."""

    def __init__(self, model, tokenizer, task: str = "QA", readout: Detector | None = None):
        if task not in RECIPES:
            raise ValueError(f"task must be one of {tuple(RECIPES)}")
        if readout is not None and readout.task != task:
            raise ValueError("Reference task differs from the requested task")
        self.model = model
        self.tokenizer = tokenizer
        self.readout = readout if readout is not None else Detector(task)
        self._observer = None

    @classmethod
    def from_pretrained(cls, model_path: str | Path, task: str = "QA",
                        references: str | Path | None = None, device: str = "cuda:0"):
        """Load the local observation model and the task's fitted references."""
        if task not in RECIPES:
            raise ValueError(f"task must be one of {tuple(RECIPES)}")
        reference_path = (Path(references) if references is not None
                          else Path(__file__).with_name("references") / f"{task}.npz")
        readout = Detector.load(reference_path)
        if readout.task != task:
            raise ValueError("Reference task differs from the requested task")
        from huggingface_hub.utils.tqdm import progress_bar_states
        from transformers.utils import logging

        verbosity = logging.get_verbosity()
        progress_enabled = logging.is_progress_bar_enabled()
        hub_progress = dict(progress_bar_states)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"(transformers|huggingface_hub)(\..*)?")
            try:
                logging.set_verbosity_error()
                logging.disable_progress_bar()
                from transformers import AutoTokenizer

                from .model import load_model

                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, local_files_only=True,
                )
                model = load_model(model_path, device)
            finally:
                logging.set_verbosity(verbosity)
                if progress_enabled:
                    logging.enable_progress_bar()
                else:
                    logging.disable_progress_bar()
                progress_bar_states.clear()
                progress_bar_states.update(hub_progress)
        return cls(model, tokenizer, task, readout)

    def _measure(self, prompt: str, response: str, source: str,
                 answer_index: int = 0) -> tuple[dict, dict]:
        encoded_source = encode_source(self.tokenizer, prompt, source)
        encoded_response = encode_response(self.tokenizer, response)
        if not encoded_response["answer_ids"]:
            raise ValueError("response must contain at least one token")
        if self._observer is None:
            from .observer import Observer

            self._observer = Observer(self.model)
        observed = self._observer.measure(encoded_source, encoded_response)
        return answer_rows(observed, encoded_response, answer_index), encoded_response

    def fit(self, examples: Iterable[Mapping[str, str]]):
        """Fit references from unlabeled examples with prompt/response/source strings."""
        batches = [self._measure(example["prompt"], example["response"], example["source"],
                                 index)[0]
                   for index, example in enumerate(examples)]
        if not batches:
            raise ValueError("fit requires at least one example")
        pack = {name: np.concatenate([batch[name] for batch in batches])
                for name in batches[0]}
        self.readout.fit(pack)
        return self

    def save(self, path: str | Path) -> None:
        """Save only fitted reference arrays for reuse with the same observation model."""
        self.readout.save(Path(path))

    def predict(self, prompt: str, response: str, source: str) -> list[dict]:
        """Return token_id/text/start/end/score; offsets index response characters."""
        if self.readout.references is None:
            raise ValueError("Fit reference examples or load references before predict")
        if not response:
            return []
        pack, encoded = self._measure(prompt, response, source)
        scores = self.readout.score(pack)
        result = []
        for target, token_id, score in zip(pack["target"], pack["token_id"], scores):
            start, end = map(int, encoded["offsets"][target])
            result.append({"token_id": int(token_id), "text": response[start:end],
                           "start": start, "end": end, "score": float(score)})
        return result
