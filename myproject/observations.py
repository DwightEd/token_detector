"""Portable token observations for reuse without another model forward pass."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class Observations:
    """Aligned numeric arrays plus the original response's text and token offsets."""

    pack: dict[str, np.ndarray]
    response: dict

    def save(self, path: str | Path) -> None:
        """Write one NPZ archive; JSON metadata avoids pickled Python objects."""
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **self.pack,
                                response=np.asarray(json.dumps(self.response, ensure_ascii=False)))

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as saved:
            pack = {key: saved[key] for key in
                    ("full_gap", "target", "token_id", "answer_index", "unit_index")}
            response = json.loads(str(saved["response"]))
        count = len(pack["full_gap"])
        if any(value.shape != (count,) for value in pack.values()):
            raise ValueError("Observation arrays must share one token axis")
        target = pack["target"]
        offsets = np.asarray(response["offsets"])
        ids = np.asarray(response["answer_ids"])
        if (offsets.shape != (len(ids), 2) or offsets.dtype.kind not in "iu"
                or np.any(offsets < 0) or np.any(offsets[:, 0] > offsets[:, 1])
                or np.any(offsets[:, 1] > len(response["text"]))):
            raise ValueError("Observation offsets must index the original response text")
        valid = ((offsets[:, 1] > offsets[:, 0])
                 & ~np.isin(ids, response["special_ids"]))
        if (target.dtype.kind not in "iu" or not np.array_equal(target, np.flatnonzero(valid))
                or not np.array_equal(pack["token_id"], ids[target])):
            raise ValueError("Observation tokens must match the original response axis")
        return cls(pack, response)
