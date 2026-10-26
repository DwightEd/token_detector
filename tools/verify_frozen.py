"""Compare installed R4 scores with external, frozen historical artifacts on CPU."""

import argparse
from pathlib import Path

import numpy as np

import myproject
from myproject.detector import Detector
from myproject_experiment import data


def assert_bitexact(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
        raise AssertionError("Array dtype or bytes differ from the frozen baseline")


def verify(packs: Path, baseline: Path, output: Path) -> None:
    """Check all original token axes, bundled references and DEV/TEST scores.

    The historical packs use context/observations fields. No labels are read;
    this is numerical migration verification, not a new benchmark evaluation.
    """
    if output.exists():
        raise FileExistsError(output)
    selection = data.read_json(baseline / "SELECTED.json")["selected"]
    references = Path(myproject.__file__).with_name("references")
    report = {"packs": str(packs.resolve()), "baseline": str(baseline.resolve()),
              "module": str(Path(myproject.__file__).resolve()), "tasks": {}}
    for task in data.TASKS:
        reference = references / f"{task}.npz"
        detector = Detector.load(reference)
        recipe = detector.recipe
        keys = {"prior": f"tempered_prior_{recipe.view}_tau{recipe.tau:g}_quantiles",
                "window": f"window_{recipe.view}_r{recipe.radius}_quantiles",
                "route": "route_quantiles"}
        with np.load(baseline / task / "REFERENCES.npz", allow_pickle=False) as old:
            for key, old_key in keys.items():
                assert_bitexact(detector.references[key], old[old_key])
        results = {"reference_sha256": data.file_hash(reference), "partitions": {}}
        for partition, split in (("dev", "train"), ("test", "test")):
            path = packs / f"{task}_{split}.npz"
            with np.load(path, allow_pickle=False) as archive:
                pack = {key: archive[key] for key in
                        ("target", "token_id", "answer_index", "unit_index", "development")}
                pack.update(delta=archive["context"][:, :2] + archive["observations"][:, :2],
                            route=archive["observations"][:, 7])
            pack = data.select_rows(pack, partition)
            score = detector.score(pack)
            saved = baseline / task / f"{partition.upper()}_SCORES.npz"
            with np.load(saved, allow_pickle=False) as old:
                for key in ("target", "token_id", "answer_index"):
                    assert_bitexact(pack[key], old[key])
                assert_bitexact(score, old[selection[task]])
            results["partitions"][partition] = {
                "tokens": len(score), "bitexact": True,
                "pack_sha256": data.file_hash(path),
                "baseline_sha256": data.file_hash(saved),
            }
        report["tasks"][task] = results
        print(f"{task}: references and DEV/TEST token axes and scores are bit-exact", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_json(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--packs", type=Path, required=True,
                        help="historical directory containing <task>_train/test.npz")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="historical R4 directory containing SELECTED.json")
    parser.add_argument("--output", type=Path, required=True, help="new verification JSON")
    args = parser.parse_args()
    verify(args.packs, args.baseline, args.output)


if __name__ == "__main__":
    main()
