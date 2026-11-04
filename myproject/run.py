"""Extract reusable observations or score an existing response."""

import argparse
import json
from pathlib import Path

from .api import HallucinationDetector
from .observations import Observations


def main() -> None:
    command = argparse.ArgumentParser(__doc__)
    stages = command.add_subparsers(dest="stage", required=True)
    for name in ("extract", "predict"):
        stage = stages.add_parser(name)
        stage.add_argument("--model", type=Path, required=True)
        stage.add_argument("--device", default="cuda:0")
        stage.add_argument("--input", type=Path, required=True,
                           help="JSON with prompt, source and response")
        stage.add_argument("--output", type=Path, required=name == "extract")
    score = stages.add_parser("score")
    score.add_argument("--input", type=Path, required=True, help="Extracted observations.npz")
    score.add_argument("--output", type=Path)
    args = command.parse_args()

    if args.stage == "score":
        result = HallucinationDetector.score(Observations.load(args.input))
    else:
        example = json.loads(args.input.read_text(encoding="utf-8"))
        detector = HallucinationDetector.from_pretrained(args.model, device=args.device)
        if args.stage == "extract":
            detector.extract(**example).save(args.output)
            return
        result = detector.predict(**example)
    output = json.dumps(result, ensure_ascii=False) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
