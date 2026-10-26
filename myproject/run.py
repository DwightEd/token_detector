"""Detect an existing response and emit only its final per-token scores."""

import argparse
import json
from pathlib import Path

from .api import HallucinationDetector
from .detector import RECIPES


def main() -> None:
    command = argparse.ArgumentParser(__doc__)
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--references", type=Path)
    command.add_argument("--task", choices=tuple(RECIPES), default="QA")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--input", type=Path, help="JSON with prompt, source and response")
    command.add_argument("--prompt")
    command.add_argument("--source")
    command.add_argument("--response")
    command.add_argument("--output", type=Path)
    args = command.parse_args()
    direct = (args.prompt, args.source, args.response)
    if args.input is not None:
        if any(value is not None for value in direct):
            command.error("use either --input or --prompt/--source/--response")
        example = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        if any(value is None for value in direct):
            command.error("provide --input or all of --prompt/--source/--response")
        example = {"prompt": args.prompt, "source": args.source, "response": args.response}
    detector = HallucinationDetector.from_pretrained(
        args.model, task=args.task, references=args.references, device=args.device,
    )
    result = detector.predict(example["prompt"], example["response"], example["source"])
    output = json.dumps(result, ensure_ascii=False) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
