"""Optional dataset reproduction: prepare → measure → fit → score → evaluate."""

import argparse
from pathlib import Path

from . import data, measure, pipeline


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(__doc__)
    stages = command.add_subparsers(dest="stage", required=True)
    export = stages.add_parser("export-hallurag", help="export official pickles as text/label JSONL")
    export.add_argument("--files", nargs="+", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    prepare = stages.add_parser("prepare", help="index frozen inputs or tokenize official text")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path)
    source.add_argument("--dataset", type=Path)
    prepare.add_argument("--model", type=Path)
    prepare.add_argument("--output", type=Path, required=True)

    observe = stages.add_parser("measure", help="read frozen observations; --native replays the model")
    observe.add_argument("--input", type=Path, required=True)
    observe.add_argument("--task", choices=data.TASKS, required=True)
    observe.add_argument("--split", choices=("train", "test"), required=True)
    observe.add_argument("--native", action="store_true")
    observe.add_argument("--device", default="cuda:0")
    observe.add_argument("--output", type=Path, required=True)

    fit = stages.add_parser("fit", help="fit only unlabeled FIT references")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--task", choices=data.TASKS, required=True)
    fit.add_argument("--output", type=Path, required=True)

    score = stages.add_parser("score", help="write and seal per-token R4 scores")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--task", choices=data.TASKS, required=True)
    score.add_argument("--partition", choices=("fit", "dev", "test"), required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    evaluate = stages.add_parser("evaluate", help="join official Gold to sealed scores")
    evaluate.add_argument("--scores", type=Path, required=True)

    dataset = stages.add_parser("prepare-dataset", help="normalize native inputs and annotations")
    dataset.add_argument("--name", choices=("ragtruth", "hallurag", "claimdb"), required=True)
    dataset.add_argument("--data", type=Path, required=True)
    dataset.add_argument("--split", default="test")
    dataset.add_argument("--task")
    selection = dataset.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--ids", type=Path, help="JSON array of IDs or an object with an ids array")
    dataset.add_argument("--output", type=Path, required=True)

    predict = stages.add_parser("predict-dataset", help="score text inputs with frozen R4 references")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--task", choices=data.TASKS, required=True)
    predict.add_argument("--references", type=Path)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--output", type=Path, required=True)

    native = stages.add_parser("evaluate-dataset", help="evaluate native or projected token labels")
    native.add_argument("--scores", type=Path, required=True)
    native.add_argument("--annotations", type=Path)
    native.add_argument("--bootstrap", type=int, default=2000)
    native.add_argument("--seed", type=int, default=17)
    native.add_argument("--claim-positive-labels", nargs="+")
    native.add_argument("--label-granularity", choices=("native", "token"), default="native")
    native.add_argument("--output", type=Path, required=True)
    return command


def main() -> None:
    command = parser()
    args = command.parse_args()
    if args.stage == "export-hallurag":
        from .export_hallurag import export_hallurag

        report = export_hallurag(args.files, args.output)
        print(f"exported {report['responses']} HalluRAG responses")
    elif args.stage == "prepare-dataset":
        from .datasets import prepare_dataset

        ids = None
        if args.ids is not None:
            value = data.read_json(args.ids)
            ids = value.get("ids") if isinstance(value, dict) else value
            if not isinstance(ids, list):
                command.error("--ids JSON must be an array or an object with an ids array")
        result = prepare_dataset(args.name, args.data, args.output,
                                 args.split, args.task, args.limit, ids)
        print(f"prepared {result['samples']} {result['dataset']} samples")
    elif args.stage == "predict-dataset":
        pipeline.predict_dataset(args.input, args.model, args.task, args.output,
                                 args.references, args.device)
    elif args.stage == "evaluate-dataset":
        from .evaluate import evaluate_dataset

        if args.bootstrap < 1:
            command.error("--bootstrap must be positive")
        report = evaluate_dataset(args.scores, args.annotations, args.output,
                                  args.bootstrap, args.seed, args.claim_positive_labels,
                                  args.label_granularity)
        print(f"{report['granularity']} AUROC={report['metrics']['auroc']} AP={report['metrics']['ap']}")
    elif args.stage == "prepare":
        if args.cache:
            data.prepare(args.cache, args.output)
        else:
            if args.model is None:
                command.error("prepare --dataset requires --model")
            from .inputs import prepare_official

            prepare_official(args.dataset, args.model, args.output)
    elif args.stage == "measure":
        measure.measure(args.input, args.output, args.task, args.split, args.native, args.device)
    elif args.stage == "fit":
        pipeline.fit(args.data, args.task, args.output)
    elif args.stage == "score":
        pipeline.score(args.data, args.task, args.partition, args.model, args.output)
    else:
        from .evaluate import evaluate

        report = evaluate(args.scores)
        print(f"AUROC={report['auroc']} AP={report['ap']}")


if __name__ == "__main__":
    main()
