"""Project-local research commands, calling reusable methods with explicit I/O."""

import argparse
from pathlib import Path

import numpy as np

from experiments.data import file_hash, read_json, write_json
from experiments.provenance import completed_identity, research_identity


def parser():
    command = argparse.ArgumentParser(__doc__)
    stages = command.add_subparsers(dest='stage', required=True)
    segment = stages.add_parser('segment', help='unlabeled graph segmentation, not a detector')
    segment.add_argument('--input', type=Path, required=True,
                         help='NPZ: history[n,n], optional allowed_cuts[n+1]')
    segment.add_argument('--horizons', nargs='+', type=int, choices=[1, 2, 4, 8], default=[1])
    segment.add_argument('--project', type=Path, default=Path.cwd())
    segment.add_argument('--output', type=Path, required=True)
    state = stages.add_parser('state', help='fit and transfer conditional span interactions')
    state.add_argument('--input', type=Path, required=True,
                       help='NPZ: fit and hold, both [2,n,n] signed causal masses')
    state.add_argument('--cuts', type=Path, required=True, help='JSON integer endpoint array')
    state.add_argument('--model', choices=['M1', 'M2', 'M3'], default='M1')
    state.add_argument('--project', type=Path, default=Path.cwd())
    state.add_argument('--output', type=Path, required=True)
    return command


def output_path(path, project):
    """Keep new research runs in the project's outputs tree."""
    project = project.resolve()
    path = path.resolve() if path.is_absolute() else (project / path).resolve()
    base = (project / 'outputs').resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Research output must be inside <project>/outputs/<run-id>/')
    return path


def segment(args):
    from .spans import constrained_partition, contexts

    with np.load(args.input, allow_pickle=False) as data:
        history = data['history']
        allowed = data['allowed_cuts'] if 'allowed_cuts' in data else np.ones(len(history) + 1, bool)
    if allowed.shape != (len(history) + 1,) or allowed.dtype.kind != 'b':
        raise ValueError('allowed_cuts must be a boolean mask of length n+1')
    pi, views, isolated = contexts(history)
    arrays = {'pi': pi}
    records = []
    for horizon in args.horizons:
        context, flow = views[horizon]
        cuts = constrained_partition(flow, np.flatnonzero(allowed))
        means = np.stack([np.average(context[a:b], axis=0, weights=pi[a:b])
                          for a, b in zip(cuts[:-1], cuts[1:])])
        arrays.update({f'context_{horizon}': context, f'cuts_{horizon}': cuts,
                       f'state_means_{horizon}': means})
        records.append({'horizon': horizon, 'cuts': cuts.tolist()})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'states.npz', **arrays)
    write_json(args.output / 'result.json', {
        'scope': 'unlabeled dependency spans; no correctness predictions',
        'input_sha256': file_hash(args.input), 'isolated_nodes': isolated, 'results': records,
        'state_file_sha256': file_hash(args.output / 'states.npz'),
        'provenance': completed_identity(args.provenance),
    })


def main():
    args = parser().parse_args()
    args.output = output_path(args.output, args.project)
    args.provenance = research_identity()
    if args.stage == 'segment':
        segment(args)
    elif args.stage == 'state':
        from .state import HistoryView, SpanInteraction

        if args.output.exists():
            raise FileExistsError(args.output)
        with np.load(args.input, allow_pickle=False) as data:
            model = SpanInteraction(HistoryView(data['fit']), HistoryView(data['hold']))
        result = model.partition(read_json(args.cuts), args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {'scope': 'conditional dependency fit; not hallucination scores',
                                'input_sha256': file_hash(args.input),
                                'cuts_sha256': file_hash(args.cuts),
                                'provenance': completed_identity(args.provenance),
                                'model': args.model, **result})
        print(f"held CE gain={result['hold_delta']}")


if __name__ == '__main__':
    main()
