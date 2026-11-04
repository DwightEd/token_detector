"""Shared token detection: prepare → measure → score → evaluate; explicit legacy R4 tools."""

import argparse
from pathlib import Path

from . import data, pipeline
from experiments.research.readouts import METHODS


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(__doc__)
    stages = command.add_subparsers(dest="stage", required=True)
    verify = stages.add_parser("verify", help="compare explicitly supplied frozen artifacts on CPU")
    verify.add_argument("--packs", type=Path, required=True)
    verify.add_argument("--baseline", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
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

    fit = stages.add_parser("fit", help="legacy R4: fit unlabeled reference distributions")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--task", choices=data.TASKS, required=True)
    fit.add_argument("--output", type=Path, required=True)

    score = stages.add_parser("score", help="legacy R4: score a frozen pack using reference distributions")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--task", choices=data.TASKS, required=True)
    score.add_argument("--partition", choices=("fit", "dev", "test"), required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    evaluate = stages.add_parser("evaluate", help="join official Gold to sealed scores")
    evaluate.add_argument("--scores", type=Path, required=True)

    dataset = stages.add_parser("prepare-dataset", help="normalize native inputs and annotations")
    dataset.add_argument("--name", choices=("ragtruth", "hallurag", "claimdb", "fred"), required=True)
    dataset.add_argument("--data", type=Path, required=True)
    dataset.add_argument("--split", default="test")
    dataset.add_argument("--task")
    selection = dataset.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--ids", type=Path, help="JSON array of IDs or an object with an ids array")
    dataset.add_argument("--output", type=Path, required=True)

    predict = stages.add_parser("predict-dataset", help="score text inputs with the default shared readout")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--task", choices=data.TASKS, required=True)
    predict.add_argument("--references", type=Path)
    predict.add_argument("--method", choices=METHODS,
                         help='shared by default; explicit R4 references select legacy r4')
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

    transfer = stages.add_parser("transfer", help="cache native observations, FIT/score, then evaluate")
    transfers = transfer.add_subparsers(dest="transfer_stage", required=True)
    relation_inventory = transfers.add_parser('inventory-relation-source',
        help='seal source-only exact occurrences; no semantic verdict or detector scores')
    relation_inventory.add_argument('--prepared', nargs='+', type=Path, required=True)
    relation_inventory.add_argument('--protocol', type=Path, required=True)
    relation_inventory.add_argument('--output', type=Path, required=True)
    relation_cache = transfers.add_parser('qualify-relation-cache',
        help='qualify physical-head RoPE/GQA cache replay on CPU without model inference')
    for option in ('capture-root', 'config', 'data-manifest', 'data-root', 'output'):
        relation_cache.add_argument('--' + option, type=Path, required=True)
    relation_cache.add_argument('--prepared-root', type=Path)
    relation_selection = relation_cache.add_mutually_exclusive_group()
    relation_selection.add_argument('--ids', type=Path)
    relation_selection.add_argument('--all', action='store_true')
    relation_selection.add_argument('--limit', type=int, default=3)
    relation_cache.add_argument('--query-chunk', type=int, default=8)
    relation_cache.add_argument('--max-memory-mb', type=int, default=512)
    relation_alignment = transfers.add_parser('qualify-relation-alignment',
        help='qualify exact original source character/token axes without model inference')
    for option in ('capture-root', 'prepared', 'inventory-root', 'qualification-root',
                   'tokenizer', 'asset-manifest', 'data-root', 'output'):
        relation_alignment.add_argument('--' + option, type=Path, required=True)
    relation_alignment.add_argument('--ids', type=Path)
    streamed = transfers.add_parser('qualify-streamed-observer',
        help='qualify two original observer prompt-prefix worlds without a final head')
    for option in ('capture-root', 'model', 'asset-manifest', 'output'):
        streamed.add_argument('--' + option, type=Path, required=True)
    streamed.add_argument('--asset-verification', type=Path)
    streamed.add_argument('--asset-state', type=Path)
    streamed.add_argument('--device', default='cuda:0')
    streamed.add_argument('--max-tokens', type=int, default=256)
    streamed.add_argument('--id', default='16465')
    streamed_head = transfers.add_parser('qualify-streamed-head',
        help='qualify original observer final logits, prefix cache and Q/K/V no-op')
    for option in ('capture-root', 'model', 'asset-manifest', 'prefix-qualification', 'output'):
        streamed_head.add_argument('--' + option, type=Path, required=True)
    streamed_head.add_argument('--asset-verification', type=Path)
    streamed_head.add_argument('--asset-state', type=Path)
    streamed_head.add_argument('--device', default='cuda:0')
    streamed_head.add_argument('--max-tokens', type=int, default=256)
    streamed_head.add_argument('--id', default='16465')
    relation_paths = transfers.add_parser('observe-relation-paths',
        help='run the frozen supplied-response role/value Q/K/V intervention assay')
    for option in ('capture-root', 'model', 'asset-manifest', 'head-qualification', 'admission', 'output'):
        relation_paths.add_argument('--' + option, type=Path, required=True)
    relation_paths.add_argument('--asset-verification', type=Path)
    relation_paths.add_argument('--asset-state', type=Path)
    relation_paths.add_argument('--device', default='cuda:0')
    natural_relation = transfers.add_parser('observe-natural-relation',
        help='record matched all-layer original-response temporal evidence without risk scores')
    for option in ('capture-root', 'prepared', 'model', 'asset-manifest', 'head-qualification',
                   'alignment-root', 'source-license', 'admission', 'output'):
        natural_relation.add_argument('--' + option, type=Path, required=True)
    natural_relation.add_argument('--asset-verification', type=Path)
    natural_relation.add_argument('--asset-state', type=Path)
    natural_relation.add_argument('--device', default='cuda:0')
    scope_relation = transfers.add_parser('observe-relation-scope',
        help='run the fixed five-world original-response scope diagnostic and finite contrasts')
    for option in ('prepared', 'primitive-root', 'capture-root', 'model', 'asset-manifest',
                   'head-qualification', 'source-license', 'probability-audit', 'admission',
                   'amendment', 'resource-review', 'output'):
        scope_relation.add_argument('--' + option, type=Path, required=True)
    scope_relation.add_argument('--asset-verification', type=Path)
    scope_relation.add_argument('--asset-state', type=Path)
    scope_relation.add_argument('--device', default='cuda:0')
    scope_relation.add_argument('--reuse', type=Path)
    scope_relation.add_argument('--reuse-stage-source', type=Path)
    operator_binding = transfers.add_parser('observe-operator-binding',
        help='finite original-response source Norm+MLP readout; no annotations or AUROC')
    for option in ('protocol', 'axes', 'prepared', 'frontend-contract', 'model',
                   'asset-manifest', 'asset-verification', 'asset-state',
                   'head-qualification', 'capture-root', 'output'):
        operator_binding.add_argument('--' + option, type=Path, required=True)
    operator_binding.add_argument('--device', default='cuda:0')
    operator_binding.add_argument('--prepare-only', action='store_true')
    operator_binding.add_argument('--plan', type=Path)
    head_carrier = transfers.add_parser('observe-head-carrier-association',
        help='finite CPU marginal-preserving head/carrier assay; no model or detector risk')
    for option in ('graphs', 'samples', 'licence', 'qualification', 'protocol', 'output'):
        head_carrier.add_argument('--' + option, type=Path, required=True)
    head_carrier.add_argument('--prepare-only', action='store_true')
    head_carrier.add_argument('--plan', type=Path)
    operator_owner = transfers.add_parser('observe-operator-owner',
        help='two constructed owner controls using frozen local readout; no annotations or AUROC')
    for option in ('protocol', 'axes', 'prepared', 'frontend-contract', 'baseline', 'model',
                   'asset-manifest', 'asset-verification', 'asset-state',
                   'head-qualification', 'capture-root', 'output'):
        operator_owner.add_argument('--' + option, type=Path, required=True)
    operator_owner.add_argument('--device', default='cuda:0')
    operator_owner.add_argument('--prepare-only', action='store_true')
    operator_owner.add_argument('--plan', type=Path)
    semantic = transfers.add_parser('semantic-qualify', help='seal label-free source observations without scalar scores')
    semantic.add_argument('--inputs', type=Path, required=True)
    semantic.add_argument('--protocol', type=Path, required=True)
    semantic.add_argument('--model', type=Path, required=True)
    semantic.add_argument('--model-identity', type=Path, required=True, help='frozen model ID and model_files_sha256 JSON')
    semantic.add_argument('--admit-resources', action='store_true', help='explicit permission for gated local HF loading')
    semantic.add_argument('--output', type=Path, required=True)
    semantic_eval = transfers.add_parser('evaluate-semantic', help='review a fully sealed semantic qualification without model calls or scalar scores')
    semantic_eval.add_argument('--observations', type=Path, required=True)
    semantic_eval.add_argument('--panel', type=Path, required=True)
    semantic_eval.add_argument('--protocol', type=Path, required=True)
    semantic_eval.add_argument('--output', type=Path, required=True)
    verdict = transfers.add_parser('qualify-verdict', help='seal finite-label whole-answer source judgments without generating spans or text')
    verdict.add_argument('--inputs', type=Path, required=True)
    verdict.add_argument('--protocol', type=Path, required=True)
    verdict.add_argument('--model', type=Path, required=True)
    verdict.add_argument('--model-identity', type=Path, required=True)
    verdict.add_argument('--admit-resources', action='store_true')
    verdict.add_argument('--output', type=Path, required=True)
    verdict_eval = transfers.add_parser('evaluate-verdict', help='validate sealed whole-answer verdicts and export CPU human review')
    verdict_eval.add_argument('--observations', type=Path, required=True)
    verdict_eval.add_argument('--panel', type=Path, required=True)
    verdict_eval.add_argument('--protocol', type=Path, required=True)
    verdict_eval.add_argument('--output', type=Path, required=True)
    typed_verdict = transfers.add_parser('qualify-typed-verdict', help='run fixed crossed source-type/rule observations with exact H003 reuse')
    typed_verdict.add_argument('--inputs', type=Path, required=True)
    typed_verdict.add_argument('--protocol', type=Path, required=True)
    typed_verdict.add_argument('--donors', nargs=2, type=Path, required=True)
    typed_verdict.add_argument('--model', type=Path, required=True)
    typed_verdict.add_argument('--model-identity', type=Path, required=True)
    typed_verdict.add_argument('--admit-resources', action='store_true')
    typed_verdict.add_argument('--output', type=Path, required=True)
    typed_eval = transfers.add_parser('evaluate-typed-verdict', help='review all frozen factorial arms on CPU without token projection or arm selection')
    typed_eval.add_argument('--inputs', type=Path, required=True)
    typed_eval.add_argument('--observations', type=Path, required=True)
    typed_eval.add_argument('--natural-panel', type=Path, required=True)
    typed_eval.add_argument('--controls-panel', type=Path, required=True)
    typed_eval.add_argument('--protocol', type=Path, required=True)
    typed_eval.add_argument('--output', type=Path, required=True)
    focused = transfers.add_parser('qualify-focused-verdict', help='freeze matched whole-response and exact-occurrence unit verdicts')
    focused.add_argument('--inputs', type=Path, required=True)
    focused.add_argument('--protocol', type=Path, required=True)
    focused.add_argument('--model', type=Path, required=True)
    focused.add_argument('--model-identity', type=Path, required=True)
    focused.add_argument('--admit-resources', action='store_true')
    focused.add_argument('--output', type=Path, required=True)
    focused_eval = transfers.add_parser('evaluate-focused-verdict', help='evaluate sealed focus pilot on explicit character/native-sentence axes')
    focused_eval.add_argument('--inputs', type=Path, required=True)
    focused_eval.add_argument('--observations', type=Path, required=True)
    focused_eval.add_argument('--natural-panel', type=Path, required=True)
    focused_eval.add_argument('--controls-panel', type=Path, required=True)
    focused_eval.add_argument('--protocol', type=Path, required=True)
    focused_eval.add_argument('--output', type=Path, required=True)
    nli = transfers.add_parser('qualify-nli', help='seal frozen external NLI diagnostic on all original response units')
    nli.add_argument('--inputs', type=Path, required=True)
    nli.add_argument('--protocol', type=Path, required=True)
    nli.add_argument('--model', type=Path, required=True)
    nli.add_argument('--model-identity', type=Path, required=True)
    nli.add_argument('--admit-resources', action='store_true')
    nli.add_argument('--output', type=Path, required=True)
    local_nli = transfers.add_parser('predict-local-nli-dataset',
        help='prepare or score the frozen full Data2txt900 external text comparator')
    for option in ('inputs','inventory','protocol','model','model-identity','output'):
        local_nli.add_argument('--'+option,type=Path,required=True)
    local_nli.add_argument('--plan',type=Path)
    local_nli.add_argument('--prepare-only',action='store_true')
    local_nli.add_argument('--device',default='cuda:0')
    local_nli.add_argument('--reuse-failed',type=Path)
    local_nli.add_argument('--reuse-stage-source',type=Path)
    nli_eval = transfers.add_parser('evaluate-nli', help='validate frozen NLI pairs on CPU before exploratory native-unit evaluation')
    for option in ['inputs', 'observations', 'natural-panel', 'controls-panel', 'model', 'protocol', 'output']:
        nli_eval.add_argument('--' + option, type=Path, required=True)
    designed = transfers.add_parser('qualify-designed-focus', help='seal fixed48 balanced relation and response-burden whole/target controls')
    for option in ['inputs', 'protocol', 'model', 'model-identity', 'output']:
        designed.add_argument('--' + option, type=Path, required=True)
    designed.add_argument('--admit-resources', action='store_true')
    designed_eval = transfers.add_parser('evaluate-designed-focus', help='audit frozen designed controls before construction-label release on CPU')
    for option in ['inputs', 'design', 'construction', 'observations', 'output', 'protocol']:
        designed_eval.add_argument('--' + option, type=Path, required=True)
    reconstruction = transfers.add_parser('qualify-reconstruction', help='freeze source-only role questions and complete literal candidate plus EOS observations')
    for option in ['inputs', 'protocol', 'model', 'model-identity', 'output']:
        reconstruction.add_argument('--' + option, type=Path, required=True)
    reconstruction.add_argument('--admit-resources', action='store_true')
    masked_reconstruction = transfers.add_parser('qualify-masked-reconstruction',
        help='freeze fixed response-masked role questions and complete literal candidate plus EOS observations')
    for option in ['inputs', 'protocol', 'model', 'model-identity', 'output']:
        masked_reconstruction.add_argument('--' + option, type=Path, required=True)
    masked_reconstruction.add_argument('--admit-resources', action='store_true')
    masked_reconstruction_eval = transfers.add_parser('evaluate-masked-reconstruction',
        help='rebuild all masked-question completions on CPU before construction/control release')
    for option in ['inputs', 'design', 'construction', 'observations', 'model', 'h008-review', 'output', 'protocol']:
        masked_reconstruction_eval.add_argument('--' + option, type=Path, required=True)
    atomic_reconstruction = transfers.add_parser('qualify-atomic-reconstruction',
        help='freeze same-template single affirmative-conjunct complete-name observations')
    for option in ['inputs', 'protocol', 'model', 'model-identity', 'output']:
        atomic_reconstruction.add_argument('--' + option, type=Path, required=True)
    atomic_reconstruction.add_argument('--admit-resources', action='store_true')
    atomic_reconstruction_eval = transfers.add_parser('evaluate-atomic-reconstruction',
        help='reconstruct sealed one-conjunct observations on CPU before reference release')
    for option in ['inputs', 'design', 'construction', 'observations', 'model', 'output', 'protocol']:
        atomic_reconstruction_eval.add_argument('--' + option, type=Path, required=True)
    natural_masked_inventory = transfers.add_parser('prepare-natural-masked-inventory',
        help='freeze label-free natural span erasure artifacts without QA or scores')
    for option in ['inputs', 'h010-root', 'output', 'protocol']:
        natural_masked_inventory.add_argument('--' + option, type=Path, required=True)
    paired_prepare = transfers.add_parser('prepare-paired-scope', help='freeze matched compound/atomic complete prefix trees without weights')
    for option in ['inputs', 'protocol', 'model', 'model-identity', 'engineering', 'output']:
        paired_prepare.add_argument('--' + option, type=Path, required=True)
    paired_collect = transfers.add_parser('qualify-paired-scope', help='observe every frozen prefix once, retaining all four readouts')
    for option in ['preparation', 'model', 'model-identity', 'output']:
        paired_collect.add_argument('--' + option, type=Path, required=True)
    paired_collect.add_argument('--admit-resources', action='store_true')
    paired_evaluate = transfers.add_parser('evaluate-paired-scope', help='verify all prefix vectors and freeze four-arm scores before construction references')
    for option in ['observations', 'design', 'construction', 'output']:
        paired_evaluate.add_argument('--' + option, type=Path, required=True)
    masked_assignment = transfers.add_parser('analyze-masked-assignments',
        help='freeze fixed H009 bijection readout of all sealed H011 matrices before CPU references')
    for option in ['inputs', 'observations', 'model', 'reconstruction-review', 'output', 'protocol']:
        masked_assignment.add_argument('--' + option, type=Path, required=True)
    reconstruction_eval = transfers.add_parser('evaluate-reconstruction', help='rebuild sealed source completions on CPU before construction reference release')
    for option in ['inputs', 'design', 'construction', 'observations', 'model', 'output', 'protocol']:
        reconstruction_eval.add_argument('--' + option, type=Path, required=True)
    assignment = transfers.add_parser('analyze-reconstruction-assignments', help='freeze cached bijection readouts before reference-only CPU ablation')
    for option in ['inputs', 'observations', 'model', 'reconstruction-review', 'output', 'protocol']:
        assignment.add_argument('--' + option, type=Path, required=True)
    fusion_cache = transfers.add_parser('prepare-fusion', help='export verified historical R4/V3 caches without labels')
    fusion_cache.add_argument('--scores', type=Path, required=True)
    fusion_cache.add_argument('--native', type=Path, required=True)
    fusion_cache.add_argument('--prepared', nargs='+', type=Path, required=True,
                              help='all relocated prepared folders in native-plan group order')
    fusion_cache.add_argument('--group', type=int, required=True, help='native TEST group index')
    fusion_cache.add_argument('--task', required=True, help='declared output task name')
    fusion_cache.add_argument('--output', type=Path, required=True)
    fusion = transfers.add_parser('fuse-ranks', help='seal fixed R4/V3 batch rank fusion on CPU, without labels')
    fusion.add_argument('--r4', type=Path, required=True)
    fusion.add_argument('--witness', type=Path, required=True)
    fusion.add_argument('--protocol', type=Path, required=True)
    fusion.add_argument('--output', type=Path, required=True)
    fusion_eval = transfers.add_parser('evaluate-fusion', help='evaluate sealed R4/V3/fusion scores against released annotations')
    fusion_eval.add_argument('--scores', type=Path, required=True)
    fusion_eval.add_argument('--prepared', type=Path, required=True)
    fusion_eval.add_argument('--output', type=Path, required=True)
    fusion_eval.add_argument('--repetitions', type=int, default=1000)
    fusion_eval.add_argument('--seed', type=int, default=17)
    collect = transfers.add_parser("collect", help="measure prepared FIT/test inputs without reading labels")
    collect.add_argument("--fit", nargs="+", type=Path, default=[])
    collect.add_argument("--test", nargs="+", type=Path, required=True)
    collect.add_argument("--model", type=Path, required=True)
    collect.add_argument("--task", choices=data.TASKS, required=True)
    collect.add_argument("--protocol", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--method", choices=METHODS, default='shared')
    cached = transfers.add_parser('collect-cached', help='verify primitive cached probabilities against complete prepared test inputs')
    cached.add_argument('--prepared', type=Path, required=True)
    cached.add_argument('--cache', type=Path, required=True)
    cached.add_argument('--task', choices=data.TASKS, required=True)
    cached.add_argument('--protocol', type=Path, required=True)
    cached.add_argument('--output', type=Path, required=True)
    repairs = transfers.add_parser('probe-repairs', help='replay source-derived temporal response interventions')
    repairs.add_argument('--native', type=Path, required=True)
    repairs.add_argument('--protocol', type=Path, required=True)
    repairs.add_argument('--ids', nargs='+', required=True)
    repairs.add_argument('--output', type=Path, required=True)
    calibrate = transfers.add_parser("score", help="seal shared scores; --method r4 runs the historical FIT comparison")
    calibrate.add_argument("--native", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--method", choices=METHODS, default='shared')
    calibrate.add_argument('--compare-r4', action='store_true', help='also score bundled frozen R4 without fitting')
    calibrate.add_argument('--protocol', type=Path, help='required separate preregistration for persistent replay')
    calibrate.add_argument('--ablate-witness', action='store_true', help='three fixed no-copy/clipping/order controls')
    compare = transfers.add_parser("evaluate", help="join released labels after all predictions are sealed")
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--scores", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    qualify = transfers.add_parser('qualify-proofs', help='source-only proof compiler coverage before label join')
    qualify.add_argument('--native', type=Path, required=True)
    qualify.add_argument('--protocol', type=Path, required=True)
    qualify.add_argument('--output', type=Path, required=True)
    assertions = transfers.add_parser('qualify-assertions', help='measure frozen dependency assertion structure without labels')
    assertion_inputs = assertions.add_mutually_exclusive_group(required=True)
    assertion_inputs.add_argument('--native', type=Path)
    assertion_inputs.add_argument('--prepared', type=Path)
    assertions.add_argument('--tokenizer', type=Path, help='original observer tokenizer for prepared inputs; no LM weights loaded')
    assertions.add_argument('--control-reference', type=Path, help='reuse a sealed panel for the identical parser/role adapter')
    assertions.add_argument('--parser-model', type=Path, required=True)
    assertions.add_argument('--protocol', type=Path, required=True)
    assertions.add_argument('--output', type=Path, required=True)
    diagnose = transfers.add_parser("diagnose", help="inspect sealed errors and source table-binding conflicts")
    diagnose.add_argument("--native", type=Path, required=True)
    diagnose.add_argument("--scores", type=Path, required=True)
    diagnose.add_argument("--protocol", type=Path, required=True)
    diagnose.add_argument("--constraint-chain", action="store_true",
                          help="condition frozen chain potentials on explicit source-binding conflicts")
    diagnose.add_argument("--output", type=Path, required=True)
    pooling = transfers.add_parser("compare-pooling", help="compare fixed-prior token-preserving pooling on sealed observations")
    pooling.add_argument("--native", type=Path, required=True)
    pooling.add_argument("--scores", type=Path, required=True)
    pooling.add_argument("--protocol", type=Path, required=True)
    pooling.add_argument("--output", type=Path, required=True)
    exploration = transfers.add_parser("explore-shared", help="test a fixed lexical hierarchy against the primary shared baseline")
    exploration.add_argument("--native", type=Path, required=True)
    exploration.add_argument("--scores", type=Path, required=True)
    exploration.add_argument("--protocol", type=Path, required=True)
    exploration.add_argument("--output", type=Path, required=True)
    prior = transfers.add_parser("fit-shared-prior", help="fit the shared model prior on declared unlabeled training sources")
    prior.add_argument("--native", type=Path, required=True)
    prior.add_argument("--scores", type=Path, required=True)
    prior.add_argument("--protocol", type=Path, required=True)
    prior.add_argument("--output", type=Path, required=True)
    relational = transfers.add_parser("collect-relational", help="measure matched source/empty/relational likelihoods")
    relational.add_argument("--native", type=Path, required=True)
    relational.add_argument("--protocol", type=Path, required=True)
    relational.add_argument("--output", type=Path, required=True)
    relational.add_argument("--ids", nargs="+")
    worlds = transfers.add_parser("compare-relational", help="evaluate the shared relational null-expert model")
    worlds.add_argument("--native", type=Path, required=True)
    worlds.add_argument("--scores", type=Path, required=True)
    worlds.add_argument("--worlds", type=Path, required=True)
    worlds.add_argument("--protocol", type=Path, required=True)
    worlds.add_argument("--output", type=Path, required=True)
    constraint = transfers.add_parser("constraint-pilot", help="test source-certified arithmetic interaction states on declared cases")
    constraint.add_argument("--native", type=Path, required=True)
    constraint.add_argument("--protocol", type=Path, required=True)
    constraint.add_argument("--output", type=Path, required=True)
    constraint.add_argument("--factorial", action="store_true", help="separate source validity and consistency with a previous result")
    constraint.add_argument("--binding", action="store_true", help="use source-certified field/value/year relations instead of arithmetic identities")
    dependency = transfers.add_parser("binding-dependency", help="qualify numeric antecedent routing against source constraints")
    dependency_input = dependency.add_mutually_exclusive_group(required=True)
    dependency_input.add_argument("--native", type=Path)
    dependency_input.add_argument("--observations", type=Path, help="rescore sealed cached attention without model execution")
    dependency.add_argument("--protocol", type=Path, required=True)
    dependency.add_argument("--output", type=Path, required=True)
    dependency.add_argument("--reference", type=Path, help="stage1 pilot to contrast with a coherent source-column reassignment")
    binding_full = transfers.add_parser('binding-full', help='evaluate relation-conditioned shared inference on all original test tokens')
    binding_full.add_argument('--native', type=Path, required=True)
    binding_full.add_argument('--scores', type=Path, required=True)
    binding_full.add_argument('--protocol', type=Path, required=True)
    binding_full.add_argument('--output', type=Path, required=True)
    prevalence = transfers.add_parser('probe-prevalence', help='label-free shared-prior influence diagnostics')
    prevalence.add_argument('--native', type=Path, required=True)
    prevalence.add_argument('--protocol', type=Path, required=True)
    prevalence.add_argument('--output', type=Path, required=True)
    full = transfers.add_parser('collect-fullgap', help='collect complete prepared two-world observations')
    full.add_argument('--prepared', type=Path, required=True)
    full.add_argument('--model', type=Path, required=True)
    full.add_argument('--protocol', type=Path, required=True)
    full.add_argument('--device', default='cuda:0')
    full.add_argument('--output', type=Path, required=True)
    for name in ['score-fullgap', 'evaluate-fullgap']:
        stage = transfers.add_parser(name, help='sealed full-gap readout or sentence-derived token evaluation')
        stage.add_argument('--native', type=Path, required=True)
        stage.add_argument('--output', type=Path, required=True)
        if name == 'score-fullgap':
            stage.add_argument('--protocol', type=Path, required=True)
            from .fullgap_dataset import ALLOWED_FULL_GAP_METHODS

            stage.add_argument('--methods', nargs='+', choices=ALLOWED_FULL_GAP_METHODS,
                               help='explicit post-review comparison; existing default remains unchanged')
        else:
            stage.add_argument('--scores', type=Path, required=True)
    for name in ['probe-native', 'local-uncertainty']:
        stage = transfers.add_parser(name, help='native-evidence response or source-paired mixed-span uncertainty')
        stage.add_argument('--native', type=Path, required=True)
        stage.add_argument('--scores', type=Path, required=True)
        stage.add_argument('--protocol', type=Path, required=True)
        stage.add_argument('--output', type=Path, required=True)
    lumina = transfers.add_parser('collect-lumina', help='dense LUMINA with real actual/random/absent sources and matched S/A packet')
    for name in ('prepared', 'model', 'reference-native', 'protocol', 'output'):
        lumina.add_argument(f'--{name}', type=Path, required=True)
    lumina.add_argument('--device', default='cuda:0')
    lumina_eval = transfers.add_parser('evaluate-lumina', help='compare sealed LUMINA with aligned full7 reference')
    for name in ('collection', 'reference-scores', 'reference-evaluation', 'output'):
        lumina_eval.add_argument(f'--{name}', type=Path, required=True)
    resolution = transfers.add_parser('collect-span-resolution', help='seal label-free span mean/centering deletion controls')
    for name in ('native', 'reference-scores', 'external-collection', 'protocol', 'output'):
        resolution.add_argument(f'--{name}', type=Path, required=True)
    resolution_eval = transfers.add_parser('evaluate-span-resolution', help='evaluate sealed span deletions against existing dense seven/external ten methods')
    for name in ('collection', 'reference-evaluation', 'external-evaluation', 'output'):
        resolution_eval.add_argument(f'--{name}', type=Path, required=True)
    return command


def main() -> None:
    command = parser()
    args = command.parse_args()
    if args.stage == "transfer":
        from . import native_dataset

        if args.transfer_stage == 'semantic-qualify':
            from .semantic_qualification import HFGenerator, qualify

            if not args.admit_resources:
                command.error('semantic-qualify requires explicit --admit-resources before HF loading')
            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFGenerator(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-semantic':
            from .semantic_evaluation import evaluate

            evaluate(args.observations, args.panel, args.output, args.protocol)
        elif args.transfer_stage == 'qualify-verdict':
            if not args.admit_resources:
                command.error('qualify-verdict requires explicit --admit-resources before HF loading')
            from .verdict_qualification import qualify
            from .semantic_qualification import HFGenerator

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFGenerator(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-verdict':
            from .verdict_evaluation import evaluate

            evaluate(args.observations, args.panel, args.output, args.protocol)
        elif args.transfer_stage == 'qualify-typed-verdict':
            if not args.admit_resources:
                command.error('qualify-typed-verdict requires explicit --admit-resources before HF loading')
            from .typed_verdict_qualification import qualify
            from .semantic_qualification import HFGenerator

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFGenerator(args.model, identity, admit_resources=True), identity, args.donors)
        elif args.transfer_stage == 'evaluate-typed-verdict':
            from .typed_verdict_evaluation import evaluate

            evaluate(args.inputs, args.observations, args.natural_panel,
                     args.controls_panel, args.output, args.protocol)
        elif args.transfer_stage == 'qualify-focused-verdict':
            if not args.admit_resources:
                command.error('qualify-focused-verdict requires explicit --admit-resources before HF loading')
            from .focused_verdict_qualification import qualify
            from .semantic_qualification import HFGenerator

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFGenerator(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-focused-verdict':
            from .focused_verdict_evaluation import evaluate

            evaluate(args.inputs, args.observations, args.natural_panel,
                     args.controls_panel, args.output, args.protocol)
        elif args.transfer_stage == 'predict-local-nli-dataset':
            from .nli_dataset import run

            run(args)
        elif args.transfer_stage == 'qualify-nli':
            if not args.admit_resources:
                command.error('qualify-nli requires explicit --admit-resources before HF loading')
            from .nli_qualification import HFEntailment, qualify

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFEntailment(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-nli':
            from .nli_evaluation import evaluate

            evaluate(args.inputs, args.observations, args.natural_panel,
                     args.controls_panel, args.model, args.output, args.protocol)
        elif args.transfer_stage == 'qualify-designed-focus':
            if not args.admit_resources:
                command.error('qualify-designed-focus requires explicit --admit-resources before HF loading')
            from .designed_focus_qualification import qualify
            from .semantic_qualification import HFGenerator

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFGenerator(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-designed-focus':
            from .designed_focus_evaluation import evaluate

            evaluate(args.inputs, args.design, args.construction, args.observations,
                     args.output, args.protocol)
        elif args.transfer_stage == 'qualify-reconstruction':
            if not args.admit_resources:
                command.error('qualify-reconstruction requires explicit --admit-resources before HF loading')
            from .reconstruction_qualification import qualify
            from .answer_completion import HFAnswerCompletion

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFAnswerCompletion(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'qualify-masked-reconstruction':
            if not args.admit_resources:
                command.error('qualify-masked-reconstruction requires explicit --admit-resources before HF loading')
            from .masked_reconstruction_qualification import qualify
            from .answer_completion import HFAnswerCompletion

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFAnswerCompletion(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'prepare-natural-masked-inventory':
            from .natural_masked_inventory import prepare

            prepare(args.inputs, args.h010_root, args.output, args.protocol)
        elif args.transfer_stage == 'prepare-paired-scope':
            from .paired_scope_qualification import prepare

            prepare(args.inputs, args.protocol, args.model, data.read_json(args.model_identity), args.engineering, args.output)
        elif args.transfer_stage == 'qualify-paired-scope':
            if not args.admit_resources:
                command.error('qualify-paired-scope requires explicit --admit-resources before HF loading')
            from .paired_scope_qualification import qualify
            from .prefix_completion import HFPrefixCompletion

            identity = data.read_json(args.model_identity)
            qualify(args.preparation, args.output, HFPrefixCompletion(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-paired-scope':
            from .paired_scope_evaluation import evaluate

            evaluate(args.observations, args.design, args.construction, args.output)
        elif args.transfer_stage == 'analyze-masked-assignments':
            from .masked_assignment import analyze

            analyze(args.inputs, args.observations, args.model,
                    args.reconstruction_review, args.output, args.protocol)
        elif args.transfer_stage == 'qualify-atomic-reconstruction':
            if not args.admit_resources:
                command.error('qualify-atomic-reconstruction requires explicit --admit-resources before HF loading')
            from .atomic_reconstruction_qualification import qualify
            from .answer_completion import HFAnswerCompletion

            identity = data.read_json(args.model_identity)
            qualify(args.inputs, args.protocol, args.output,
                    HFAnswerCompletion(args.model, identity, admit_resources=True), identity)
        elif args.transfer_stage == 'evaluate-atomic-reconstruction':
            from .atomic_reconstruction_evaluation import evaluate

            evaluate(args.inputs, args.design, args.construction, args.observations,
                     args.model, args.output, args.protocol)
        elif args.transfer_stage == 'evaluate-masked-reconstruction':
            from .masked_reconstruction_evaluation import evaluate

            evaluate(args.inputs, args.design, args.construction, args.observations,
                     args.model, args.h008_review, args.output, args.protocol)
        elif args.transfer_stage == 'evaluate-reconstruction':
            from .reconstruction_evaluation import evaluate

            evaluate(args.inputs, args.design, args.construction, args.observations,
                     args.model, args.output, args.protocol)
        elif args.transfer_stage == 'analyze-reconstruction-assignments':
            from .reconstruction_assignment import analyze

            analyze(args.inputs, args.observations, args.model,
                    args.reconstruction_review, args.output, args.protocol)
        elif args.transfer_stage == 'prepare-fusion':
            from .fusion_cache import export

            export(args.scores, args.native, args.prepared, args.group, args.output, args.task)
        elif args.transfer_stage == 'fuse-ranks':
            from .rank_fusion import score

            score(args.r4, args.witness, args.protocol, args.output)
        elif args.transfer_stage == 'evaluate-fusion':
            from .fusion_evaluation import evaluate

            evaluate(args.scores, args.prepared, args.output, args.repetitions, args.seed)
        elif args.transfer_stage == 'collect-span-resolution':
            from .span_resolution import collect

            collect(args.native, args.reference_scores, args.external_collection, args.protocol, args.output)
        elif args.transfer_stage == 'evaluate-span-resolution':
            from .span_resolution import evaluate

            evaluate(args.collection, args.reference_evaluation, args.external_evaluation, args.output)
        elif args.transfer_stage == 'probe-prevalence':
            from .prevalence_probe import probe

            probe(args.native, args.output, args.protocol)
        elif args.transfer_stage == 'collect-lumina':
            from .lumina_dataset import collect

            collect(args.prepared, args.model, args.reference_native, args.protocol, args.output, args.device)
        elif args.transfer_stage == 'evaluate-lumina':
            from .lumina_dataset import evaluate

            evaluate(args.collection, args.reference_scores, args.reference_evaluation, args.output)
        elif args.transfer_stage == 'collect-fullgap':
            from .fullgap_dataset import collect

            collect(args.prepared, args.model, args.protocol, args.output, args.device)
        elif args.transfer_stage == 'score-fullgap':
            from .fullgap_dataset import score

            options = {'methods': tuple(args.methods)} if args.methods else {}
            score(args.native, args.protocol, args.output, **options)
        elif args.transfer_stage == 'evaluate-fullgap':
            from .fullgap_dataset import evaluate

            evaluate(args.native, args.scores, args.output)
        elif args.transfer_stage in {'probe-native', 'local-uncertainty'}:
            from .native_response import local_uncertainty, probe_native

            stage = probe_native if args.transfer_stage == 'probe-native' else local_uncertainty
            stage(args.native, args.scores, args.protocol, args.output)
        elif args.transfer_stage == 'observe-operator-binding':
            from .operator_binding_observation import run

            report = run(args)
            print('Operator binding finite stage: ' +
                  ('prepared; 0 model calls' if args.prepare_only else
                   str(report['actual_counts']) + '; no AUROC or efficacy claim'))
        elif args.transfer_stage == 'observe-head-carrier-association':
            from .head_carrier_observation import run

            report = run(args)
            print('Head/carrier finite CPU stage: ' + report.get('stage', 'unknown') +
                  '; no model calls, risk scores or AUROC')
        elif args.transfer_stage == 'observe-operator-owner':
            from .operator_owner_observation import run

            report = run(args)
            print('Operator owner finite control: ' +
                  ('prepared; 0 model calls' if args.prepare_only else
                   str(report['actual_counts']) + '; constructed controls, no natural AUROC'))
        elif args.transfer_stage == 'observe-relation-scope':
            from .relation_scope_observation import run

            report = run(args)
            print(f"Finite scope contrast gate={report['primary_all_three_pass']}; counts={report['actual_counts']}; no natural AUROC")
        elif args.transfer_stage == 'observe-natural-relation':
            from .natural_relation_observation import run

            report = run(args)
            print(f"Original natural relation observations: {report['actual_counts']}; no detector evaluation")
        elif args.transfer_stage == 'observe-relation-paths':
            from .relation_intervention import run

            report = run(args)
            print(f"Scoped relation-path gate={report['scoped_mechanism_gate']}; natural detection not evaluated")
        elif args.transfer_stage == 'qualify-streamed-head':
            from .streamed_head_qualification import run

            report = run(args)
            print(f"Original head/cache/no-op qualified={report['qualified']}; no detector evaluation")
        elif args.transfer_stage == 'qualify-streamed-observer':
            from .streamed_qualification import run

            report = run(args)
            print(f"Original prefix qualified={report['all_qualified']}; no detector evaluation")
        elif args.transfer_stage == 'qualify-relation-alignment':
            from .relation_alignment import run

            report = run(args)
            print(f"Original source/token alignment: {report['status_counts']}; no detector scores")
        elif args.transfer_stage == 'inventory-relation-source':
            from .relation_observation import inventory

            report = inventory(args)
            print(f"Inventoried {report['answers']} original answers / {report['sources']} sources; no scores")
        elif args.transfer_stage == 'qualify-relation-cache':
            from .relation_cache_qualification import run

            if args.ids is not None:
                selection = data.read_json(args.ids)
                args.ids = selection.get('ids') if isinstance(selection, dict) else selection
                if not isinstance(args.ids, list):
                    command.error('--ids requires a JSON array or an object with an ids array')
            elif args.all:
                args.limit = None
            report = run(args)
            print(f"Qualified replay: {report['status_counts']}; detector effectiveness not evaluated")
        elif args.transfer_stage == "collect":
            native_dataset.collect(args.output, args.fit, args.test, args.model, args.task, args.protocol, args.method)
        elif args.transfer_stage == 'probe-repairs':
            from .counterfactual_repairs import run

            run(args.native, args.output, args.protocol, args.ids)
        elif args.transfer_stage == 'collect-cached':
            native_dataset.collect_cached(args.output, args.prepared, args.cache, args.task, args.protocol)
        elif args.transfer_stage == "score":
            native_dataset.score(args.native, args.output, args.method, args.compare_r4, args.protocol, args.ablate_witness)
        elif args.transfer_stage == "evaluate":
            native_dataset.evaluate(args.native, args.scores, args.output)
        elif args.transfer_stage == 'qualify-proofs':
            native_dataset.qualify_proofs(args.native, args.output, args.protocol)
        elif args.transfer_stage == 'qualify-assertions':
            from .assertion_pilot import run

            run(args.native, args.parser_model, args.protocol, args.output,
                prepared=args.prepared, tokenizer_path=args.tokenizer,
                control_reference=args.control_reference)
        elif args.transfer_stage == 'binding-full':
            from .binding_transfer import run

            run(args.native, args.scores, args.output, args.protocol)
        elif args.transfer_stage == "binding-dependency":
            from .binding_dependency_pilot import replay, run

            if args.observations:
                if args.reference:
                    command.error('A cached replay inherits its source-world reference')
                replay(args.observations, args.output, args.protocol)
            else:
                run(args.native, args.output, args.protocol, args.reference)
        elif args.transfer_stage == "constraint-pilot":
            from .constraint_pilot import run

            run(args.native, args.output, args.protocol, args.factorial, args.binding)
        elif args.transfer_stage == "collect-relational":
            from .relational_transfer import collect

            collect(args.native, args.output, args.protocol, args.ids)
        else:
            from .transfer_diagnostics import diagnose

            if args.transfer_stage == "compare-relational":
                diagnose(args.native, args.scores, args.output, args.protocol,
                         method='relational', worlds=args.worlds)
            elif args.transfer_stage == "fit-shared-prior":
                diagnose(args.native, args.scores, args.output, args.protocol, method='prior_fit')
            elif args.transfer_stage == "explore-shared":
                diagnose(args.native, args.scores, args.output, args.protocol, method='hierarchy')
            elif args.transfer_stage == "compare-pooling":
                diagnose(args.native, args.scores, args.output, args.protocol, method='pooling')
            else:
                diagnose(args.native, args.scores, args.output, args.protocol, args.constraint_chain)
    elif args.stage == "verify":
        from .verify import verify

        verify(args.packs, args.baseline, args.output)
    elif args.stage == "export-hallurag":
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
                                 args.references, args.device, args.method)
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
        from . import measure

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
