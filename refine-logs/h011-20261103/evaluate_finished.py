"""Reserve one installed CPU review; release constructed references only inside it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
receipt = read(ROOT / 'outputs/h011-cpu-wheel-20261103/receipt.json')
assert receipt['status'] == 'passed' and receipt['test_summary'].startswith('6 passed')
target = HERE / 'evaluator_installed'
shutil.copytree(receipt['installed_target'], target, ignore=shutil.ignore_patterns('__pycache__'))
actual = {path.relative_to(target).as_posix(): sha(path)
          for package in ['myproject', 'experiments'] for path in (target / package).rglob('*')
          if path.is_file() and path.suffix in {'.py', '.npz'}}
assert len(actual) == 118 and actual == receipt['checkout_sha256']
sys.path.insert(0, str(target))
from experiments import run, masked_reconstruction_evaluation as evaluator
assert Path(run.__file__).resolve().is_relative_to(target)
module = importlib.util.spec_from_file_location('memory', ROOT / 'tools/research_memory.py')
memory = importlib.util.module_from_spec(module)
module.loader.exec_module(memory)
observation = memory.inspect_run(ROOT, read(ROOT / 'research-state/protocols/H011_masked_observation_run_spec_v1.json'))
assert observation['action'] == 'reuse' and observation['run_id'] == read(HERE / 'autostart-state.json')['run_id']
inputs = ROOT / 'outputs/h007-burden-20261103/inference/inputs.jsonl'
design = inputs.parent / 'plan.json'
construction = ROOT / 'outputs/h007-burden-20261103/construction/construction_truth.jsonl'
observations = HERE / 'masked_reconstruction_v1'
control = ROOT / 'outputs/h008-reconstruction-20261103/posthoc_review_v1'
model = ROOT.parent / 'models/Qwen2.5-3B-Instruct'
output = HERE / 'posthoc_review_v1'
protocol = ROOT / 'research-state/protocols/H011_masked_evaluation_v1.json'
write(protocol, dict(evaluator.DEFAULT_PROTOCOL,
    h008_control_cpu_seal_sha256='c62f8906d7cfb6986c4da99d29bb152d2f6ea7e85c5e02305356fb0d17dbd32b',
    h008_control_observation_seal_sha256='7558352ae7c3f5d46b4ab5b7bce72940262382806aaa4e9eaab5d81557e85a33',
    h008_observer_wheel_sha256='9c8cf5e0425da2f97ca3704d25f8d0734d5c7343953a70bcfcd5b2707a171f03',
    h011_observer_wheel_sha256='60cc608fbc408cde8f09845c71a4c978e6306c5c728585b11da1ad8b1617e5df',
    h008_cpu_module_target='separate_immutable_observer112_target',
    h011_cpu_module_target='separate_immutable_observer117_target',
    hypothesis_id='H011_RESPONSE_MASKED_RECONSTRUCTION_BRIDGE',
    notes='All three fixed burdens/48Q96raw completions/96slots and the full earlier16Q32completion control retained. No model calls or natural detection claim.'))
spec = dict(hypothesis_id='H011_RESPONSE_MASKED_RECONSTRUCTION_BRIDGE',
    intervention='independent-CPU-reconstruction-masked-question-interface-vs-source-question-control',
    stage='masked-question-reconstruction-CPU-review',
    method=dict(entrypoint='experiments.run:main transfer evaluate-masked-reconstruction', config=read(protocol)),
    code_sha256=actual,
    data_sha256={path.relative_to(ROOT).as_posix(): sha(path)
                 for path in [inputs, observations / 'seal.json', protocol, HERE / 'cpu-reader-review.md']},
    observer=dict(kind='CPU_tokenizer_only_no_model', wheel_sha256=receipt['wheel_sha256']),
    population=dict(original_cases=48, response_slots=96, queries=48, completions=96,
        logical_requests=192, retained_burdens=[0, 4, 12], control_queries=16, control_completions=32),
    evaluation=dict(reference_and_control_access_after_CPU_freeze=True, target_benchmark_gold=False,
                    all_results_retained=True, natural_task_auroc=False), seed=None)
write(ROOT / 'research-state/protocols/H011_masked_cpu_review_run_spec_v1.json', spec)
claim = memory.claim(ROOT, spec)
assert claim['action'] == 'claimed', claim
admission_path = HERE / 'evaluation-admission-v1.json'
write(admission_path, dict(observation_run_id=observation['run_id'], cpu_run_id=claim['run_id'],
    all_observation_artifacts_verified=True, runtime_files=118,
    reference_or_control_inputs_opened_by_wrapper_before_stage=False))
try:
    sys.argv = ['r4-experiment', 'transfer', 'evaluate-masked-reconstruction',
        '--inputs', str(inputs), '--design', str(design), '--construction', str(construction),
        '--observations', str(observations), '--model', str(model), '--h008-review', str(control),
        '--output', str(output), '--protocol', str(protocol)]
    run.main()
    complete = read(output / 'complete.json')
    prior_review = memory.inspect_run(ROOT, read(ROOT / 'research-state/protocols/H008_reconstruction_cpu_review_run_spec_v1.json'))
    assert prior_review['action'] == 'reuse'
    binding = HERE / 'post-review-reference-binding-v1.json'
    write(binding, dict(design_path=str(design), design_sha256=sha(design),
        truth_path=str(construction), truth_sha256=sha(construction),
        construction_seal_sha256=sha(construction.parent / 'seal.json'), review_seal_sha256=sha(output / 'seal.json'),
        observation_seal_sha256=sha(observations / 'seal.json'),
        h008_control_cpu_seal_sha256=complete['h008_control_cpu_seal_sha256'],
        h008_control_observation_seal_sha256=complete['h008_control_observation_seal_sha256'],
        obtained_after_predictions_and_evaluator_freeze=True))
    memory.finish(ROOT, claim['run_id'], 'inconclusive',
        'All96 masked-question observations reconstructed; all fixed burdens and full old source-question control retained after CPUfreeze. Exposed narrow bridge only, no natural efficacy or promotion.',
        [path for path in output.rglob('*') if path.is_file()] + [admission_path, binding,
            design, construction, construction.parent / 'seal.json'])
    summary = read(output / 'summary.json')
    print(json.dumps(dict(run_id=claim['run_id'], state=complete['state'],
        by_burden={key: {name: value[name] for name in ['queries', 'correct', 'ties', 'wrong', 'secondary_case_metrics']}
                   for key, value in summary['by_burden'].items()},
        flips={key: {name: value[name]['correctly_reversed_pairs'] for name in ['sourceflip', 'eventflip']}
               for key, value in summary['by_burden'].items()},
        secondary_case_metrics=summary['secondary_case_metrics'], costs=summary['costs']), indent=2))
except BaseException:
    import traceback
    failure = HERE / 'evaluation-failure-v1.json'
    write(failure, dict(error=traceback.format_exc()))
    retained = [failure, admission_path]
    if output.is_dir():
        retained += [path for path in output.rglob('*') if path.is_file() and not path.is_symlink()
                     and path.resolve() == path and path.resolve().is_relative_to(output.resolve())]
    memory.finish(ROOT, claim['run_id'], 'engineering_failure',
        'Retain CPU review failure; no automatic new model or reference-stage retry.', retained)
    raise
