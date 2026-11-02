"""One reserved, installed, cached-only H009 ablation; no neural inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')

receipt = read(ROOT / 'outputs/h009-wheel-20261103/receipt.json')
assert receipt['status'] == 'passed' and receipt['test_summary'] == '6 passed in 16.30s'
target = HERE / 'installed'
shutil.copytree(receipt['installed_target'], target, ignore=shutil.ignore_patterns('__pycache__'))
actual = {path.relative_to(target).as_posix(): sha(path)
          for package in ['myproject', 'experiments']
          for path in (target / package).rglob('*')
          if path.is_file() and path.suffix in {'.py', '.npz'}}
assert len(actual) == 115 and actual == receipt['checkout_sha256']
sys.path.insert(0, str(target))
from experiments import run, reconstruction_assignment as evaluator
assert Path(run.__file__).resolve().is_relative_to(target)
module = importlib.util.spec_from_file_location('memory', ROOT / 'tools/research_memory.py')
memory = importlib.util.module_from_spec(module)
module.loader.exec_module(memory)
observation = memory.inspect_run(ROOT, read(ROOT / 'research-state/protocols/H008_reconstruction_observation_run_spec_v1.json'))
assert observation['action'] == 'reuse'
inputs = ROOT / 'outputs/h007-burden-20261103/inference/inputs.jsonl'
observations = ROOT / 'outputs/h008-reconstruction-20261103/reconstruction_v1'
review = ROOT / 'outputs/h008-reconstruction-20261103/posthoc_review_v1'
model = ROOT.parent / 'models/Qwen2.5-3B-Instruct'
output = HERE / 'assignment_review_v1'
protocol = ROOT / 'research-state/protocols/H009_bijection_cached_ablation_v1.json'
followup = ROOT / 'outputs/h008-reconstruction-20261103/STRUCTURED_FOLLOWUP.md'
assert sha(followup) == '1d1a099c7a766c633fe5c6ade3230d960663dc8bb41a7c8dcae6e16904fca7b2'
write(protocol, dict(evaluator.DEFAULT_PROTOCOL,
    hypothesis_id='H009_CACHED_SOURCE_BIJECTION_READOUT',
    notes='Both fixed arms and both readouts retained. Predeclared additive score intervention; no language changes, weight search, target benchmark evaluation, or natural efficacy claim.'))
spec = dict(hypothesis_id='H009_CACHED_SOURCE_BIJECTION_READOUT',
    intervention='fixed-additive-nuisance-vs-two-explicit-bijection-energy-readouts',
    stage='cached-source-reconstruction-assignment-ablation',
    method=dict(entrypoint='experiments.run:main transfer analyze-reconstruction-assignments', config=read(protocol)),
    code_sha256=actual,
    data_sha256={path.relative_to(ROOT).as_posix(): sha(path)
                 for path in [inputs, observations / 'seal.json', protocol, followup]},
    observer=dict(kind='CPU_reader_tokenizer_only_no_model', wheel_sha256=receipt['wheel_sha256']),
    population=dict(source_bijections=8, unique_queries=16, retained_completions=32, original_cases=48,
                    fixed_score_arms=2, readouts_per_arm=2),
    evaluation=dict(H008_reference_read_after_scores_freeze=True, target_benchmark_gold=False,
                    natural_task_auroc=False, newly_generated_observations=0), seed=None)
spec_path = ROOT / 'research-state/protocols/H009_bijection_cached_run_spec_v1.json'
write(spec_path, spec)
claim = memory.claim(ROOT, spec)
assert claim['action'] == 'claimed', claim
admission_path = HERE / 'admission-v1.json'
write(admission_path, dict(run_id=claim['run_id'], donor_run_id=observation['run_id'],
    donor_artifacts_verified=True, runtime_files=115, reference_inputs_opened_by_wrapper=False,
    source_followup_sha256=sha(followup), new_model_calls=0))
try:
    sys.argv = ['r4-experiment', 'transfer', 'analyze-reconstruction-assignments',
        '--inputs', str(inputs), '--observations', str(observations), '--model', str(model),
        '--reconstruction-review', str(review), '--output', str(output), '--protocol', str(protocol)]
    run.main()
    complete = read(output / 'complete.json')
    prior_review = memory.inspect_run(ROOT, read(ROOT / 'research-state/protocols/H008_reconstruction_cpu_review_run_spec_v1.json'))
    assert prior_review['action'] == 'reuse'
    binding = HERE / 'post-review-reference-binding-v1.json'
    write(binding, dict(reference_seal_sha256=sha(review / 'seal.json'),
        donor_seal_sha256=sha(observations / 'seal.json'), assignment_seal_sha256=sha(output / 'seal.json'),
        prior_review_run_id=prior_review['run_id'], read_after_assignment_stage_complete=True))
    memory.finish(ROOT, claim['run_id'], 'inconclusive',
        'All fixed cached-only arms sealed. Conditional additive-nuisance invariance qualification only; no new model calls, natural efficacy, or method promotion.',
        [path for path in output.rglob('*') if path.is_file()] + [admission_path, binding])
    print(json.dumps(dict(run_id=claim['run_id'], complete=complete, summary=read(output / 'summary.json')), indent=2))
except BaseException:
    import traceback
    failure = HERE / 'failure-v1.json'
    write(failure, dict(error=traceback.format_exc()))
    retained = [failure, admission_path]
    if output.is_dir():
        retained += [path for path in output.rglob('*') if path.is_file() and not path.is_symlink()
                     and path.resolve() == path and path.resolve().is_relative_to(output.resolve())]
    memory.finish(ROOT, claim['run_id'], 'engineering_failure',
        'Retain cached-analysis failure; no automatic model or CPU retry.', retained)
    raise
