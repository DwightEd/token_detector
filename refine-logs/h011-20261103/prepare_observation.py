"""Admit the frozen masked-question observer and pin every launch dependency."""
from copy import deepcopy
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
receipt = read(ROOT / 'outputs/h011-observer-wheel-v2-20261103/receipt.json')
assert receipt['status'] == 'passed' and receipt['test_summary'].startswith('24 passed')
target = HERE / 'installed'
shutil.copytree(receipt['installed_target'], target, ignore=shutil.ignore_patterns('__pycache__'))
actual = {path.relative_to(target).as_posix(): sha(path)
          for package in ['myproject', 'experiments'] for path in (target / package).rglob('*')
          if path.is_file() and path.suffix in {'.py', '.npz'}}
assert len(actual) == 117 and actual == receipt['checkout_sha256']
sys.path.insert(0, str(target))
from experiments import masked_reconstruction_qualification as qualifier
from experiments import datasets
from experiments.research import masked_reconstruction as kernel
inputs = ROOT / 'outputs/h007-burden-20261103/inference/inputs.jsonl'
assert sha(inputs) == '0d298932778c5ad60a43b059cc860b31679e7649350a825f5b1de72d2585cf66'
module = importlib.util.spec_from_file_location('memory', ROOT / 'tools/research_memory.py')
memory = importlib.util.module_from_spec(module)
module.loader.exec_module(memory)
prior = memory.inspect_run(ROOT, read(ROOT / 'research-state/protocols/H008_reconstruction_observation_run_spec_v1.json'))
assert prior['action'] == 'reuse'
prepared = kernel.prepare_masked_queries(list(datasets.read_samples(inputs)))
assert prepared['counts'] == dict(originals=48, sources=8, logical_queries=96,
    unique_queries=48, logical_requests=192, unique_requests=96, unknown_occurrences=0)
write(HERE / 'frontend-preflight-v1.json', prepared)
old_plan = read(ROOT / 'outputs/h008-reconstruction-20261103/reconstruction_v1/plan.json')
def event_key(messages, candidate):
    return json.dumps([messages, candidate], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
old_keys = {event_key(row['execution']['messages'], row['candidate']) for row in old_plan['executions']}
new_keys = {event_key(row['messages'], row['candidate']) for row in prepared['requests'].values()}
assert len(old_keys) == 32 and len(new_keys) == 96 and not old_keys.intersection(new_keys)
write(HERE / 'cache-admission-v1.json', dict(old_observation_run_id=prior['run_id'],
    old_unique_message_candidate_pairs=32, new_unique_message_candidate_pairs=96,
    equal_message_candidate_pairs=0, new_completion_upper_bound=96,
    basis='All message/candidate pairs differ before metadata. Exact full completion reuse is impossible when messages differ; protocol/source versions are not used to justify new observations.',
    reference_design_or_truth_opened=False))
protocol = ROOT / 'research-state/protocols/H011_masked_qualification_v1.json'
identity_path = ROOT / 'research-state/protocols/H011_masked_model_identity_v1.json'
write(protocol, dict(qualifier.DEFAULT_PROTOCOL, original_inputs_sha256=sha(inputs),
    hypothesis_id='H011_RESPONSE_MASKED_RECONSTRUCTION_BRIDGE',
    notes='Fixed exact template, full48 originals/96slots/48Q96 completions; strict toy frontend and event anchors. No natural efficacy, parameter or prompt search.'))
identity = deepcopy(read(ROOT / 'research-state/protocols/H008_reconstruction_model_identity_v1.json'))
identity['execution'].update(installed_observer_wheel_sha256=receipt['wheel_sha256'],
    CPU_module_target='separate_immutable_observer117_target')
identity['readout']['prefix'] = 'entire_response_person_literals_erased_fixed_masked_question'
write(identity_path, identity)
source = (ROOT / 'outputs/h008-reconstruction-20261103/idle_launch.py').read_text(encoding='utf-8')
for old, new in [
    ('H008_reconstruction', 'H011_masked'),
    ("HERE / 'reconstruction_v1'", "HERE / 'masked_reconstruction_v1'"),
    ("HERE / 'reconstruction.log'", "HERE / 'masked_reconstruction.log'"),
    ('qualify-reconstruction', 'qualify-masked-reconstruction'),
    ('Frozen source-only role reconstruction completions sealed. All48 originals/16queries/32complete names+EOS/96slots retained; references pending; no benchmark efficacy.',
     'Frozen response-masked reconstruction completions sealed. All48 originals/48queries/96complete names+EOS/96slots retained; references pending; no benchmark efficacy.')]:
    assert old in source, old
    source = source.replace(old, new)
(HERE / 'idle_launch.py').write_text(source, encoding='utf-8')
paths = [inputs, protocol, identity_path, HERE / 'DESIGN.md', HERE / 'collector-review.md',
         HERE / 'idle_launch.py', HERE / 'frontend-preflight-v1.json', HERE / 'cache-admission-v1.json',
         ROOT / 'outputs/h010-natural-span-20261103/NATURAL_CONTRACT_NEXT.md']
spec = dict(hypothesis_id='H011_RESPONSE_MASKED_RECONSTRUCTION_BRIDGE',
    intervention='source-defined-vs-fixed-response-person-erased-questions-with-full-burden',
    stage='masked-question-source-reconstruction-observations',
    method=dict(entrypoint='experiments.run:main transfer qualify-masked-reconstruction',
                config=read(protocol)), code_sha256=actual,
    data_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
    observer=identity,
    population=dict(original_cases=48, source_bijections=8, response_occurrences=96,
        unique_questions=48, completions=96, logical_completion_instances=192,
        burdens=[0, 4, 12], previously_exposed_toy=True),
    evaluation=dict(phase='none_observations_only', construction_truth_access_allowed=False,
        design_reference_access_allowed=False, H008_control_reference_access_allowed=False,
        primary='by_fixed_burden_closed_candidate_QA_and_all_source_event_flips',
        natural_task_auroc=False, method_promotion=False), seed=None)
write(ROOT / 'research-state/protocols/H011_masked_observation_run_spec_v1.json', spec)
print(json.dumps(dict(ready=True, runtime_files=117, kernel_counts=prepared['counts'],
    wheel_sha256=receipt['wheel_sha256'], cache_reuses=0, reference_inputs_opened=False)))
