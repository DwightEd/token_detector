from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
receipt=read(ROOT/'outputs/h008-observer-wheel-20261103/receipt.json');assert receipt['status']=='passed'
target=HERE/'installed';shutil.copytree(receipt['installed_target'],target,ignore=shutil.ignore_patterns('__pycache__'))
import sys
sys.path.insert(0,str(target))
from experiments import reconstruction_qualification as qualifier
actual={p.relative_to(target).as_posix():sha(p) for package in ['myproject','experiments'] for p in (target/package).rglob('*')
        if p.is_file() and p.suffix in {'.py','.npz'}}
assert len(actual)==112 and actual==receipt['checkout_sha256']
inputs=ROOT/'outputs/h007-burden-20261103/inference/inputs.jsonl'
assert sha(inputs)=='0d298932778c5ad60a43b059cc860b31679e7649350a825f5b1de72d2585cf66'
protocol_path=ROOT/'research-state/protocols/H008_reconstruction_qualification_v1.json'
identity_path=ROOT/'research-state/protocols/H008_reconstruction_model_identity_v1.json'
protocol=dict(qualifier.DEFAULT_PROTOCOL,original_inputs_sha256=sha(inputs),
    hypothesis_id='H008_SOURCE_ONLY_EXACT_SLOT_RECONSTRUCTION',
    notes='Previously exposed toy qualification. Questions/canonical candidates only from S; no original R value in QA prefix. Full literal name+fixed EOS joint likelihood. All48/8S/16Q/32 completions/96slots required. Not natural coverage or a new effective method.')
write(protocol_path,protocol)
identity=deepcopy(read(ROOT/'research-state/protocols/H007_burden_model_identity_v1.json'))
identity['execution'].update(installed_observer_wheel_sha256=receipt['wheel_sha256'],
    CPU_module_target='separate_immutable_observer112_target')
identity['readout']=dict(kind='full_literal_candidate_plus_fixed_eos_joint_log_probability',
    prefix='source_only_role_question_no_R_values',candidate_pool='two_source_names_canonical_order',
    use_cache=False,logits_to_keep='candidate_tokens_plus_one',output_tokens=0,
    normalizer='per_target_full_vocab_float64_logsumexp',candidate_length_normalization=False)
write(identity_path,identity)
source=(ROOT/'outputs/h007-burden-20261103/idle_launch.py').read_text(encoding='utf-8')
for old,new in [('H007_burden','H008_reconstruction'),("HERE / 'burden_v1'","HERE / 'reconstruction_v1'"),
    ("HERE / 'burden.log'","HERE / 'reconstruction.log'"),('qualify-designed-focus','qualify-reconstruction'),
    ("HERE / 'inference/inputs.jsonl'","ROOT / 'outputs/h007-burden-20261103/inference/inputs.jsonl'"),
    ('Frozen whole/first-unit response-burden controls sealed. All48 originals and96 focuses retained; construction truth evaluation pending; no benchmark efficacy.',
     'Frozen source-only role reconstruction completions sealed. All48 originals/16queries/32complete names+EOS/96slots retained; references pending; no benchmark efficacy.')]:
    assert old in source,old;source=source.replace(old,new)
(HERE/'idle_launch.py').write_text(source,encoding='utf-8')
paths=[inputs,inputs.parent/'plan.json',inputs.parent/'seal.json',protocol_path,identity_path,
    HERE/'idle_launch.py',HERE/'DESIGN.md',HERE/'collector-review.md',HERE/'backend-root-review.md']
if (HERE/'STRUCTURED_FOLLOWUP.md').is_file():paths.append(HERE/'STRUCTURED_FOLLOWUP.md')
spec=dict(hypothesis_id='H008_SOURCE_ONLY_EXACT_SLOT_RECONSTRUCTION',
    intervention='source-only-role-question-complete-two-candidate-name-plus-EOS-reconstruction',
    stage='source-reconstruction-observations',
    method=dict(entrypoint='experiments.run:main transfer qualify-reconstruction',config=protocol),
    code_sha256=actual,data_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
    observer=identity,population=dict(original_records=48,unique_sources=8,unique_questions=16,
        completion_observations=32,response_name_slots=96,population_previously_exposed=True),
    evaluation=dict(phase='none_observations_only',reference_design_access_allowed=False,
        construction_truth_access_allowed=False,target_benchmark_labels_access_allowed=False,
        primary='raw_log_odds_sourceflip_and_eventflip',natural_task_auroc=False),seed=None)
write(ROOT/'research-state/protocols/H008_reconstruction_observation_run_spec_v1.json',spec)
print(json.dumps(dict(ready=True,runtime_files=112,wheel_sha256=receipt['wheel_sha256'])))
