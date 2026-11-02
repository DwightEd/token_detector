"""Admit exact completed source-QA observations and the installed CPU reader."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
receipt=read(ROOT/'outputs/h008-evaluation-wheel-20261103/receipt.json');assert receipt['status']=='passed'
target=HERE/'evaluator_installed';shutil.copytree(receipt['installed_target'],target,ignore=shutil.ignore_patterns('__pycache__'))
actual={p.relative_to(target).as_posix():sha(p) for package in ['myproject','experiments'] for p in (target/package).rglob('*')
        if p.is_file() and p.suffix in {'.py','.npz'}}
assert len(actual)==113 and actual==receipt['checkout_sha256']
sys.path.insert(0,str(target))
from experiments import run,reconstruction_evaluation as evaluator
assert Path(run.__file__).resolve().is_relative_to(target)
module=importlib.util.spec_from_file_location('memory',ROOT/'tools/research_memory.py')
memory=importlib.util.module_from_spec(module);module.loader.exec_module(memory)
admission=memory.inspect_run(ROOT,read(ROOT/'research-state/protocols/H008_reconstruction_observation_run_spec_v1.json'))
assert admission['action']=='reuse' and admission['run_id']==read(HERE/'autostart-state.json')['run_id'],admission
inputs=ROOT/'outputs/h007-burden-20261103/inference/inputs.jsonl';design=inputs.parent/'plan.json'
construction=ROOT/'outputs/h007-burden-20261103/construction/construction_truth.jsonl'
observations=HERE/'reconstruction_v1';model_path=ROOT.parent/'models/Qwen2.5-3B-Instruct';output=HERE/'posthoc_review_v1'
protocol=ROOT/'research-state/protocols/H008_reconstruction_evaluation_v1.json'
write(protocol,dict(evaluator.DEFAULT_PROTOCOL,hypothesis_id='H008_SOURCE_ONLY_EXACT_SLOT_RECONSTRUCTION',
    notes='All32 complete candidate+EOS observations reconstructed before reference design/truth release. Main16 closed-candidate QA and8sourceflip/8eventflip raw log-odds qualification. No fitted calibration, length correction, target-label selection or natural-task efficacy.'))
spec=dict(hypothesis_id='H008_SOURCE_ONLY_EXACT_SLOT_RECONSTRUCTION',
    intervention='frozen-source-only-completion-CPU-reconstruction-and-role-reference-evaluation',
    stage='source-reconstruction-cpu-review',
    method=dict(entrypoint='experiments.run:main transfer evaluate-reconstruction',config=read(protocol)),
    code_sha256=actual,data_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [inputs,observations/'seal.json',protocol]},
    observer=dict(kind='CPU_reader_tokenizer_only_no_model',wheel_sha256=receipt['wheel_sha256']),
    population=dict(original_cases=48,unique_queries=16,complete_candidates=32,sourceflip_pairs=8,eventflip_pairs=8),
    evaluation=dict(reference_design_and_truth_after_CPU_freeze=True,target_benchmark_gold=False,natural_task_auroc=False),seed=None)
write(ROOT/'research-state/protocols/H008_reconstruction_cpu_review_run_spec_v1.json',spec)
claim=memory.claim(ROOT,spec);assert claim['action']=='claimed',claim
write(HERE/'evaluation-admission-v1.json',dict(observation_run_id=admission['run_id'],cpu_run_id=claim['run_id'],
    all_registered_observation_artifacts_validated=True,runtime_files=113,reference_inputs_opened_by_wrapper=False))
try:
    sys.argv=['r4-experiment','transfer','evaluate-reconstruction','--inputs',str(inputs),'--design',str(design),
        '--construction',str(construction),'--observations',str(observations),'--model',str(model_path),
        '--output',str(output),'--protocol',str(protocol)]
    run.main()
    complete=read(output/'complete.json')
    binding=HERE/'post-review-reference-binding-v1.json'
    write(binding,dict(scope='Reference snapshot immediately after validated sealed CPU release',
        design_path=str(design),design_sha256=sha(design),truth_path=str(construction),truth_sha256=sha(construction),
        construction_seal_sha256=sha(construction.parent/'seal.json'),review_seal_sha256=sha(output/'seal.json'),
        obtained_after_predictions_and_evaluator_freeze=True))
    memory.finish(ROOT,claim['run_id'],'inconclusive',
        'Complete32 source-only literal name+EOS observations reconstructed and references released after CPUfreeze. Closed toy role qualification only, no natural task efficacy or method promotion.',
        [p for p in output.rglob('*') if p.is_file()]+[HERE/'evaluation-admission-v1.json',binding,
            design,construction,construction.parent/'seal.json'])
    print(json.dumps(dict(status=complete['state'],output=str(output),summary_keys=list(read(output/'summary.json')))))
except BaseException:
    import traceback
    failure=HERE/'evaluation-failure-v1.json';write(failure,dict(error=traceback.format_exc()))
    retained=[failure]
    if output.is_dir():
        retained += [p for p in output.rglob('*') if p.is_file() and not p.is_symlink()
            and p.resolve()==p and p.resolve().is_relative_to(output.resolve())]
    memory.finish(ROOT,claim['run_id'],'engineering_failure',
        'CPU reader failure; retain safely admitted artifacts. No automatic source-QA model retry.',retained)
    raise
