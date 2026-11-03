"""Freeze tokenizer budget and reserve the new, reviewed instrument intervention."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
receipt_path = ROOT / 'outputs/h016-wheel-20261103/receipt.json'
receipt = read(receipt_path); assert receipt['status'] == 'passed' and receipt['installed_checks'] == 70
manifest = read(HERE / 'preparation_v1/manifest.json')
assert manifest['model_calls_during_preparation'] == manifest['cuda_calls_during_preparation'] == 0
assert manifest['budget']['reused_nodes'] == 0 and manifest['budget']['maximum_context_tokens'] <= 512
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
assert commit == manifest['model_identity']['execution']['core_commit']
delta = HERE / 'FRESH_CODE_REVIEW_DELTA.md'
assert delta.is_file()
spec = dict(schema='paired-scope-one-attempt-v1', preparation=str(HERE / 'preparation_v1'),
    preparation_seal_sha256=sha(HERE / 'preparation_v1/seal.json'),
    package_receipt=str(receipt_path), package_receipt_sha256=sha(receipt_path),
    core_commit=commit, python='D:/projects/research/envs/token-detector-cu128/Scripts/python.exe',
    output=str(HERE / 'observation_v1'), runner_sha256=sha(HERE / 'run_once.py'),
    evaluation_protocol_sha256=sha(HERE / 'evaluation-protocol-v1.json'), budget=manifest['budget'],
    core_review_sha256=sha(HERE / 'FRESH_CODE_REVIEW.md'), delta_review_sha256=sha(delta),
    one_attempt=True, automatic_retry=False, max_observation_wall_seconds=1800,
    resource_admission=dict(minimum_ram_bytes=3 * 1024**3, minimum_free_gpu_bytes=3584 * 1024**2))
with (HERE / 'run-spec-v1.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(spec, stream, indent=2); stream.write('\n')
code_names = list(receipt['reviewed_critical_sha256']) + [str(p.relative_to(ROOT)).replace('\\', '/') for p in
    (HERE / 'run_once.py', HERE / 'launch_once.py', HERE / 'freeze_run.py')]
registry = dict(hypothesis_id='H016_PAIRED_SCOPE_SHARED_PREFIX', intervention='same_instrument_scope_by_readout_complete_grid',
    stage='observation', method=dict(entrypoint='r4-experiment transfer qualify-paired-scope',
        config=dict(protocol=manifest['protocol'], budget=manifest['budget'], readouts=['independent_rows_max_slot', 'bijection_joint_assignment_mass'],
            runner_sha256=spec['runner_sha256'], wheel_sha256=receipt['wheel_sha256'], core_commit=commit)),
    code_sha256={name: sha(ROOT / name) for name in code_names},
    data_sha256=dict(original_inputs=manifest['original_inputs_sha256'], preparation_seal=spec['preparation_seal_sha256'],
        engineering_seal=manifest['engineering']['snapshot']['seal_sha256'], run_spec=sha(HERE / 'run-spec-v1.json'),
        evaluation_protocol=spec['evaluation_protocol_sha256']), observer=manifest['model_identity'],
    population=dict(all48_input_rows_retained_as_provenance=True, primary_originals=16, unique_sources=8,
        unique_queries_per_scope=16, all_four_arms=True, previously_exposed=True),
    evaluation=dict(protocol=read(HERE / 'evaluation-protocol-v1.json'), references_after_four_arm_score_freeze=True,
        natural_task_performance_claim=False), seed=None)
with (HERE / 'registry-spec-v1.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(registry, stream, indent=2); stream.write('\n')
module_spec = importlib.util.spec_from_file_location('_research_registry', ROOT / 'tools/research_memory.py')
memory = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(memory)
reservation = memory.claim(ROOT, registry)
assert reservation['action'] == 'claimed', reservation
with (HERE / 'reservation-v1.json').open('x', encoding='utf-8') as stream: json.dump(reservation, stream, indent=2)
print(json.dumps(reservation))
