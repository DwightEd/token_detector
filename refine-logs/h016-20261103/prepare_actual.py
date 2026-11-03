"""Invoke the integrated tokenizer-only stage from the immutable H016 wheel."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
receipt = json.loads((ROOT / 'outputs/h016-wheel-20261103/receipt.json').read_text(encoding='utf-8'))
assert receipt['status'] == 'passed' and receipt['installed_checks'] == 70
target = Path(receipt['installed_target'])
assert all(hashlib.sha256((target / n).read_bytes()).hexdigest() == h
    for n, h in receipt['installed_files_sha256'].items())
identity = json.loads((ROOT / 'outputs/h015-prefix-trie-20261103/engineering_v1/model-identity.json').read_text(encoding='utf-8'))
identity['execution'].update(installed_target=str(target), installed_observer_wheel_sha256=receipt['wheel_sha256'],
    core_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    prefix_only_engineering=False, four_arm_paired_scope=True)
protocol_script = r"""import sys,json,pathlib,hashlib
from experiments.paired_scope_qualification import DEFAULT_PROTOCOL
from experiments.paired_scope_evaluation import EVALUATION_CONTRACT
root=pathlib.Path(sys.argv[1]);here=root/'outputs/h016-paired-scope-20261103'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
value=dict(DEFAULT_PROTOCOL,original_inputs_sha256=sha(root/'outputs/h007-burden-20261103/inference/inputs.jsonl'),engineering_seal_sha256=sha(root/'outputs/h015-prefix-trie-20261103/engineering_v1/seal.json'))
with (here/'protocol-v1.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
with (here/'evaluation-protocol-v1.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(EVALUATION_CONTRACT,f,ensure_ascii=False,indent=2);f.write('\n')
"""
identity_path = HERE / 'model-identity-v1.json'
if identity_path.exists():
    assert json.loads(identity_path.read_text(encoding='utf-8')) == identity
else:
    with identity_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(identity, stream, ensure_ascii=False, indent=2); stream.write('\n')
environment = os.environ.copy()
environment.update(PYTHONPATH=str(target), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
python = 'D:/projects/research/envs/token-detector-cu128/Scripts/python.exe'
working = receipt['working_directory']
subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-c', protocol_script, str(ROOT)], cwd=working, env=environment, check=True, timeout=60)
argv = [python, '-m', 'experiments.run', 'transfer', 'prepare-paired-scope',
    '--inputs', str(ROOT / 'outputs/h007-burden-20261103/inference/inputs.jsonl'),
    '--protocol', str(HERE / 'protocol-v1.json'), '--model', 'D:/projects/research/models/Qwen2.5-3B-Instruct',
    '--model-identity', str(HERE / 'model-identity-v1.json'),
    '--engineering', str(ROOT / 'outputs/h015-prefix-trie-20261103/engineering_v1'),
    '--output', str(HERE / 'preparation_v1')]
with (HERE / 'prepare.stdout.log').open('x', encoding='utf-8') as log:
    process = subprocess.run(argv, cwd=working, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300)
record = dict(argv=argv, cwd=working, exit_code=process.returncode, installed_target=str(target),
    wheel_sha256=receipt['wheel_sha256'], model_calls=0, cuda_calls=0)
with (HERE / 'preparation-command.json').open('x', encoding='utf-8') as stream: json.dump(record, stream, indent=2)
assert process.returncode == 0
manifest = json.loads((HERE / 'preparation_v1/manifest.json').read_text(encoding='utf-8'))
print(json.dumps(manifest['budget'], indent=2))
