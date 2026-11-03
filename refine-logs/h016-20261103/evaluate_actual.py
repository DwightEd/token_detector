"""CPU-only integrated replay and evaluation after the complete observation seal."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
receipt = json.loads((ROOT / 'outputs/h016-wheel-20261103/receipt.json').read_text(encoding='utf-8'))
launch = json.loads((HERE / 'launch-state-v1.json').read_text(encoding='utf-8'))
assert launch['status'] == 'completed' and launch['armed'] is False and launch['actual_forward_passes'] == 312
assert (HERE / 'observation_v1/seal.json').is_file() and not (HERE / 'evaluation_v1').exists()
assert receipt['status'] == 'passed' and all(hashlib.sha256((Path(receipt['installed_target']) / n).read_bytes()).hexdigest() == h
    for n, h in receipt['installed_files_sha256'].items())
environment = os.environ.copy()
environment.update(PYTHONPATH=receipt['installed_target'], CUDA_VISIBLE_DEVICES='-1',
    PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONOPTIMIZE='0',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
argv = [str(ROOT / '.venv/Scripts/python.exe'), '-m', 'experiments.run', 'transfer', 'evaluate-paired-scope',
    '--observations', str(HERE / 'observation_v1'),
    '--design', str(ROOT / 'outputs/h007-burden-20261103/inference/plan.json'),
    '--construction', str(ROOT / 'outputs/h007-burden-20261103/construction/construction_truth.jsonl'),
    '--output', str(HERE / 'evaluation_v1')]
with (HERE / 'evaluation.log').open('xb') as log:
    result = subprocess.run(argv, cwd=receipt['working_directory'], env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600)
with (HERE / 'evaluation-command.json').open('x', encoding='utf-8') as stream:
    json.dump(dict(argv=argv, cwd=receipt['working_directory'], exit_code=result.returncode,
        log_sha256=hashlib.sha256((HERE / 'evaluation.log').read_bytes()).hexdigest(), model_calls=0), stream, indent=2)
assert result.returncode == 0
summary = json.loads((HERE / 'evaluation_v1/summary.json').read_text(encoding='utf-8'))
print(json.dumps(summary, ensure_ascii=False, indent=2))
