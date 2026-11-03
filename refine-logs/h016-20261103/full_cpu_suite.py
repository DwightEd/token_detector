"""One complete outside-checkout CPU check with an explicit 180 s handoff bound."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
receipt = json.loads((ROOT / 'outputs/h016-wheel-20261103/receipt.json').read_text(encoding='utf-8'))
assert receipt['status'] == 'passed'
assert not (HERE / 'full-cpu-suite.json').exists()
environment = os.environ.copy()
environment.update(PYTHONPATH=receipt['installed_target'], CUDA_VISIBLE_DEVICES='-1',
    PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONOPTIMIZE='0',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
argv = [str(ROOT / '.venv/Scripts/python.exe'), '-m', 'pytest', '--import-mode=importlib',
    str(ROOT / 'tests'), '-q', '-p', 'no:cacheprovider', '--color=no']
started = time.monotonic(); status = 'incomplete'; child = None
record = dict(started_utc=datetime.now(timezone.utc).isoformat(), argv=argv, cwd=receipt['working_directory'],
    wheel_sha256=receipt['wheel_sha256'], installed_target=receipt['installed_target'], wall_limit_seconds=180)
try:
    with (HERE / 'full-cpu-suite.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=receipt['working_directory'], env=environment,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        record['pid'] = child.pid
        child.wait(timeout=180)
        status = 'passed' if child.returncode == 0 else 'failed'
        record['exit_code'] = child.returncode
except subprocess.TimeoutExpired:
    record['reason'] = 'explicit_180_second_handoff_wall_limit_full_suite_incomplete'
finally:
    if child is not None and child.poll() is None:
        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True,
            timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        child.wait(timeout=30)
        record['owned_child_tree_terminated'] = True
    record.update(status=status, elapsed_seconds=time.monotonic() - started,
        log_sha256=hashlib.sha256((HERE / 'full-cpu-suite.log').read_bytes()).hexdigest())
    with (HERE / 'full-cpu-suite.json').open('x', encoding='utf-8') as stream: json.dump(record, stream, indent=2)
print(json.dumps(record))
