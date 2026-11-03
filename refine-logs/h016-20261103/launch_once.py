"""Bounded resource admission and one owned child tree; never a model retry."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = HERE / 'launch-state-v1.json'
LOCK = HERE / 'launch-v1.lock'
STOP = HERE / 'STOP_H016'
class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
        ('total_phys', 'available_phys', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resources():
    memory = Memory(); memory.length = ctypes.sizeof(Memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)): raise OSError('RAM measurement failed')
    query = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, check=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(available_ram_bytes=memory.available_phys, free_gpu_bytes=int(query.stdout.decode().splitlines()[0]) * 1024**2)

def main():
    assert len(sys.argv) == 2 and not STATE.exists() and not STOP.exists()
    run_id = sys.argv[1]
    claim = json.loads((ROOT / 'research-state/runs' / run_id / 'claim.json').read_text(encoding='utf-8'))
    spec_path = HERE / 'run-spec-v1.json'
    spec_hash = sha(spec_path); spec = json.loads(spec_path.read_text(encoding='utf-8'))
    own = Path(__file__).relative_to(ROOT).as_posix()
    assert claim['spec']['code_sha256'][own] == sha(__file__)
    assert claim['spec']['data_sha256']['run_spec'] == spec_hash
    assert not Path(spec['output']).exists()
    lock = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(lock, str(os.getpid()).encode()); os.close(lock)
    state = dict(run_id=run_id, pid=os.getpid(), armed=True, actual_forward_passes=None,
        status='resource_admission', utc=datetime.now(timezone.utc).isoformat(), child_pid=None, automatic_retry=False)
    def save():
        tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2), encoding='utf-8'); tmp.replace(STATE)
    child = None
    try:
        save()
        for i in range(2):
            current = resources(); state['resources'] = current; save()
            assert current['available_ram_bytes'] >= 3 * 1024**3 and current['free_gpu_bytes'] >= 3584 * 1024**2, 'Resource gate not met; no launch'
            if i == 0: time.sleep(1)
        assert not STOP.exists() and sha(spec_path) == spec_hash and sha(HERE / 'run_once.py') == spec['runner_sha256']
        receipt = json.loads(Path(spec['package_receipt']).read_text(encoding='utf-8'))
        env = os.environ.copy()
        env.update(PYTHONPATH=receipt['installed_target'], PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        argv = [spec['python'], str(HERE / 'run_once.py'), str(spec_path)]
        with (HERE / 'observation.stdout.log').open('xb') as stdout, (HERE / 'observation.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(argv, cwd=receipt['working_directory'], env=env, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(status='running_once', child_pid=child.pid, argv=argv); save()
            deadline = time.monotonic() + 1800
            while child.poll() is None:
                if STOP.exists(): raise RuntimeError('User STOP requested')
                if time.monotonic() >= deadline: raise TimeoutError('1800s observation wall limit')
                time.sleep(1)
        state.update(exit_code=child.returncode, status='completed' if child.returncode == 0 else 'failed_retained_no_retry')
        if child.returncode == 0:
            result = json.loads((Path(spec['output']) / 'complete.json').read_text(encoding='utf-8'))
            state['actual_forward_passes'] = result['costs']['actual_forward_passes']
        else:
            state['actual_counts_only_in_retained_raw'] = True
    except BaseException as error:
        state.update(status='controller_stopped_no_retry', error_type=type(error).__name__, reason=str(error))
        if child is None: state['actual_forward_passes'] = 0
        raise
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True,
                timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            child.wait(timeout=30)
            state['owned_child_tree_terminated'] = True
        state.update(armed=False, finished_utc=datetime.now(timezone.utc).isoformat()); save()
    print(json.dumps(state))

if __name__ == '__main__': main()
