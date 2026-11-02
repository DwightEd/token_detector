"""Root-launched, bounded 7B engineering qualification; never a detector run.

Preparation only: root reviews this file before starting it with the pinned GPU
Python. It does not restart/download assets, install packages, close applications,
offload models, or touch datasets/truth. STOP_7B_READINESS stops only this process.
All waits and model work share a 45-minute deadline. Any invalid dependency or
failure defers this one attempt and retains its receipt; there is no auto retry.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT.parent / 'models/Qwen2.5-7B-Instruct-bnb-4bit'
PYTHON = ROOT.parent / 'envs/token-detector-cu128/Scripts/python.exe'
DOWNLOAD_SCRIPT = HERE / 'prepare_assets_v2.py'
DOWNLOAD = HERE / 'receipt-v2.json'
RECEIPT = HERE / 'runtime-readiness-v2.json'
STOP = HERE / 'STOP_7B_READINESS_V2'
REPO = 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'
REVISION = 'bdd404162d94997f390efbfa660eb3f21cbbc81d'
DOWNLOAD_SCRIPT_SHA = '74592d261cbd6d526ef58d4826ad4df5941635d985f2ce63774dae3fa72545b6'
# Actual v2 downloader child, not merely the Windows venv launcher PID 33404.
DOWNLOAD_PID, DOWNLOAD_CREATION = 28176, 134381184104340737
RUNTIME = {'torch': '2.8.0+cu128', 'transformers': '4.57.1',
           'bitsandbytes': '0.48.0', 'accelerate': '1.10.1'}
FILES = {
    'added_tokens.json': (605, '58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b'),
    'config.json': (1338, 'c9b481809ae2e99227ac323b77a9da0b5387a0a21f46329327132273435fa3f7'),
    'generation_config.json': (271, '424e77633729a4c63a0810e0fa32e3344b03499cd3a0b2fc626bfae19e7eb923'),
    'merges.txt': (1671853, '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'),
    'model.safetensors': (5547254622, '99b85155b7bb40344d8c3938c8ec1147b27028804bca5d2e5559cb8e6533747d'),
    'special_tokens_map.json': (614, 'b5acd1507cb3a3539e63369cb4a2b675a599cf13afe62282d371518288967797'),
    'tokenizer.json': (11421896, '9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa'),
    'tokenizer_config.json': (7362, '0553ada6e92b1a29b4431c72782c6e5e6ab675ef0b20ceb2833cd1074c7fb932'),
    'vocab.json': (2776833, 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
}
H007 = ROOT / 'outputs/h007-burden-20261103'
H007_STATE_SHA = '28c52448fc17b8bd0d2832613ce68309a754cf9de218a516346ab0a62aa91aa3'
H007_SEAL_SHA = '36118e42cb86960ba5581a48739e00a0d9a344870586013205babbc8fa94bbf1'
H007_COMPLETE_SHA = '8d9d3579bfcb908909799d397f229c52e69973855224683846a6ccbe4069fb28'
H007_RUN = '92cf2597ab7d2217583f07048001da2d715530fee62de9aa314f3c3eedcb8d88'
H007_PID = 59824
# An actual OpenProcess probe returned ERROR_INVALID_PARAMETER (absent) here.
H007_EXIT_WITNESS = '2026-11-02T19:06:07.535977+00:00'
GIB = 1024**3
PREWARM_RAM, POSTWARM_RAM = 3 * GIB, (23 * GIB + 9) // 10
GPU_FREE, ALLOCATOR_CAP = 13 * GIB // 2, 6 * GIB
MAX_SECONDS, POLL_SECONDS = 45 * 60, 30


class Deferred(RuntimeError):
    """Terminal engineering deferral; never a scientific failure or a retry."""


def require(value, reason):
    if not value:
        raise Deferred(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    # The downloader updates its receipt in place. Re-read a torn JSON snapshot;
    # this never retries a model call or relaxes any payload identity check.
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            if Path(path) != DOWNLOAD or attempt == 2:
                raise
            time.sleep(.1)


def ram():
    class Status(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (name, ctypes.c_uint64) for name in ('total', 'available', 'total_page',
                'available_page', 'total_virtual', 'available_virtual', 'extra')]
    status = Status(); status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Status)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.available)


def process(pid):
    """PID plus exact OS creation identity; never terminate another process."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        require(ctypes.get_last_error() == 87, 'Cannot establish process identity/absence')
        return {'pid': pid, 'present': False}
    try:
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(v) for v in values]):
            raise ctypes.WinError(ctypes.get_last_error())
        exit_code = wintypes.DWORD()
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
        return dict(pid=pid, present=True, creation_filetime_100ns=created, running=exit_code.value == 259)
    finally:
        kernel.CloseHandle(handle)


def witness_ticks():
    delta = datetime.fromisoformat(H007_EXIT_WITNESS) - datetime(1601, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**7 + delta.microseconds * 10


def h007_gate():
    """Pin a sealed job and an observed exit, rather than trusting a PID alone."""
    state_path, seal_path = H007 / 'autostart-state.json', H007 / 'burden_v1/seal.json'
    complete_path = H007 / 'burden_v1/complete.json'
    require(not (H007 / 'autostart-failure.json').exists(), 'H007 has an engineering-failure marker')
    require(sha(state_path) == H007_STATE_SHA and sha(seal_path) == H007_SEAL_SHA,
            'Pinned H007 state/seal changed; explicit inspection required')
    state, seal = read(state_path), read(seal_path)
    require(state.get('pid') == H007_PID and state.get('run_id') == H007_RUN and
            state.get('stage') == 'observations_sealed_not_evaluated', 'H007 is not the pinned sealed run')
    require(isinstance(seal, dict) and seal.get('complete.json') == H007_COMPLETE_SHA and
            complete_path.resolve() == complete_path and sha(complete_path) == H007_COMPLETE_SHA,
            'H007 completion does not match its retained seal')
    identity = process(H007_PID)
    if identity.get('running'):
        require(identity['creation_filetime_100ns'] > witness_ticks(),
                'H007 worker identity is still active/indeterminate; do not compete for GPU')
        identity['interpretation'] = 'PID reused after independently witnessed H007 exit'
    else:
        identity['interpretation'] = 'Pinned H007 worker absent or exited'
    return dict(run_id=H007_RUN, state_sha256=H007_STATE_SHA, seal_sha256=H007_SEAL_SHA,
        complete_sha256=H007_COMPLETE_SHA, exit_witness_utc=H007_EXIT_WITNESS, process_identity=identity,
        scope='Sealed completion identity only; no inference inputs or construction truth are opened')


def gpu_external():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free,uuid,driver_version',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW)
    free, uuid, driver = result.stdout.strip().split(',')
    return dict(free_bytes=int(free) * 1024**2, uuid=uuid.strip(), driver_version=driver.strip())


def download_state():
    require(sha(DOWNLOAD_SCRIPT) == DOWNLOAD_SCRIPT_SHA, 'Pinned download script changed')
    state = read(DOWNLOAD)
    require(state.get('model_id') == REPO and state.get('revision') == REVISION and
            Path(state.get('target', '')).resolve() == MODEL.resolve(), 'Download identity changed')
    status = state.get('status')
    require(status in {'started', 'ready_assets_only_no_model_load'}, 'Download failed/invalid; no retry')
    if status == 'started':
        identity = process(DOWNLOAD_PID)
        require(identity.get('running') and identity.get('creation_filetime_100ns') == DOWNLOAD_CREATION,
                'Original download worker exited or PID was reused before completion')
        return None, dict(status=status, stage=state.get('stage'), file=state.get('current_file'),
            received_bytes=state.get('current_received_bytes'), worker=identity)
    entries = state.get('files')
    require(isinstance(entries, list) and len(entries) == 9 and
            {row.get('file') for row in entries} == set(FILES), 'Complete nine-file download receipt required')
    require(state.get('total_bytes') == sum(value[0] for value in FILES.values()), 'Download total-byte identity changed')
    for entry in entries:
        size, digest = FILES[entry['file']]
        require(entry.get('bytes') == size and entry.get('sha256') == digest,
                'Download payload identity differs: ' + entry['file'])
    return state, dict(status=status, receipt_sha256=sha(DOWNLOAD))


def verify_assets():
    require(MODEL.resolve() == MODEL and MODEL.is_dir() and not MODEL.is_symlink(), 'Model directory aliases another target')
    require({path.name for path in MODEL.iterdir()} == set(FILES), 'Missing, partial or extra model assets')
    records = []
    for name, (size, digest) in FILES.items():
        path = MODEL / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == size and sha(path) == digest,
                'Frozen asset hash/size mismatch: ' + name)
        records.append(dict(file=name, bytes=size, sha256=digest))
    config = read(MODEL / 'config.json')
    quant = config.get('quantization_config')
    require(config.get('architectures') == ['Qwen2ForCausalLM'] and config.get('model_type') == 'qwen2'
            and config.get('torch_dtype') == 'bfloat16' and not config.get('auto_map'), 'Native BF16 Qwen2 config required')
    require(isinstance(quant, dict) and quant.get('quant_method') == 'bitsandbytes' and
            quant.get('bnb_4bit_compute_dtype') == 'bfloat16' and quant.get('bnb_4bit_quant_type') == 'nf4'
            and quant.get('bnb_4bit_quant_storage') == 'uint8' and quant.get('bnb_4bit_use_double_quant') is True
            and quant.get('llm_int8_enable_fp32_cpu_offload') is False, 'Saved community NF4/BF16 semantics changed')
    for public, private, expected in [('load_in_4bit', '_load_in_4bit', True), ('load_in_8bit', '_load_in_8bit', False)]:
        values = [quant[key] for key in (public, private) if key in quant]
        require(values and all(type(value) is bool and value is expected for value in values), 'Conflicting saved quantization flags')
    return records, config


def qualify(state, save, check_stop, done):
    require(os.name == 'nt' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Use the pinned Windows GPU environment')
    state['runtime_versions'] = {name: importlib.metadata.version(name) for name in RUNTIME}
    require(state['runtime_versions'] == RUNTIME, 'Pinned torch/Transformers/bitsandbytes/accelerate runtime required')
    for name in ('safetensors', 'tokenizers'):
        state['runtime_versions'][name] = importlib.metadata.version(name)
    while True:
        check_stop()
        state['h007_exit_gate'] = h007_gate()
        complete, progress = download_state()
        save(stage='waiting_for_download', download=progress)
        if complete is not None:
            break
        done.wait(POLL_SECONDS)
    save(stage='verify_complete_payloads')
    state['model_files'], config = verify_assets()
    state['download_receipt_sha256'] = progress['receipt_sha256']
    state['saved_quantization_config'] = config['quantization_config']
    save()
    while True:
        check_stop(); h007_gate()
        available, external = ram(), gpu_external()
        save(stage='waiting_prewarm_resources', available_ram_bytes=available, external_gpu=external)
        if available >= PREWARM_RAM and external['free_bytes'] >= GPU_FREE:
            break
        done.wait(POLL_SECONDS)
    check_stop()
    require(sha(DOWNLOAD) == state['download_receipt_sha256'], 'Completed download receipt changed')
    # Only this process's settings are changed. No API/network calls are permitted.
    sys.dont_write_bytecode = True
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0')
    save(stage='zero_weight_backend_warmup')
    import torch
    import bitsandbytes
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.__version__ == RUNTIME['torch'] and torch.version.cuda == '12.8'
            and transformers.__version__ == RUNTIME['transformers']
            and bitsandbytes.__version__ == RUNTIME['bitsandbytes'], 'Imported runtime differs from pinned package metadata')
    require(torch.cuda.is_available(), 'CUDA is unavailable')
    torch.cuda.set_device(0)
    require(torch.cuda.get_device_capability(0) == (12, 0) and 'sm_120' in torch.cuda.get_arch_list()
            and torch.cuda.is_bf16_supported(), 'SM120/BF16 CUDA runtime required')
    state['hardware'] = dict(device='cuda:0', name=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)), compiled_architectures=torch.cuda.get_arch_list())
    while True:
        check_stop(); h007_gate()
        free, total = torch.cuda.mem_get_info(0)
        available = ram()
        save(stage='waiting_postwarm_resources', available_ram_bytes=available, free_cuda_bytes=free)
        if available >= POSTWARM_RAM and free >= GPU_FREE:
            break
        done.wait(POLL_SECONDS)
    fraction = ALLOCATOR_CAP / total
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    state['admission'] = dict(prewarm_minimum_ram_bytes=PREWARM_RAM, postwarm_minimum_ram_bytes=POSTWARM_RAM,
        minimum_free_gpu_bytes=GPU_FREE, admitted_postwarm_ram_bytes=available, admitted_postwarm_gpu_bytes=free,
        allocator_limit_bytes=ALLOCATOR_CAP, allocator_fraction=fraction,
        allocator_scope='PyTorch caching allocator only; external CUDA allocations are outside this cap')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast, 'Saved fast tokenizer required')
    messages = [{'role': 'user', 'content': 'Choose A, B, or C for a runtime engineering shape check.'}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    short_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    require(0 < len(short_ids) < 512 and tokenizer.encode(prefix, add_special_tokens=False) == short_ids,
            'Synthetic short prefix token alignment differs')
    label_ids = {}
    for label in ('A', 'B', 'C'):
        ids = tokenizer.encode(label, add_special_tokens=False)
        require(len(ids) == 1 and tokenizer.encode(prefix + label, add_special_tokens=False) == short_ids + ids,
                'Engineering label is not one aligned token: ' + label)
        label_ids[label] = ids[0]
    require(len(set(label_ids.values())) == 3, 'Engineering label IDs must be distinct')
    token = next(value for value in tokenizer.encode('engineering', add_special_tokens=False)
                 if value not in set(tokenizer.all_special_ids))
    boundary_ids = [token] * 512
    state['synthetic_inputs'] = dict(short_messages=messages, short_tokens=len(short_ids),
        boundary_tokens=512, boundary_repeated_non_special_token=token, truncation=False,
        label_token_ids=label_ids, semantic_labels_or_expected_predictions=None)
    check_stop(); h007_gate()
    save(stage='reverify_payloads_immediately_before_weights')
    verified_again, config_again = verify_assets()
    require(verified_again == state['model_files'] and config_again == config,
            'Assets changed while waiting for resources')
    require(sha(DOWNLOAD) == state['download_receipt_sha256'] and
            ram() >= POSTWARM_RAM and torch.cuda.mem_get_info(0)[0] >= GPU_FREE,
            'Download identity/resources changed immediately before weights')
    save(stage='load_saved_community_nf4_bf16_cuda0')
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, device_map={'': 0}, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
    # Deliberately no quantization_config override, .to(), CPU/disk map, or offload directory.
    state['weights_loaded'] = True
    torch.cuda.synchronize(0)
    quant = model.config.quantization_config
    quant = quant.to_dict() if hasattr(quant, 'to_dict') else quant
    require(not model.training and getattr(model, 'is_loaded_in_4bit', False)
            and model.config._attn_implementation == 'sdpa' and model.dtype == torch.bfloat16,
            'Loaded model is not eval / saved 4-bit / SDPA / BF16')
    require(isinstance(quant, dict) and quant.get('bnb_4bit_compute_dtype') == 'bfloat16'
            and quant.get('bnb_4bit_quant_type') == 'nf4' and quant.get('bnb_4bit_use_double_quant') is True,
            'Loaded quantization changed saved BF16/NF4 semantics')
    mapping = getattr(model, 'hf_device_map', {})
    require(mapping and all(str(value) in {'0', 'cuda:0'} for value in mapping.values())
            and all(str(parameter.device) == 'cuda:0' for parameter in model.parameters()),
            'CPU/disk offload or a non-cuda0 model device is forbidden')
    state['loaded_model'] = dict(class_name=type(model).__name__, dtype=str(model.dtype), attention='sdpa',
        hf_device_map={key: str(value) for key, value in mapping.items()}, quantization_config=quant,
        load_seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    save()
    for name, ids in [('synthetic_short', short_ids), ('synthetic_exact_512', boundary_ids)]:
        check_stop(); h007_gate()
        require(state['forward_attempts'] < 2, 'Only two engineering forwards are allowed')
        inputs = torch.tensor([ids], dtype=torch.long, device='cuda:0')
        mask = torch.ones_like(inputs)
        torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
        state['forward_attempts'] += 1
        save(stage=name)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=inputs, attention_mask=mask, logits_to_keep=1, use_cache=False).logits
        torch.cuda.synchronize(0)
        shape = list(logits.shape)
        finite = bool(torch.isfinite(logits).all().item())
        record = dict(name=name, input_shape=list(inputs.shape), logits_shape=shape, finite_logits=finite,
            engineering_label_logits_finite=None,
            seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0), use_cache=False, logits_to_keep=1,
            semantic_prediction_or_accuracy=None)
        state['forwards'].append(record); save()
        require(shape == [1, 1, model.config.vocab_size] and finite,
                'Engineering output shape/finiteness failed')
        record['engineering_label_logits_finite'] = bool(torch.isfinite(logits[0, 0, list(label_ids.values())]).all().item())
        require(record['engineering_label_logits_finite'], 'Engineering-label logits are nonfinite')
        require(record['peak_reserved_bytes'] <= ALLOCATOR_CAP, '6.5 GiB PyTorch allocator cap exceeded')
        state['forward_successes'] += 1
        del logits, mask, inputs
        save()
    require(state['forward_attempts'] == state['forward_successes'] == 2, 'Exactly two successful engineering forwards required')
    state['peak_reserved_bytes'] = max(state['loaded_model']['peak_reserved_bytes'],
                                      *(row['peak_reserved_bytes'] for row in state['forwards']))
    state['peak_allocated_bytes'] = max(state['loaded_model']['peak_allocated_bytes'],
                                       *(row['peak_allocated_bytes'] for row in state['forwards']))
    require(state['peak_reserved_bytes'] <= ALLOCATOR_CAP and sha(DOWNLOAD) == state['download_receipt_sha256'],
            'Final allocator/download identity changed')


def main():
    try:
        stream = RECEIPT.open('x', encoding='utf-8')
    except FileExistsError:
        print(json.dumps({'status': 'refused_existing_runtime_receipt'}), flush=True)
        return 2
    started, lock, done = time.monotonic(), threading.RLock(), threading.Event()
    state = dict(schema='community7b-engineering-readiness-v2', status='started', stage='initial_admission',
        started_utc=now(), helper_pid=os.getpid(), python=sys.executable, python_version=sys.version,
        script=str(Path(__file__).resolve()), script_sha256=sha(__file__), model_id=REPO, revision=REVISION,
        download_script_sha256=DOWNLOAD_SCRIPT_SHA, maximum_wall_seconds=MAX_SECONDS, poll_seconds=POLL_SECONDS,
        thresholds=dict(prewarm_ram=PREWARM_RAM, postwarm_ram=POSTWARM_RAM, free_gpu=GPU_FREE, allocator_cap=ALLOCATOR_CAP),
        forward_attempts=0, forward_successes=0, forwards=[], weights_loaded=False, automatic_retry=False,
        dataset_or_gold_opened=False, applications_closed=False, user_processes_terminated=False,
        packages_installed=False, model_assets_modified=False,
        scientific_scope='Community reference engineering only; no historical replay, semantic accuracy, detector efficacy or method adoption.')

    def save(**updates):
        with lock:
            state.update(updates, wall_seconds=time.monotonic() - started, updated_utc=now())
            stream.seek(0); json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.truncate(); stream.flush(); os.fsync(stream.fileno())

    def check_stop():
        require(not STOP.exists(), 'STOP_7B_READINESS cancels this helper only')
        require(time.monotonic() - started < MAX_SECONDS, 'Bounded 45-minute helper deadline elapsed')

    def cancellation():
        # Ephemeral in-process watchdog only; no service/task or external process is controlled.
        while not done.wait(1):
            cancelled = STOP.exists()
            if cancelled or time.monotonic() - started >= MAX_SECONDS:
                save(status='cancelled_this_helper_no_retry' if cancelled else 'engineering_deferred_no_retry',
                    error_stage=state['stage'], error_type='StopMarker' if cancelled else 'FortyFiveMinuteDeadline', ended_utc=now())
                os._exit(125 if cancelled else 124)

    try:
        save()
        watcher = threading.Thread(target=cancellation, daemon=True); watcher.start()
        qualify(state, save, check_stop, done)
        save(status='engineering_ready_two_synthetic_forwards_only', stage='complete', finished_utc=now())
        result = 0
    except BaseException as error:
        save(status='cancelled_this_helper_no_retry' if STOP.exists() else 'engineering_deferred_no_retry',
             error_stage=state['stage'], error_type=type(error).__name__, error_message=str(error),
             failure_traceback=traceback.format_exc(), ended_utc=now())
        result = 1
    finally:
        done.set(); stream.close()
    print(json.dumps({'status': state['status'], 'receipt': str(RECEIPT),
                      'forward_attempts': state['forward_attempts'], 'forward_successes': state['forward_successes']}), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
