"""Bounded exact-weight observer execution using the complete native HF forward.

This is resource-qualification infrastructure, not a detector.  BF16 parameters
are mapped from safetensors one tensor at a time; only one embedding, decoder
block, final norm or unchunked vocabulary head is resident.  Every model equation,
mask, cache update and rotary operation remains in Transformers 4.57.1.
Original-weight execution needs a separately claimed run and historical canaries.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import struct
import threading
from types import MethodType

from experiments.data import file_hash, read_json


def _contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or Path(name).name != name:
        raise ValueError('Weight index shard must be a contained basename')
    return path


def _headers(path):
    """Safetensors metadata only; never deserialize a shard's tensor payload."""
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors prefix')
        length, = struct.unpack('<Q', prefix)
        if length > 16 * 1024**2:
            raise ValueError('Unexpected safetensors header size')
        metadata = json.loads(stream.read(length))
    size = path.stat().st_size - 8 - length
    result = {}
    intervals = []
    for key, row in metadata.items():
        if key == '__metadata__':
            continue
        if row['dtype'] != 'BF16' or any(d < 1 for d in row['shape']):
            raise ValueError('Observer weights must be unquantized BF16 tensors')
        begin, end = row['data_offsets']
        count = 1
        for dimension in row['shape']:
            count *= dimension
        if begin < 0 or end > size or end - begin != count * 2:
            raise ValueError('Invalid safetensors shape/offset contract')
        intervals.append((begin, end))
        result[key] = {'shape': tuple(row['shape']), 'bytes': count * 2}
    intervals.sort()
    if intervals and (intervals[0][0] != 0 or intervals[-1][1] != size
                      or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))):
        raise ValueError('Noncontiguous safetensors payload offsets')
    return result


class SafeTensorStore:
    """Witness exact files, then mmap only requested tensors to the device.

    ``expected_files`` uses MODEL_ENVIRONMENT's {name:{sha256,bytes}} format.
    The index hash is separately witnessed: its historical hash was not captured.
    Initial full-file hashing is bounded-memory I/O, not a whole-shard load.
    """

    def __init__(self, directory, *, index_sha256, expected_files):
        self.root = Path(directory).resolve()
        self.expected_files = expected_files
        self.index_path = self.root / 'model.safetensors.index.json'
        if file_hash(self.index_path) != index_sha256:
            raise ValueError('Newly witnessed index SHA differs')
        self.weight_map = read_json(self.index_path)['weight_map']
        shards = set(self.weight_map.values())
        expected_shards = {n for n in expected_files if n.endswith('.safetensors')}
        if shards != expected_shards:
            raise ValueError('Shard index differs from captured weight file set')
        self.headers, self.stats, self.identity = {}, {}, {}
        self.bytes_transferred = 0
        for name in sorted(shards):
            path = _contained(self.root, name)
            expected = expected_files[name]
            if path.stat().st_size != expected['bytes'] or file_hash(path) != expected['sha256']:
                raise ValueError('Weight shard bytes differ from captured identity')
            header = _headers(path)
            if set(header) != {k for k, v in self.weight_map.items() if v == name}:
                raise ValueError('Every index key must match its shard header')
            if set(self.headers) & set(header):
                raise ValueError('Duplicate tensor key across shards')
            self.headers.update(header)
            self.stats[name] = (path.stat().st_size, path.stat().st_mtime_ns)
            self.identity[name] = dict(expected)
        self.index_sha256 = index_sha256

    def tensor(self, name, device):
        """No load_file/get_tensors: one mmap view, one requested device transfer."""
        from safetensors import safe_open
        shard = self.weight_map[name]
        path = self.root / shard
        if (path.stat().st_size, path.stat().st_mtime_ns) != self.stats[shard]:
            raise RuntimeError('Verified weight shard changed after admission')
        with safe_open(path, framework='pt', device='cpu', backend='mmap') as stream:
            value = stream.get_tensor(name)
            output = value.to(device=device)
        self.bytes_transferred += self.headers[name]['bytes']
        return output


@dataclass(frozen=True)
class ResourceLimits:
    max_sequence_tokens: int = 256
    max_planned_gpu_bytes: int = 6 * 1024**3
    max_host_tensor_bytes: int = 1024**3
    free_gpu_floor_bytes: int = 512 * 1024**2
    free_host_floor_bytes: int = 512 * 1024**2
    workspace_bytes: int = 256 * 1024**2


def _available_ram():
    if os.name == 'nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(n, ctypes.c_ulonglong) for n in
                          ('total', 'available', 'page_total', 'page_available',
                           'virtual_total', 'virtual_available', 'extended')]]
        state = Status()
        state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError('Cannot query physical-memory admission')
        return state.available
    return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


def _fp32_linear(module, inputs):
    import torch
    import torch.nn.functional as functional
    if threading.get_ident() != module._stream_owner:
        raise RuntimeError('Streamed observer is restricted to its owner thread')
    _check_precision_context()
    _rows(inputs, module._stream_row_limit)
    if module.weight.is_meta:
        raise RuntimeError('Direct child Linear is not resident; use observer.materialize(path)')
    return functional.linear(inputs.float(), module.weight.float())


def _float_embedding(module, inputs, output):
    return output.float()


def _check_precision_context():
    import torch
    if torch.is_grad_enabled():
        raise RuntimeError('Streamed observer requires no-grad/inference-mode execution')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise RuntimeError('Autocast is incompatible with the FP32 observer contract')


def _rows(value, maximum, *, embedding=False):
    if embedding:
        valid = value.ndim == 2 and value.shape[0] == 1
        rows = value.shape[1] if valid else 0
    elif value.ndim == 3:
        valid = value.shape[0] == 1
        rows = value.shape[1]
    else:
        valid = value.ndim == 2
        rows = value.shape[0] if valid else 0
    if not valid:
        raise ValueError('Bounded direct observer calls require batch1')
    if rows < 1 or rows > maximum:
        raise MemoryError('Direct input rows exceed the frozen resource limit')
    return rows


class StreamedObserver:
    """Native Llama interface with bounded module residency and explicit cleanup.

    ``original_contract=False`` exists only for tiny synthetic engineering tests;
    it is not an alternative observer or a scientific resource fallback.
    Direct calls to decoder descendants (e.g. archived graph_arrays' o_proj)
    need ``with observer.materialize('model.layers.15.self_attn.o_proj')``.
    """

    def __init__(self, store, config, *, config_sha256, device='cuda:0',
                 limits=ResourceLimits(), original_contract=True):
        import torch
        from transformers.models.llama import modeling_llama as implementation
        from transformers.models.llama.configuration_llama import LlamaConfig
        if version('transformers') != '4.57.1' or version('safetensors') != '0.8.0':
            raise ValueError('Exact Transformers4.57.1/Safetensors0.8.0 required')
        if torch.__version__.split('+')[0] != '2.8.0':
            raise ValueError('Exact Torch2.8.0 arithmetic contract required')
        self.device, self.store, self.limits = torch.device(device), store, limits
        self._owner = threading.get_ident()
        self.handles, self._active = [], None
        self._closed, self._sequence_tokens = False, 0
        self.loads, self.max_planned_bytes = 0, 0
        self._tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        if original_contract:
            if self.device.type != 'cuda':
                raise ValueError('Original-weight local resource plan requires CUDA')
            expected = {'model_type': 'llama', 'hidden_size': 4096,
                        'num_hidden_layers': 32, 'num_attention_heads': 32,
                        'num_key_value_heads': 8, 'vocab_size': 128256,
                        'tie_word_embeddings': False, 'attention_bias': False,
                        'mlp_bias': False}
            if any(config.get(k) != v for k, v in expected.items()):
                raise ValueError('Unsupported original observer configuration')
            if len(store.identity) != 4:
                raise ValueError('Original observer requires its four attested shards')
        if (file_hash(store.root / 'config.json') != config_sha256
                or read_json(store.root / 'config.json') != config
                or (original_contract and store.expected_files.get('config.json', {}).get('sha256') != config_sha256)):
            raise ValueError('Observer config SHA differs from captured identity')
        native_config = LlamaConfig.from_dict(deepcopy(config))
        native_config._attn_implementation = 'sdpa'
        execution_device = self.device
        class NativeStreamedLlama(implementation.LlamaForCausalLM):
            @property
            def device(self):
                return execution_device
        with torch.device('meta'):
            self.native = NativeStreamedLlama(native_config).to(dtype=torch.bfloat16)
        self.native.eval().requires_grad_(False)
        expected_state = self.native.state_dict()
        if set(expected_state) != set(store.headers):
            raise ValueError('Official model keys differ from every indexed tensor')
        if any(tuple(value.shape) != store.headers[name]['shape']
               for name, value in expected_state.items()):
            raise ValueError('Official model parameter shape differs from shard header')
        self.implementation = implementation
        # The official rotary buffers must be constructed outside the meta scope.
        self.native.model.rotary_emb = implementation.LlamaRotaryEmbedding(
            native_config, device=self.device)
        if any(buffer.is_meta for _, buffer in self.native.named_buffers()):
            raise RuntimeError('Unmaterialized non-weight model buffer')
        for module in self.native.modules():
            if isinstance(module, torch.nn.Linear):
                if module.bias is not None:
                    raise ValueError('Frozen observer FP32 Linear requires bias-free weights')
                module.forward = MethodType(_fp32_linear, module)
                module._stream_owner = self._owner
                module._stream_row_limit = limits.max_sequence_tokens
        self.handles.append(self.native.model.embed_tokens.register_forward_hook(_float_embedding))
        self.managed = {'model.embed_tokens': self.native.model.embed_tokens,
                        **{f'model.layers.{i}': layer for i, layer in enumerate(self.layers)},
                        'model.norm': self.native.model.norm, 'lm_head': self.native.lm_head}
        for name, module in self.managed.items():
            self.handles.append(module.register_forward_pre_hook(self._preload(name), prepend=True))
            self.handles.append(module.register_forward_hook(self._release, always_call=True))
        self.handles.append(self.native.model.register_forward_pre_hook(self._guard, with_kwargs=True))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @property
    def layers(self):
        return self.native.model.layers

    def head_layout(self, index):
        config = self.native.config
        return config.num_attention_heads, config.num_key_value_heads, config.hidden_size // config.num_attention_heads

    def input_ids(self, tokens):
        import torch
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _guard(self, module, args, kwargs):
        import torch
        if self._closed or torch.is_grad_enabled():
            raise RuntimeError('Streamed observer requires an open no-grad/inference-mode run')
        self._execution_context()
        tokens = kwargs.get('input_ids', args[0] if args else None)
        embeds = kwargs.get('inputs_embeds')
        value = tokens if tokens is not None else embeds
        if value is None or value.shape[0] != 1:
            raise ValueError('Bounded observer requires batch1 input')
        past = kwargs.get('past_key_values')
        length = past.get_seq_length() if past is not None else 0
        self._sequence_tokens = value.shape[1] + length
        if self._sequence_tokens > self.limits.max_sequence_tokens:
            raise MemoryError('Sequence exceeds the frozen resource limit')

    def _execution_context(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Streamed observer is restricted to its owner thread')
        _check_precision_context()

    def _admit(self, module, prefix, rows):
        import torch
        names = [f'{prefix}.{name}' for name, _ in module.named_parameters()]
        weight_bytes = sum(self.store.headers[n]['bytes'] for n in names)
        largest = max((self.store.headers[n]['bytes'] for n in names), default=0)
        if largest > self.limits.max_host_tensor_bytes:
            raise MemoryError('Largest mapped tensor exceeds the host transfer budget')
        if _available_ram() < self.limits.free_host_floor_bytes:
            raise MemoryError('Available physical RAM is below the reserved floor')
        # One Linear cast is live at a time; embedding does not cast its weight.
        cast = max((self.store.headers[f'{prefix}.{n}.weight']['bytes'] * 2
                    for n, part in module.named_modules() if isinstance(part, torch.nn.Linear)
                    and n), default=0)
        if isinstance(module, torch.nn.Linear):
            cast = weight_bytes * 2
        config, seq = self.native.config, rows
        activation = seq * (config.hidden_size * 16 + config.intermediate_size * 24)
        if prefix == 'lm_head':
            activation += seq * config.vocab_size * 4
        planned = weight_bytes + cast + activation + self.limits.workspace_bytes
        if planned > self.limits.max_planned_gpu_bytes:
            raise MemoryError('Module plan exceeds the GPU allocation budget')
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            if planned + self.limits.free_gpu_floor_bytes > free:
                raise MemoryError('Free GPU memory cannot admit the next module')
        self.max_planned_bytes = max(self.max_planned_bytes, planned)

    def _load(self, name, module, rows):
        import torch
        self._execution_context()
        if self._closed or self._active is not None or torch.is_grad_enabled():
            raise RuntimeError('Concurrent or nested module materialization is forbidden')
        self._admit(module, name, rows)
        self._active = (name, module)
        try:
            for relative, parameter in list(module.named_parameters()):
                parent_name, _, leaf = relative.rpartition('.')
                parent = module.get_submodule(parent_name) if parent_name else module
                tensor = self.store.tensor(f'{name}.{relative}', self.device)
                parent._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
            self.loads += 1
        except BaseException:
            self._unload(module)
            raise

    def _unload(self, module):
        module.to('meta')
        if self._active is not None and self._active[1] is module:
            self._active = None

    def _preload(self, name):
        def load(module, args):
            import torch
            self._execution_context()
            rows = _rows(args[0], self.limits.max_sequence_tokens,
                         embedding=isinstance(module, torch.nn.Embedding))
            # A cached native decoder may attend to a longer admitted prefix.
            if name.startswith('model.layers.'):
                rows = max(rows, self._sequence_tokens)
            self._load(name, module, rows)
        return load

    def _release(self, module, args, output):
        # always_call also runs when a preload was rejected.  A foreign caller
        # must never unload the owner thread's currently resident module.
        if (threading.get_ident() == self._owner and self._active is not None
                and self._active[1] is module):
            self._unload(module)

    @contextmanager
    def materialize(self, path, *, rows=None):
        """Explicit bounded residency for a directly called decoder descendant."""
        if path in self.managed:
            raise ValueError('Top-level streamed modules already have residency hooks')
        module = self.native.get_submodule(path)
        if not any(path.startswith(prefix + '.') for prefix in self.managed):
            raise ValueError('Direct materialization must belong to a managed module')
        import torch
        with torch.inference_mode():
            self._execution_context()
            rows = self.limits.max_sequence_tokens if rows is None else rows
            if rows < 1 or rows > self.limits.max_sequence_tokens:
                raise MemoryError('Direct context rows exceed the frozen resource limit')
            self._load(path, module, rows)
            for part in module.modules():
                if isinstance(part, torch.nn.Linear):
                    part._stream_row_limit = rows
            try:
                yield module
            finally:
                self._unload(module)
                for part in module.modules():
                    if isinstance(part, torch.nn.Linear):
                        part._stream_row_limit = self.limits.max_sequence_tokens

    def hidden(self, tokens, **kwargs):
        import torch
        if self._closed:
            raise RuntimeError('Streamed observer requires an open no-grad/inference-mode run')
        with torch.inference_mode():
            return self.native.model(input_ids=self.input_ids(tokens), **kwargs)

    def logits(self, hidden):
        import torch
        if self._closed:
            raise RuntimeError('Streamed observer is closed')
        with torch.inference_mode():
            return self.native.lm_head(hidden)

    def resource_report(self):
        return {'module_loads': self.loads, 'tensor_bytes_transferred': self.store.bytes_transferred,
                'max_planned_gpu_bytes': self.max_planned_bytes,
                'active_module': self._active[0] if self._active else None,
                'all_parameters_meta': all(p.is_meta for p in self.native.parameters()),
                'index_historical_identity': False,
                'precision': 'original BF16 weights, FP32 Linear/embedding-output/native arithmetic',
                'original_capture_qualified': False}

    def close(self):
        import torch
        if threading.get_ident() != self._owner:
            raise RuntimeError('Close must run on the observer owner thread')
        if self._active is not None:
            self._unload(self._active[1])
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module in self.managed.values():
            module.to('meta')
        self.native.model.rotary_emb.to('meta')
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self._tf32
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
