"""Admitted prefix-only node observations, with complete FP16 rows for CPU audit.

Each unique causal context gets one forward and one shared vocabulary row. No
files are written here. Failures return captured raw arrays and actual costs.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

from . import data
from .answer_completion import completion_metadata
from .research import completion_trie as kernel
from .semantic_qualification import HFGenerator

CONTRACT_VERSION = 'prefix-only-full-vector-observation-v1'
MAX_CONTEXT_TOKENS = 4096
MAX_NODES = 32
MAX_TEMPORARY_VECTOR_BYTES = 160 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class HFPrefixCompletion(HFGenerator):
    """Reuse frozen Qwen-3B NF4/FP16/SDPA and the parent's 3-GiB admission."""

    def __init__(self, model, model_identity, *, admit_resources=False):
        super().__init__(model, deepcopy(model_identity), admit_resources=admit_resources)
        _require(len(self.model_identity['model_files_sha256']) == 9, 'All nine frozen model assets required')
        self._identity = deepcopy(self.model_identity)
        self._assets = None
        self._validated = {}
        self.last_result = None

    def _asset_guard(self):
        _require(kernel._canonical(self.model_identity) == kernel._canonical(self._identity), 'Model identity mutated')
        files = self._identity['model_files_sha256']
        if self._assets is None:
            self.verify_frozen_identity()
            config = data.read_json(self.model_path / 'config.json')
            vocab = config.get('vocab_size')
            _require(type(vocab) is int and vocab > 0, 'Frozen config must declare a positive full vocabulary')
            self._assets = dict(vocab_size=vocab, stamps={name: ((self.model_path / name).stat().st_size,
                (self.model_path / name).stat().st_mtime_ns) for name in files})
        _require({str(p.relative_to(self.model_path)) for p in self.model_path.rglob('*') if p.is_file()} == set(files),
                 'Model directory population changed')
        for name, stamp in self._assets['stamps'].items():
            path = self.model_path / name
            _require(path.resolve() == path and not path.is_symlink()
                and (path.stat().st_size, path.stat().st_mtime_ns) == stamp, 'Frozen model asset changed')
            if not name.endswith('.safetensors'):
                _require(data.file_hash(path) == files[name], 'Frozen tokenizer/config asset changed')
        return self._assets['vocab_size']

    def _framing(self, messages, candidates):
        framed = {}
        for candidate in sorted(candidates):
            copied = deepcopy(messages)
            before = kernel._canonical(copied)
            framed[candidate] = completion_metadata(self.tokenizer, copied, candidate)
            _require(kernel._canonical(copied) == before, 'Tokenizer mutated input messages')
        return framed

    def validate_trie(self, messages, candidates):
        """Tokenizer/identity only; register exact caller inputs before any model."""
        if self.admitted is not True:
            raise RuntimeError('Explicit HF resource admission required')
        _require(isinstance(messages, list) and messages and all(isinstance(m, dict)
            and set(m) == {'role', 'content'} and m['role'] in ('system', 'user', 'assistant')
            and isinstance(m['content'], str) for m in messages), 'Exact text-only chat messages required')
        _require(isinstance(candidates, (list, tuple)) and candidates
            and all(isinstance(c, str) for c in candidates) and len(set(candidates)) == len(candidates),
            'Complete unique candidate literals required')
        message_hash, candidate_hash = kernel._digest(messages), kernel._digest(list(candidates))
        self._prepare()  # Inherited admitted tokenizer-only stage; no _ensure_model.
        vocab = self._asset_guard()
        trie = kernel.build_trie(self._framing(messages, candidates))
        _require(trie['status'] == 'ok', 'Tokenizer trie rejected: ' + str(trie['reason']))
        self._limits(trie, vocab)
        _require(kernel._digest(messages) == message_hash and kernel._digest(list(candidates)) == candidate_hash,
                 'Input messages/candidates changed during validation')
        signature = kernel._digest(trie)
        previous = self._validated.get(signature)
        _require(previous is None or previous['message_hash'] == message_hash, 'Different messages alias the same token trie')
        self._validated[signature] = dict(messages_input=messages, candidates_input=candidates,
            messages=deepcopy(messages), candidates=tuple(candidates), message_hash=message_hash,
            candidate_hash=candidate_hash, metadata=deepcopy(trie['completion_metadata']), tokenizer=self.tokenizer)
        return trie

    @staticmethod
    def _limits(trie, vocab):
        _require(0 < len(trie['nodes']) <= MAX_NODES, 'Tree exceeds the fixed32-node bound')
        _require(len(trie['nodes']) * vocab * 32 <= MAX_TEMPORARY_VECTOR_BYTES,
                 'Temporary Python full-vector budget exceeded')
        for node in trie['nodes'].values():
            context = node['context_ids']
            _require(0 < len(context) <= MAX_CONTEXT_TOKENS, 'Prefix-only context exceeds4096; truncation forbidden')
            _require(all(type(i) is int and 0 <= i < vocab for i in context)
                and all(type(e['token_id']) is int and 0 <= e['token_id'] < vocab for e in node['children']),
                'Node context or child target is outside frozen vocabulary')

    def _guard(self, trie, signature, registration, vocab):
        _require(kernel._digest(trie) == signature, 'Caller trie/token metadata mutated')
        _require(kernel._digest(registration['messages_input']) == registration['message_hash']
            and kernel._digest(list(registration['candidates_input'])) == registration['candidate_hash'],
            'Caller messages/candidates mutated after validation')
        _require(self.tokenizer is registration['tokenizer']
            and self._framing(registration['messages'], registration['candidates']) == registration['metadata'],
            'Tokenizer completion framing changed')
        _require(self._asset_guard() == vocab, 'Declared vocabulary changed')
        admission = getattr(self, 'admission', {})
        _require(admission.get('device') == 'cuda:0' and admission.get('cuda_allocator_limit_bytes') == 3 * 1024**3
            and type(admission.get('available_ram_bytes')) is int and admission['available_ram_bytes'] >= 2 * 1024**3
            and type(admission.get('free_cuda_bytes')) is int and admission['free_cuda_bytes'] >= 3 * 1024**3,
            'Inherited resource-admission policy changed')
        if self.model is not None:
            _require(getattr(self.model, 'training', None) is False
                and getattr(self.model.config, 'vocab_size', None) == vocab, 'Loaded model mode/config changed')

    def score_trie(self, trie):
        """Return status plus partial raw on failure; never retry or write arrays.

        Successful raw_nodes contain read-only native FP16 vectors. Only the
        temporary oracle input uses Python float lists, which are released here.
        """
        started = time.monotonic()
        raw_nodes, temporary = [], []
        result = dict(schema=CONTRACT_VERSION, status='engineering_failed', tree_result=None,
            raw_nodes=raw_nodes, model_calls=0, attempted_model_calls=0, actual_forward_passes=0,
            output_tokens=0, input_tokens=0, pure_kernel_model_calls=0, error_stage='preflight',
            error_type=None, reason=None, automatic_retry=False,
            model_identity=deepcopy(self._identity), model_identity_sha256=kernel._digest(self._identity))
        self.last_result = result
        try:
            if self.admitted is not True:
                raise RuntimeError('Explicit HF resource admission required')
            canonical = kernel.build_trie(trie.get('completion_metadata') if isinstance(trie, dict) else None)
            _require(canonical['status'] == 'ok' and kernel._canonical(trie) == kernel._canonical(canonical),
                     'Complete canonical trie preflight failed')
            signature = kernel._digest(trie)
            registration = self._validated.get(signature)
            _require(registration is not None, 'This exact trie must first pass validate_trie on this backend')
            vocab = self._asset_guard()
            self._limits(trie, vocab)  # Every node/edge/budget, before any weight load.
            self._guard(trie, signature, registration, vocab)
            result.update(trie_sha256=signature, unique_nodes=len(trie['nodes']),
                          temporary_vector_budget_bytes=len(trie['nodes']) * vocab * 32)
            result['error_stage'] = 'model_load'
            self._ensure_model()  # Inherited local-only NF4/FP16/SDPA/3-GiB policy.
            self._guard(trie, signature, registration, vocab)
            import torch
            for key, node in trie['nodes'].items():
                self._guard(trie, signature, registration, vocab)
                context = list(node['context_ids'])
                encoded = torch.tensor([context], dtype=torch.long, device='cuda:0')
                _require(tuple(encoded.shape) == (1, len(context)), 'Prefix-only input shape changed')
                torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
                tick = time.monotonic()
                result['error_stage'] = 'node_forward'
                result['attempted_model_calls'] += 1; result['model_calls'] += 1
                result['input_tokens'] += len(context)
                raw = dict(node_id=key, context_ids=context, context_sha256=kernel._digest(context),
                    resource_admission=deepcopy(self.admission), full_fp16_vector=None,
                    actual_forward_passes=0, output_tokens=0, input_tokens=len(context))
                raw_nodes.append(raw)
                try:
                    with torch.inference_mode():
                        output = self.model(encoded, attention_mask=torch.ones_like(encoded), logits_to_keep=1, use_cache=False)
                        result['actual_forward_passes'] += 1; raw['actual_forward_passes'] = 1
                        logits = output.logits
                        raw.update(logits_shape=list(logits.shape), observed_dtype=str(logits.dtype))
                        result['error_stage'] = 'node_capture'
                        # Preserve bounded invalid observations without silently changing dtype.
                        if logits.numel() <= 2 * vocab:
                            cpu_logits = logits.detach().cpu().contiguous()
                            try:
                                retained = cpu_logits.numpy().copy()
                                retained.setflags(write=False)
                            except TypeError:
                                retained = None
                                bits = cpu_logits.view(torch.uint8).numpy().copy()
                                bits.setflags(write=False)
                                raw['invalid_raw_bytes'] = bits  # Unsupported NumPy dtype: exact bits only.
                            if retained is not None and tuple(logits.shape) == (1, 1, vocab) and logits.dtype == torch.float16:
                                vector = retained.reshape(vocab); vector.setflags(write=False)
                                raw.update(full_fp16_vector=vector, full_vector_dtype=vector.dtype.str,
                                    full_vector_shape=list(vector.shape), full_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest())
                            elif retained is not None:
                                raw['invalid_raw_logits'] = retained
                        _require(tuple(logits.shape) == (1, 1, vocab) and logits.dtype == torch.float16,
                                 'One complete FP16 vocabulary row required; no dtype conversion')
                        _require(raw['full_fp16_vector'] is not None and bool(torch.isfinite(logits).all().item()),
                                 'Every full-vocabulary logit must be finite and retained')
                finally:
                    raw['seconds'] = time.monotonic() - tick
                    try:
                        torch.cuda.synchronize(0)
                        raw.update(seconds=time.monotonic() - tick,
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
                    except Exception as measurement_error:
                        raw['measurement_error'] = type(measurement_error).__name__
                result['error_stage'] = 'resource_measurement'
                _require('measurement_error' not in raw, 'Required CUDA timing/peak measurement failed')
                allocated, reserved = raw['peak_cuda_allocated_bytes'], raw['peak_cuda_reserved_bytes']
                _require(type(allocated) is int and type(reserved) is int
                    and 0 <= allocated <= reserved <= 3 * 1024**3, 'Allocator peaks violate the inherited3-GiB policy')
                temporary.append(dict(node_id=key, context_ids=context, vocab_size=vocab,
                                      full_vocab_logits=raw['full_fp16_vector'].tolist()))
                result['error_stage'] = 'immutability_guard'
                self._guard(trie, signature, registration, vocab)
            result['error_stage'] = 'pure_probability_gate'
            result['tree_result'] = kernel.score_trie(trie, temporary)
            _require(result['tree_result']['status'] == 'ok', 'Pure probability tree rejected: ' + str(result['tree_result']['reason']))
            _require(result['actual_forward_passes'] == result['model_calls'] == len(trie['nodes']) == len(raw_nodes),
                     'Exactly one actual forward per unique context required')
            self._guard(trie, signature, registration, vocab)
            result.update(status='scored', error_stage=None)
        except Exception as error:
            result.update(error_type=type(error).__name__, reason=str(error))
        finally:
            temporary.clear()
            result['seconds_total'] = time.monotonic() - started
            result['seconds_forward'] = sum(row.get('seconds', 0.) for row in raw_nodes)
        return result
