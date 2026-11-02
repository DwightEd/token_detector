"""Full literal candidate-plus-EOS likelihood; no semantic class normalization.

The collector must freeze all completion_metadata before calling the admitted
GPU scorer. CPU readers use the pure helper, never the GPU parent's preparation.
"""
from copy import deepcopy
import math
import time

from .semantic_qualification import DEFAULT_PROTOCOL, HFGenerator

CONTRACT_VERSION = 'full-candidate-plus-eos-v1'
WHITESPACE_POLICY = 'reject_leading_or_trailing_whitespace'


def _ids(value, name):
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(name + ' must be a flat nonnegative integer token list')
    return list(value)


def completion_metadata(tokenizer, messages, candidate):
    """Pure tokenizer-only framing; preserve the literal, context-dependent suffix.

    With P prefix tokens and K candidate tokens, input has P+K tokens without
    EOS. The K+1 targets are the complete candidate followed by tokenizer EOS;
    their causal prediction positions are P-1 through P+K-1 inclusive.
    """
    if not isinstance(candidate, str) or not candidate or candidate != candidate.strip():
        raise ValueError('A nonempty literal candidate without edge whitespace is required; never strip it')
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True), 'prefix_ids')
    if not isinstance(prefix, str) or not prefix_ids:
        raise ValueError('A nonempty chat prefix is required for the first-token causal prediction')
    if _ids(tokenizer.encode(prefix, add_special_tokens=False), 'encoded_prefix') != prefix_ids:
        raise ValueError('Chat-template prefix tokenization is not aligned')
    input_ids = _ids(tokenizer.encode(prefix + candidate, add_special_tokens=False), 'input_ids')
    if input_ids[:len(prefix_ids)] != prefix_ids or len(input_ids) <= len(prefix_ids):
        raise ValueError('Full prefix+candidate encoding does not retain a strict prefix and nonempty suffix')
    candidate_ids = input_ids[len(prefix_ids):]
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0 or eos in candidate_ids:
        raise ValueError('One fixed tokenizer EOS after the complete non-EOS candidate is required')
    decoded = tokenizer.decode(candidate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != candidate:
        raise ValueError('Context-dependent candidate tokens do not decode to the complete literal candidate')
    target_ids = [*candidate_ids, eos]
    positions = list(range(len(prefix_ids) - 1, len(input_ids)))
    if len(positions) != len(target_ids) or positions[-1] != len(input_ids) - 1:
        raise ValueError('Candidate/EOS causal prediction positions are not aligned')
    return dict(completion_contract_version=CONTRACT_VERSION, raw_candidate=candidate,
        whitespace_policy=WHITESPACE_POLICY, prefix_ids=prefix_ids, prefix_tokens=len(prefix_ids),
        candidate_token_ids=candidate_ids, candidate_tokens=len(candidate_ids), eos_token_id=eos,
        input_ids=input_ids, input_tokens=len(input_ids), completion_eos_appended_to_input=False,
        target_token_ids=target_ids, target_tokens=len(target_ids), target_prediction_positions=positions,
        logits_to_keep=len(target_ids), prefix_alignment_verified=True,
        candidate_roundtrip_verified=True, causal_shift_verified=True)


class HFAnswerCompletion(HFGenerator):
    """Inherit the frozen Qwen3B/NF4 identity and existing 3-GiB resource gate."""

    def validate_completion(self, messages, candidate):
        self._prepare()  # Parent: admitted tokenizer only; never loads model weights.
        return completion_metadata(self.tokenizer, messages, candidate)

    def score_completion(self, messages, candidate):
        """One teacher-forced forward over full name tokens and one fixed EOS.

        This method supplies raw full-vocabulary observations only. The caller
        owns the frozen request manifest; there is no generation, class label,
        candidate-length normalization or fitted candidate normalization here.
        """
        if not self.admitted:
            raise RuntimeError('Explicit resource admission is required before completion scoring')
        metadata = self.validate_completion(messages, candidate)
        if metadata['input_tokens'] > DEFAULT_PROTOCOL['max_input_tokens']:
            raise RuntimeError('Completion input exceeds the frozen limit; truncation is forbidden')
        self._ensure_model()  # Reuse the parent's runtime/identity/resource policy unchanged.
        vocab = getattr(self.model.config, 'vocab_size', None)
        if type(vocab) is not int or vocab <= 0 or any(token >= vocab for token in
                metadata['input_ids'] + metadata['target_token_ids']):
            raise ValueError('Completion token IDs exceed the declared complete model vocabulary')
        import torch
        encoded = torch.tensor([metadata['input_ids']], dtype=torch.long, device='cuda:0')
        count = metadata['target_tokens']
        if tuple(encoded.shape) != (1, metadata['input_tokens']):
            raise RuntimeError('Frozen completion input shape changed')
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model(encoded, attention_mask=torch.ones_like(encoded),
                                logits_to_keep=count, use_cache=False)
            logits = output.logits
            if (tuple(logits.shape) != (1, count, vocab) or not logits.is_floating_point()
                    or not bool(torch.isfinite(logits).all().item())):
                raise ValueError('Completion requires finite full-vocabulary logits at every causal target position')
            rows = logits[0].double()
            normalizers = torch.logsumexp(rows, dim=-1).tolist()
            selected = [float(rows[index, token].item()) for index, token in
                        enumerate(metadata['target_token_ids'])]
        logp = [value - normalizer for value, normalizer in zip(selected, normalizers, strict=True)]
        joint = math.fsum(logp)
        if (not all(math.isfinite(value) for value in [*selected, *normalizers, *logp, joint])
                or any(value > 1e-10 for value in logp)):
            raise ValueError('Invalid full-vocabulary completion log probabilities; no clipping is allowed')
        torch.cuda.synchronize(0)
        return dict(metadata, target_token_logits=selected, full_vocab_logsumexp=normalizers,
            target_token_log_probabilities=logp, joint_log_probability=joint,
            full_vocab_normalizer_dtype='float64', vocab_size=vocab, logits_shape=list(logits.shape),
            actual_forward_passes=1, output_tokens=0, seconds=time.monotonic() - started,
            resource_admission=deepcopy(self.admission),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
