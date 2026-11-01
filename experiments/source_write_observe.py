"""Prediction-row attention writes, with explicit native rounding diagnostics."""

import numpy as np
import torch
import torch.nn.functional as F

from myproject.model import QUERY_CHUNK, attention_backend, input_ids, prefill


def partition_writes(attention, values, projection, groups):
    """Project each disjoint key group before summing; retain physical GQA heads."""
    heads = attention.shape[0]
    if heads % values.shape[0] or groups.shape != (3, attention.shape[-1]):
        raise ValueError('Invalid grouped-query heads or key partition')
    if not torch.all(groups.sum(0) == 1):
        raise ValueError('Key groups must form a complete disjoint partition')
    values = values.float().repeat_interleave(heads // values.shape[0], dim=0)
    writes = []
    for mask in groups:
        mixed = torch.einsum('hqk,hkd->qhd', attention.float()*mask, values)
        writes.append(F.linear(mixed.flatten(1), projection.float()))
    return torch.stack(writes)


def source_readouts(model, residual, source, history, native_hidden, targets):
    """Native activation ablations and a distinct smooth FP32 gradient control."""
    actual = targets[:, None]

    def logp(hidden):
        distribution = model.lm_head(hidden).float().log_softmax(-1)
        return distribution.gather(1, actual)[:, 0]

    native = logp(native_hidden)
    replay = model.model.norm(residual.to(native_hidden.dtype))
    minus_source = model.model.norm((residual-source).to(native_hidden.dtype))
    minus_history = model.model.norm((residual-history).to(native_hidden.dtype))
    rms = (residual.square().mean(-1, keepdim=True)+model.model.norm.variance_epsilon).sqrt()
    gamma = model.model.norm.weight.float()
    unembedding = model.lm_head.weight.float()
    smooth_hidden = gamma * residual/rms
    smooth_logp = F.linear(smooth_hidden, unembedding).log_softmax(-1)
    ablated = residual-source
    ablated_rms = (ablated.square().mean(-1,keepdim=True)+model.model.norm.variance_epsilon).sqrt()
    smooth_ablated = F.linear(gamma*ablated/ablated_rms,unembedding).log_softmax(-1)
    direction = unembedding[targets]-smooth_logp.exp() @ unembedding
    scaled = gamma*direction
    gradient = scaled/rms - residual*(scaled*residual).mean(-1,keepdim=True)/rms.pow(3)
    return {
        'native_logp':native, 'direct_gap':logp(minus_source)-native,
        'history_gap':logp(minus_history)-native,
        'source_gradient':-(gradient*source).sum(-1),
        'source_projection':-(unembedding[targets]*gamma*source/rms).sum(-1),
        'norm_replay_max_abs':(replay.float()-native_hidden.float()).abs().max(),
        'smooth_logp':smooth_logp.gather(1,actual)[:,0],
        'smooth_minus_source_logp':smooth_ablated.gather(1,actual)[:,0],
    }


@torch.no_grad()
def write_trace(model, prompt, answer, source_mask):
    """Original y_t is scored at P+t-1; no prompts or answer IDs are rewritten."""
    if not prompt or not answer or len(source_mask) != len(prompt):
        raise ValueError('Nonempty prompts/answers and aligned source mask required')
    cache = prefill(model, prompt[:-1])
    queries = [prompt[-1], *answer[:-1]]
    result = {}
    diagnostics = {'partition_max_abs':0., 'residual_add_max_abs':0., 'norm_replay_max_abs':0.}
    for first in range(0, len(answer), QUERY_CHUNK):
        block = queries[first:first+QUERY_CHUNK]
        query_ids = input_ids(model, block)
        size, width = len(block), model.config.hidden_size
        grouped = torch.zeros((3,size,width),device=model.device,dtype=torch.float32)
        native_attention = torch.zeros_like(grouped[0])
        mlp = torch.zeros_like(native_attention)
        captured, handles = {}, []

        def attention_hook(index):
            def record(module, _args, output):
                weights = output[1][0]
                positions = torch.arange(weights.shape[-1],device=model.device)
                source = torch.zeros_like(positions,dtype=torch.bool)
                source[:len(prompt)] = torch.as_tensor(source_mask,device=model.device)
                history = positions >= len(prompt)
                groups = torch.stack([source, history, ~(source|history)])
                writes = partition_writes(weights, cache.layers[index].values[0],
                                          module.o_proj.weight, groups)
                grouped.add_(writes)
                native = output[0][0].float()
                native_attention.add_(native)
                error = (writes.sum(0)-native).abs().max().item()
                diagnostics['partition_max_abs'] = max(diagnostics['partition_max_abs'],error)
            return record

        def mlp_hook(_module, _args, output):
            mlp.add_(output[0].float())

        def final_input(_module, args):
            captured['residual'] = args[0][0].float().clone()

        try:
            for index, layer in enumerate(model.model.layers):
                handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
                handles.append(layer.mlp.register_forward_hook(mlp_hook))
            handles.append(model.model.norm.register_forward_pre_hook(final_input))
            with attention_backend(model,'eager'):
                hidden = model.model(input_ids=query_ids,past_key_values=cache,
                                     use_cache=True,return_dict=True).last_hidden_state[0]
        finally:
            for handle in handles: handle.remove()
        embedding = model.model.embed_tokens(query_ids)[0].float()
        residual = captured['residual']
        rounding = residual-embedding-native_attention-mlp
        diagnostics['residual_add_max_abs'] = max(diagnostics['residual_add_max_abs'],rounding.abs().max().item())
        targets = input_ids(model,answer[first:first+size])[0]
        readouts = source_readouts(model,residual,grouped[0],grouped[1],hidden,targets)
        diagnostics['norm_replay_max_abs'] = max(diagnostics['norm_replay_max_abs'],
                                               readouts.pop('norm_replay_max_abs').item())
        arrays = {'source_write':grouped[0], 'history_write':grouped[1], 'other_write':grouped[2],
                  'native_attention':native_attention, 'mlp_write':mlp, 'embedding':embedding,
                  'residual':residual, 'rounding':rounding, **readouts}
        for name, value in arrays.items():
            result.setdefault(name,[]).append(value.float().cpu().numpy())
    return {name:np.concatenate(parts) for name,parts in result.items()}, diagnostics
