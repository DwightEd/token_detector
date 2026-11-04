"""Same-forward H018 capture; replay while the parent decoder is resident.

The callback receives flat detached device tensors and must stream them instead
of retaining a CPU layer history. This module never loads a model or writes I/O.
"""
from contextlib import contextmanager
from numbers import Integral
import threading

from experiments.source_write_observe import partition_writes
from .operator_binding import (
    OrderedAccumulator, ReplayFailure, _coordinate_gate, _same_bits, local_effect,
)


class CaptureFailure(RuntimeError):
    """Raw failed-layer data has been offered to the caller before this error."""

    def __init__(self, message, layer_index, gates):
        super().__init__(message)
        self.layer_index, self.gates = layer_index, gates


def _list(value):
    return value.tolist() if hasattr(value, 'tolist') else list(value)


@contextmanager
def capture_local_effects(observer, token_ids, source_mask, prechoice_positions,
                          counts, on_layer):
    """Yield ``H_raw/native_postnorm/accumulation/layer_gates/gates`` state.

    Caller runs exactly one native eager full-input forward in this context.
    All T original prechoice rows are kept. Four local F calls execute in the
    native MLP output hook BEFORE StreamedObserver's parent-release hook; late
    decoder hooks only compare/output tensors and never touch weights.
    Numeric failures offer raw tensors to ``on_layer(index, arrays)`` first.
    ``partition_o_calls`` counts three requested helper calls before entry; on
    an exceptional helper exit it is a requested upper bound, not three returns.
    """
    import torch

    ids, mask, positions = map(_list, (token_ids, source_mask, prechoice_positions))
    if (not ids or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in ids)
            or len(mask) != len(ids) or any(type(v) is not bool for v in mask)
            or not positions or any(isinstance(v, bool) or not isinstance(v, Integral) for v in positions)):
        raise ValueError('Exact original integer IDs/positions and Boolean source mask required')
    n, p = len(ids), int(positions[0])+1
    if (not 0 < p < n or positions != list(range(p-1, n-1))
            or not any(mask[:p]) or any(mask[p:])):
        raise ValueError('All original R prechoice rows and prompt-only source mask required')
    if not callable(on_layer) or getattr(observer, '_operator_capture_active', False):
        raise ValueError('One streaming callback and one active capture per observer required')
    layers = observer.layers
    if not len(layers):
        raise ValueError('A native decoder with layers is required')
    owner = threading.get_ident()
    device = observer.device
    selected = torch.as_tensor(positions, dtype=torch.long, device=device)
    keys = torch.arange(n, device=device)
    source = torch.as_tensor(mask, dtype=torch.bool, device=device)
    history = keys >= p  # Includes the receiver self key whenever it is in R.
    groups = torch.stack((source, history, ~(source|history)))
    allowed = keys[None,:] <= selected[:,None]
    accumulator = OrderedAccumulator(expected_layers=len(layers))
    state = dict(H_raw=None, native_postnorm=None, accumulation=None,
                 layer_gates=[], completed_layers=[], gates=dict(passed=False),
                 partition_o_count_scope='three requested calls before helper; exceptional exit is an upper bound',
                 accumulator_cpu_reference_bytes=0)
    records, handles = {}, []
    replay = False
    observer._operator_capture_active = True

    def model_pre(module,args,kwargs):
        active()
        actual = kwargs.get('input_ids',args[0] if args else None)
        if (torch.is_grad_enabled() or any(torch.is_autocast_enabled(d) for d in ('cpu','cuda'))
                or state.get('native_forward_seen',False)):
            raise RuntimeError('One native inference-mode/no-autocast forward is required')
        if actual is not None and actual.device.type!='cpu':
            counts['host_transfer_bytes']=counts.get('host_transfer_bytes',0)+actual.numel()*actual.element_size()
        if (actual is None or tuple(actual.shape)!=(1,n) or actual.dtype!=torch.int64
                or actual[0].tolist()!=ids or kwargs.get('past_key_values') is not None):
            raise ValueError('Native forward must use the complete original token IDs with no past cache')
        state['native_forward_seen'] = True

    def active():
        if threading.get_ident() != owner:
            raise RuntimeError('Native capture is restricted to its owner thread')
        return not replay

    def tensor(value, name, *, width=None):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device != selected.device):
            raise TypeError(name+' must be native FP32 on the observer device')
        if width is not None and tuple(value.shape) != (1,n,width):
            raise ValueError(name+' must preserve the full native [1,N,H] shape')
        if not torch.isfinite(value).all().item():
            raise ValueError(name+' is nonfinite')
        return value

    def rows(value, name, width):
        return tensor(value, name, width=width).index_select(1, selected).detach().clone()

    def returned_rows(value):
        """Keep returned tensors before their dtype/finiteness qualification."""
        if isinstance(value,torch.Tensor):
            if value.ndim==3 and value.shape[:2]==(1,n):
                return value.index_select(1,selected).detach().clone()
            return value.detach().clone()
        return None

    def raw(record):
        result = {key:value for key,value in record.items() if isinstance(value, torch.Tensor)}
        partial = record.get('local', {})
        for group in ('inputs','norms','mlps'):
            for name, value in partial.get(group, {}).items():
                if isinstance(value, torch.Tensor):
                    result['replay_'+group+'_'+name] = value.detach()
        for name in ('delta_source','delta_sham','sham_write'):
            if isinstance(partial.get(name), torch.Tensor):
                result[name] = partial[name].detach()
        return result

    def retain_failure(index, record, error):
        gate = dict(layer=index, passed=False, failure_type=type(error).__name__, message=str(error),
                    numerical=record.get('numeric_gates', {}), local=record.get('local', {}).get('gates', {}),
                    partition_o_count_scope=state['partition_o_count_scope'])
        state['layer_gates'].append(gate)
        state['gates'] = dict(passed=False, failed_layer=index)
        on_layer(index, raw(record))
        records.pop(index, None)
        raise CaptureFailure(str(error), index, gate) from error

    def layer_pre(index, width):
        def hook(module, args):
            if not active():
                return
            if index != len(state['completed_layers']) or index in records:
                raise RuntimeError('Native layers must execute exactly once in ascending order')
            records[index] = dict(layer_input=rows(args[0], 'layer input', width),
                query_positions=selected.detach(), key_groups=groups.detach())
        return hook

    def v_hook(index, kv, width):
        def hook(module, args, output):
            if active():
                try:
                    records[index]['native_v'] = (output.reshape(1,n,kv,width).transpose(1,2)[0].detach().clone()
                        if isinstance(output,torch.Tensor) and tuple(output.shape)==(1,n,kv*width)
                        else output.detach().clone() if isinstance(output,torch.Tensor) else None)
                    value = tensor(output, 'native V', width=kv*width)
                except Exception as error:
                    retain_failure(index,records[index],error)
        return hook

    def pre_o(index, width):
        def hook(module, args):
            if active():
                try:
                    records[index]['pre_o_av'] = returned_rows(args[0])
                    tensor(args[0],'native pre-O AV',width=width)
                except Exception as error:
                    retain_failure(index,records[index],error)
        return hook

    def attn_hook(index, heads, width):
        def hook(module, args, output):
            if not active():
                return
            record = records[index]
            try:
                if not isinstance(output, tuple) or len(output)!=2 or output[1] is None:
                    raise ValueError('Native eager attention must return its actual softmax; no SDPA fallback')
                record['native_attention_output'] = returned_rows(output[0])
                tensor(output[0],'native attention O',width=width)
                if isinstance(output[1],torch.Tensor):
                    record['native_attention'] = (output[1][0].index_select(1,selected).detach().clone()
                        if tuple(output[1].shape)==(1,heads,n,n) else output[1].detach().clone())
                attention = tensor(output[1], 'native attention')
                if tuple(attention.shape)!=(1,heads,n,n):
                    raise ValueError('Native attention full-input axes changed')
            except Exception as error:
                retain_failure(index, record, error)
        return hook

    def norm_pre(index, width):
        def hook(module, args):
            if active():
                try:
                    records[index]['x_mid'] = returned_rows(args[0])
                    tensor(args[0],'post-attention residual',width=width)
                    counts['native_postattention_norm_calls'] = counts.get('native_postattention_norm_calls',0)+1
                except Exception as error:
                    retain_failure(index,records[index],error)
        return hook

    def norm_hook(index, width):
        def hook(module, args, output):
            if active():
                try:
                    record = records[index]
                    record['native_norm_out'] = returned_rows(output)
                    tensor(output,'native post-attention RMSNorm output',width=width)
                except Exception as error:
                    retain_failure(index,records[index],error)
        return hook

    def mlp_pre(index, width):
        def hook(module, args):
            if active():
                actual = rows(args[0], 'native MLP input', width)
                if not _same_bits(actual, records[index]['native_norm_out']):
                    retain_failure(index, records[index], RuntimeError('Native MLP input differs from captured norm output'))
                counts['native_mlp_calls'] = counts.get('native_mlp_calls',0)+1
        return hook

    def mlp_hook(index, layer, heads, kv, head_width, width):
        def hook(module, args, output):
            nonlocal replay
            if not active():
                return
            record = records[index]
            try:
                record['native_mlp_write'] = returned_rows(output)
                tensor(output,'native MLP output',width=width)
                if any(parameter.is_meta for parameter in layer.parameters()):
                    raise RuntimeError('Local F requires the still-resident native parent layer')
                attention, values = record['native_attention'], record['native_v']
                future = attention.masked_select(~allowed[None]).count_nonzero().item()
                numeric = dict(future_attention=dict(passed=future==0, nonzero_coordinates=int(future)),
                    x_mid_add=dict(passed=_same_bits(record['layer_input']+record['native_attention_output'],record['x_mid'])))
                counts['av_replays'] = counts.get('av_replays',0)+1
                repeated = values.repeat_interleave(heads//kv, dim=0)
                av = torch.matmul(attention,repeated).transpose(0,1).reshape(1,len(positions),width)
                record['replayed_av'] = av.detach()
                numeric['native_av'] = _coordinate_gate(av,record['pre_o_av'],.002,.0001)
                counts['partition_o_calls'] = counts.get('partition_o_calls',0)+3
                writes = partition_writes(attention,values,layer.self_attn.o_proj.weight,groups)
                for name, value in zip(('source_write','history_write','other_write'),writes,strict=True):
                    record[name] = value[None].detach()
                numeric['partition_o'] = _coordinate_gate(writes.sum(0)[None],record['native_attention_output'],.002,.0001)
                record['numeric_gates'] = numeric
                if not all(value['passed'] for value in numeric.values()):
                    raise RuntimeError('Native AV/partition/causal/residual gate failed')
                # All local Norm/MLP hooks must ignore this nested computation.
                replay = True
                try:
                    record['local'] = local_effect(layer,record['x_mid'],record['source_write'],
                        record['native_norm_out'],record['native_mlp_write'],index,counts)
                finally:
                    replay = False
            except ReplayFailure as error:
                record['local'] = error.partial
                retain_failure(index, record, error)
            except Exception as error:
                retain_failure(index, record, error)
        return hook

    def layer_out(index, width):
        def hook(module, args, output):
            if not active():
                return
            record = records[index]
            callback_attempted = False
            try:
                value = output[0] if isinstance(output,tuple) else output
                record['layer_output'] = returned_rows(value)
                tensor(value,'native decoder output',width=width)
                numeric = record['numeric_gates']
                numeric['layer_output_add'] = dict(passed=_same_bits(
                    record['x_mid']+record['native_mlp_write'],record['layer_output']))
                gate = dict(layer=index, numerical=numeric,local=record['local']['gates'],
                    partition_o_count_scope=state['partition_o_count_scope'])
                gate['passed'] = all(v['passed'] for v in numeric.values()) and gate['local']['passed']
                state['layer_gates'].append(gate)
                callback_attempted = True
                on_layer(index,raw(record))  # Caller streams raw BEFORE failure/accumulation.
                if not gate['passed']:
                    state['gates'] = dict(passed=False,failed_layer=index)
                    records.pop(index,None)
                    raise CaptureFailure('Layer numeric gate failed after raw retention',index,gate)
                old_bytes = accumulator.cpu_transfer_bytes
                accumulator.add(index,record['local']['delta_source'],record['local']['delta_sham'],record['source_write'])
                counts['host_transfer_bytes'] = counts.get('host_transfer_bytes',0)+accumulator.cpu_transfer_bytes-old_bytes
                state['accumulator_cpu_reference_bytes'] = 2*len(positions)*width*8
                state['completed_layers'].append(index)
                records.pop(index,None)
            except CaptureFailure:
                raise
            except Exception as error:
                if not callback_attempted:
                    retain_failure(index,record,error)
                # Raw was already retained if accumulation or callback failed.
                state['gates'] = dict(passed=False,failed_layer=index,error=str(error))
                raise CaptureFailure(str(error),index,state['gates']) from error
        return hook

    def final_pre(module,args):
        if active():
            state['H_raw'] = args[0].detach().clone() if isinstance(args[0],torch.Tensor) else None
            tensor(args[0],'native full finalnorm input',width=layers[0].self_attn.o_proj.out_features)

    def final_out(module,args,output):
        if active():
            state['native_postnorm'] = output.detach().clone() if isinstance(output,torch.Tensor) else None
            tensor(output,'native full finalnorm output',width=layers[0].self_attn.o_proj.out_features)

    try:
        handles.append(observer.native.model.register_forward_pre_hook(model_pre,with_kwargs=True))
        for index, layer in enumerate(layers):
            heads,kv,head_width = observer.head_layout(index)
            width = heads*head_width
            if heads%kv or getattr(layer.self_attn.o_proj,'bias',None) is not None:
                raise ValueError('Native bias-free contiguous-GQA layout required')
            handles.extend((layer.register_forward_pre_hook(layer_pre(index,width)),
                layer.self_attn.v_proj.register_forward_hook(v_hook(index,kv,head_width)),
                layer.self_attn.o_proj.register_forward_pre_hook(pre_o(index,width)),
                layer.self_attn.register_forward_hook(attn_hook(index,heads,width)),
                layer.post_attention_layernorm.register_forward_pre_hook(norm_pre(index,width)),
                layer.post_attention_layernorm.register_forward_hook(norm_hook(index,width)),
                layer.mlp.register_forward_pre_hook(mlp_pre(index,width)),
                layer.mlp.register_forward_hook(mlp_hook(index,layer,heads,kv,head_width,width)),
                layer.register_forward_hook(layer_out(index,width))))
        handles.append(observer.native.model.norm.register_forward_pre_hook(final_pre))
        handles.append(observer.native.model.norm.register_forward_hook(final_out))
        yield state
        if (state['completed_layers']!=list(range(len(layers))) or state['H_raw'] is None
                or state['native_postnorm'] is None):
            raise CaptureFailure('Native capture did not complete every layer/finalnorm',-1,state['gates'])
        old_bytes = accumulator.cpu_transfer_bytes
        with torch.inference_mode():
            state['accumulation'] = accumulator.finish()
        counts['host_transfer_bytes'] = counts.get('host_transfer_bytes',0)+accumulator.cpu_transfer_bytes-old_bytes
        state['gates'] = dict(passed=state['accumulation']['gates']['passed']
            and all(item['passed'] for item in state['layer_gates']),
            layers=len(layers), accumulation=state['accumulation']['gates'])
    finally:
        replay = False
        for handle in reversed(handles):
            handle.remove()
        records.clear()
        del accumulator  # Release the two CPU FP64 audit sums before caller heads.
        observer._operator_capture_active = False
