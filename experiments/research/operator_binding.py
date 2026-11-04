"""Frozen H018 source-write -> Norm+MLP readout, without model loading or I/O.

This is a held-native-input surrogate, not a propagated counterfactual world or
truth estimator.  Every original layer contributes; attention and MLP do not
have an assumed copy/truth division.  The local reference and counter branches
use the same [1, T, H] shape and must qualify against the captured native rows.
"""

from collections.abc import MutableMapping

from .relation_intervention import matched_noise


class ReplayFailure(RuntimeError):
    """Keep successful local results if a later module call cannot complete."""

    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def _context():
    import torch

    if torch.is_grad_enabled() or any(torch.is_autocast_enabled(device)
                                      for device in ('cpu', 'cuda')):
        raise RuntimeError('H018 requires no-grad/inference mode and no autocast')


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _tensor(value, name, *, shape=None, dtype=None, device=None):
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} has a different shape')
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f'{name} has a different dtype')
    if device is not None and value.device != device:
        raise ValueError(f'{name} is on a different device')
    if not torch.isfinite(value).all().item():
        raise ValueError(f'{name} contains nonfinite values')


def _same_bits(left, right):
    import torch

    if (left.shape != right.shape or left.dtype != right.dtype
            or left.device != right.device):
        return False
    integer = {torch.float32: torch.int32, torch.float64: torch.int64}[left.dtype]
    return torch.equal(left.contiguous().view(integer), right.contiguous().view(integer))


def _coordinate_gate(actual, reference, absolute, relative):
    import torch

    finite = bool(torch.isfinite(actual).all().item()
                  and torch.isfinite(reference).all().item())
    if not finite:
        return dict(passed=False, finite=False, max_abs_error=None,
                    max_error_over_tolerance=None, violating_coordinates=None)
    error = (actual - reference).abs()
    tolerance = absolute + relative * reference.abs()
    if not torch.isfinite(error).all().item() or not torch.isfinite(tolerance).all().item():
        return dict(passed=False, finite=False, max_abs_error=None,
                    max_error_over_tolerance=None, violating_coordinates=None)
    return dict(passed=bool((error <= tolerance).all().item()), finite=True,
                max_abs_error=float(error.max().item()),
                max_error_over_tolerance=float((error / tolerance).max().item()),
                violating_coordinates=int((error > tolerance).sum().item()))


def _counts(counts):
    if not isinstance(counts, MutableMapping):
        raise TypeError('counts must be a mutable mapping')
    for name in ('extra_norm_calls', 'extra_norm_rows',
                 'extra_mlp_calls', 'extra_mlp_rows'):
        _integer(counts.get(name, 0), name)


def local_effect(layer, x_mid_T, source_write, native_norm_out, native_mlp_out,
                 layer_index, counts):
    """Run reference/source/sham/m0 in that order, exactly four local F calls.

    Inputs are FP32 [1,T,H] on one device. ``native_norm_out`` is the captured
    *post-attention RMSNorm output / native MLP input*, not x_mid.  Numerical
    gate failures return all raw results plus ``gates['passed']=False`` so the
    caller can persist them before failing. Interface errors run no modules;
    exceptional/invalid module results raise ReplayFailure with ``partial``.
    Caller owns native capture/replay-hook guards, persistence and resource
    admission. Counts increment immediately before each attempted module call.
    """
    import torch

    _context()
    _integer(layer_index, 'layer_index')
    if layer_index >= 32:
        raise ValueError('Original observer layer_index must be in 0..31')
    _counts(counts)
    _tensor(x_mid_T, 'x_mid_T', dtype=torch.float32)
    if x_mid_T.ndim != 3 or x_mid_T.shape[0] != 1 or min(x_mid_T.shape[1:]) < 1:
        raise ValueError('Local replay requires nonempty [1,T,H] tensors')
    shape, device = x_mid_T.shape, x_mid_T.device
    for name, value in (('source_write', source_write),
                        ('native_norm_out', native_norm_out),
                        ('native_mlp_out', native_mlp_out)):
        _tensor(value, name, shape=shape, dtype=torch.float32, device=device)
    target_norm = torch.linalg.vector_norm(source_write, dim=-1, keepdim=True)
    _tensor(target_norm, 'source norm', dtype=torch.float32)
    sham = matched_noise(source_write, seed=17 + layer_index)
    _tensor(sham, 'sham', shape=shape, dtype=torch.float32, device=device)
    sham_norm = torch.linalg.vector_norm(sham, dim=-1, keepdim=True)
    _tensor(sham_norm, 'sham norm', dtype=torch.float32)
    if ((target_norm > 0) & (sham_norm == 0)).any().item():
        raise ValueError('Nonzero source norm has zero sham norm')
    if ((target_norm == 0) & (sham_norm != 0)).any().item():
        raise ValueError('Zero source norm has a nonzero sham')
    inputs = dict(reference=x_mid_T.detach().clone().contiguous(),
                  source=(x_mid_T - source_write).contiguous(),
                  sham=(x_mid_T - sham).contiguous(),
                  m0=(x_mid_T - torch.zeros_like(source_write)).contiguous())
    for name, value in inputs.items():
        _tensor(value, f'{name} input', shape=shape, dtype=torch.float32, device=device)
    result = dict(layer_index=layer_index, inputs=inputs, norms={}, mlps={},
                  sham_write=sham)
    rows = shape[1]
    for branch, value in inputs.items():
        try:
            counts['extra_norm_calls'] = counts.get('extra_norm_calls', 0) + 1
            counts['extra_norm_rows'] = counts.get('extra_norm_rows', 0) + rows
            norm = layer.post_attention_layernorm(value)
            result['norms'][branch] = norm
            _tensor(norm, f'{branch} norm', shape=shape, dtype=torch.float32, device=device)
            counts['extra_mlp_calls'] = counts.get('extra_mlp_calls', 0) + 1
            counts['extra_mlp_rows'] = counts.get('extra_mlp_rows', 0) + rows
            mlp = layer.mlp(norm)
            result['mlps'][branch] = mlp
            _tensor(mlp, f'{branch} MLP', shape=shape, dtype=torch.float32, device=device)
        except Exception as error:
            raise ReplayFailure(f'Local {branch} replay failed: {error}', result) from error
    norms, mlps = result['norms'], result['mlps']
    result['delta_source'] = mlps['reference'] - mlps['source']
    result['delta_sham'] = mlps['reference'] - mlps['sham']
    gates = dict(reference_norm=_coordinate_gate(norms['reference'], native_norm_out, 1e-6, 1e-5),
                 reference_mlp=_coordinate_gate(mlps['reference'], native_mlp_out, 2e-5, 2e-4),
                 m0_norm=dict(passed=_same_bits(norms['m0'], norms['reference'])),
                 m0_mlp=dict(passed=_same_bits(mlps['m0'], mlps['reference'])),
                 m0_delta=dict(passed=bool(((mlps['reference'] - mlps['m0']) == 0).all().item())),
                 sham_norm=_coordinate_gate(sham_norm, target_norm, 1e-6, 1e-5),
                 delta_source_finite=dict(passed=bool(torch.isfinite(result['delta_source']).all().item())),
                 delta_sham_finite=dict(passed=bool(torch.isfinite(result['delta_sham']).all().item())))
    gates['passed'] = all(gate['passed'] for gate in gates.values())
    result['gates'] = gates
    return result


class OrderedAccumulator:
    """Streaming FP64 source/sham deltas, FP32 P006 writes, fixed layer order.

    CPU FP64 references are updated one layer at a time; full layer tensors are
    not retained. Actual experiments must pass expected_layers=32. Smaller
    counts are available for synthetic engineering tests only.
    """

    def __init__(self, expected_layers=32):
        _integer(expected_layers, 'expected_layers', minimum=1)
        self.expected_layers = expected_layers
        self.next_layer = 0
        self._finished = False
        self._sums = None
        self._cpu = None
        self.cpu_transfer_bytes = 0

    def add(self, layer_index, delta_source, delta_sham, source_write):
        import torch

        _context()
        _integer(layer_index, 'layer_index')
        if self._finished or layer_index != self.next_layer or layer_index >= self.expected_layers:
            raise ValueError('Accumulator requires each layer once in ascending order')
        _tensor(delta_source, 'delta_source', dtype=torch.float32)
        if delta_source.ndim != 3 or delta_source.shape[0] != 1 or min(delta_source.shape[1:]) < 1:
            raise ValueError('Accumulator requires nonempty [1,T,H] tensors')
        shape, device = delta_source.shape, delta_source.device
        for name, value in (('delta_sham', delta_sham), ('source_write', source_write)):
            _tensor(value, name, shape=shape, dtype=torch.float32, device=device)
        if self._sums is None:
            self._sums = dict(source=torch.zeros(shape, device=device, dtype=torch.float64),
                              sham=torch.zeros(shape, device=device, dtype=torch.float64),
                              p006=torch.zeros(shape, device=device, dtype=torch.float32))
            self._cpu = dict(source=torch.zeros(shape, dtype=torch.float64),
                             sham=torch.zeros(shape, dtype=torch.float64))
        _tensor(delta_source, 'delta_source', shape=self._sums['source'].shape,
                dtype=torch.float32, device=self._sums['source'].device)
        for name, value in (('source', delta_source), ('sham', delta_sham)):
            self._sums[name].add_(value.to(dtype=torch.float64))
            self._cpu[name].add_(value.detach().cpu().to(dtype=torch.float64))
            if value.device.type != 'cpu':
                self.cpu_transfer_bytes += value.numel() * value.element_size()
        self._sums['p006'].add_(source_write)
        self.next_layer += 1

    def finish(self):
        import torch

        _context()
        if self._finished or self.next_layer != self.expected_layers:
            raise ValueError('Accumulator is incomplete or has already finished')
        self._finished = True
        gates = {}
        for name in ('source', 'sham'):
            actual = self._sums[name].detach().cpu()
            if self._sums[name].device.type != 'cpu':
                self.cpu_transfer_bytes += actual.numel() * actual.element_size()
            gates[name] = dict(passed=_same_bits(actual, self._cpu[name]))
        source32 = self._sums['source'].to(dtype=torch.float32)
        sham32 = self._sums['sham'].to(dtype=torch.float32)
        gates['finite'] = dict(passed=all(bool(torch.isfinite(value).all().item())
                                        for value in (*self._sums.values(), source32, sham32)))
        gates['passed'] = all(gate['passed'] for gate in gates.values())
        return dict(source64=self._sums['source'], sham64=self._sums['sham'],
                    source32=source32, sham32=sham32,
                    p006_source32=self._sums['p006'], gates=gates,
                    layers=self.next_layer, cpu_transfer_bytes=self.cpu_transfer_bytes)


def full_vocab_endpoints(logits, target_ids):
    """Canonical FP32 z_y - logsumexp(full logits), not log_softmax/FP64.

    Raw complete logits belong to the caller. CPU FP64 audits are references,
    never producers of the method scores.
    """
    import torch

    _context()
    _tensor(logits, 'logits', dtype=torch.float32)
    if logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError('logits must be nonempty [T,V]')
    _tensor(target_ids, 'target_ids', shape=(logits.shape[0],),
            dtype=torch.int64, device=logits.device)
    if ((target_ids < 0) | (target_ids >= logits.shape[1])).any().item():
        raise ValueError('Target ID is outside the full vocabulary')
    target_logit = logits.gather(1, target_ids[:, None])[:, 0]
    lse = torch.logsumexp(logits, dim=-1)
    logp = target_logit - lse
    _tensor(lse, 'full-vocabulary logsumexp', dtype=torch.float32)
    _tensor(logp, 'target logp', dtype=torch.float32)
    return dict(target_logit=target_logit, logsumexp=lse, logp=logp)


def score_endpoints(native, counter):
    """Fixed signed e and sigmoid, retaining FP32 subtraction-order error.

    Decomposition gate = 32*f32eps*(|z_cf|+|z_n|+|LSE_cf|+|LSE_n|+1).
    This frozen operational tolerance is neither a truth cutoff nor a formal
    numerical-error theorem. No fitted sign, layer selection or score fusion.
    """
    import torch

    _context()
    shape = None
    device = None
    for branch, values in (('native', native), ('counter', counter)):
        for key in ('target_logit', 'logsumexp', 'logp'):
            value = values[key]
            _tensor(value, f'{branch} {key}', dtype=torch.float32,
                    shape=shape, device=device)
            if value.ndim != 1 or not value.numel():
                raise ValueError('Endpoint arrays must be nonempty [T]')
            shape, device = value.shape, value.device
    logit_difference = counter['target_logit'] - native['target_logit']
    lse_difference = counter['logsumexp'] - native['logsumexp']
    e = counter['logp'] - native['logp']
    decomposed = logit_difference - lse_difference
    error = (e - decomposed).abs()
    tolerance = 32 * torch.finfo(torch.float32).eps * (
        counter['target_logit'].abs() + native['target_logit'].abs()
        + counter['logsumexp'].abs() + native['logsumexp'].abs() + 1)
    risk = torch.sigmoid(e)
    finite = all(bool(torch.isfinite(value).all().item()) for value in (
        e, risk, logit_difference, lse_difference, decomposed, error, tolerance))
    return dict(e=e, risk=risk, logit_difference=logit_difference,
                lse_difference=lse_difference, decomposed_e=decomposed,
                decomposition_error=error, decomposition_tolerance=tolerance,
                gates=dict(passed=finite and bool((error <= tolerance).all().item()),
                           finite=finite))
