"""Marginal-preserving erasure of one independent head--carrier association.

This NumPy kernel consumes captured operands. It loads no files, parameters or
models, and emits no risk or semantic verdict. Each batch item is a separate
two-head/two-carrier experiment; overlapping head pairs must not be added into
one intervention. The caller establishes the same query and shared GQA KV group.

Production float32 operands are promoted to canonical float64 arithmetic. The
output describes a local attention-message change, not a BF16 forward or a
complete downstream counterfactual. See the frozen MINIMAL013351 contract.
"""

import numpy as np


U64 = np.float64(2.0**-53)
_K = 2


def _operands(attention, values, all_visible):
    a, v = np.asarray(attention), np.asarray(values)
    if a.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError('Captured attention and values must have float32 dtype')
    if a.ndim < 2 or a.shape[-2:] != (2, 2) or not a.size:
        raise ValueError('Attention must be a nonempty [..., 2, 2] array')
    if v.ndim < 2 or v.shape[-2] != 2 or v.shape[-1] == 0 or not v.size:
        raise ValueError('Shared values must be a nonempty [..., 2, D] array')
    if not np.isfinite(a).all() or not np.isfinite(v).all():
        raise ValueError('Captured operands must be finite')
    if (a < 0).any():
        raise ValueError('Attention weights must be nonnegative')
    visible = np.asarray(all_visible)
    if visible.dtype != np.bool_ or visible.shape not in ((), a.shape):
        raise ValueError('all_visible must be a bool or an attention-shaped bool mask')
    if not visible.all():
        raise ValueError('Every carrier must be causally visible to both heads')
    try:
        batch = np.broadcast_shapes(a.shape[:-2], v.shape[:-2])
    except ValueError as error:
        raise ValueError('Attention/value batch dimensions are not broadcastable') from error
    # Preserve zero-stride shared value axes instead of expanding Q copies before
    # the float64 cast. The returned difference retains these compact dimensions.
    index = tuple(slice(0, 1) if n > 1 and stride == 0 else slice(None)
                  for n, stride in zip(v.shape[:-2], v.strides[:-2], strict=True))
    compact_v = v[index + (slice(None), slice(None))]
    a64 = np.broadcast_to(a.astype(np.float64), batch + (2, 2))
    v64 = compact_v.astype(np.float64)
    return a64, v64, batch


def erase_head_carrier_association(attention, values, *, all_visible):
    """Erase one 2x2 association per broadcast batch, preserving both marginals.

    ``attention[..., head, carrier]`` and shared ``values[..., carrier, D]``
    must be float32. ``all_visible`` is explicitly True, or a same-shaped bool
    mask containing only True. Physical query/head/KV identities belong to the
    caller; a false visibility mask is rejected even for a zero-mass block.

    Returns a plain dict. ``gates`` contains global booleans; ``tuple_passed``
    retains the per-batch result. Numerical gate failure is returned with raw
    diagnostics, so orchestration can save them before stopping. Invalid inputs
    raise before arithmetic. MI/chi use zero sentinels with explicit defined
    flags, never epsilon smoothing or clipping. Finite negative MI roundoff is
    retained rather than clamped to zero.

    The erased-minus-native pre-O messages are reconstructed from
    ``delta_coefficients[..., :, None] * value_difference[..., None, :]``.
    ``value_difference`` preserves the compact, broadcastable value batch axes.
    No [..., 2, D] delta is retained in the result. Caller-side chunking bounds
    temporary direct-AV comparisons, which require O(batch_size * D) memory.
    """
    a, v, batch = _operands(attention, values, all_visible)
    rows, columns = a.sum(axis=-1), a.sum(axis=-2)
    mass = rows.sum(axis=-1)
    zero = mass == 0
    erased = np.zeros(batch + (2, 2), dtype=np.float64)
    np.divide(rows[..., :, None] * columns[..., None, :], mass[..., None, None],
              out=erased, where=~zero[..., None, None])
    erased = np.where(zero[..., None, None], a, erased)
    determinant = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    kappa = np.zeros(batch, dtype=np.float64)
    np.divide(determinant, mass, out=kappa, where=~zero)
    coefficients = np.stack((-kappa, kappa), axis=-1)
    difference = v[..., 0, :] - v[..., 1, :]

    # Standard conditional mutual information of the normalized 2x2 table.
    p, independent = np.zeros_like(a), np.zeros_like(erased)
    np.divide(a, mass[..., None, None], out=p, where=~zero[..., None, None])
    np.divide(erased, mass[..., None, None], out=independent,
              where=~zero[..., None, None])
    logp, log_independent = np.zeros_like(p), np.zeros_like(independent)
    np.log(p, out=logp, where=p > 0)
    np.log(independent, out=log_independent, where=independent > 0)
    mi = (p * (logp - log_independent)).sum(axis=(-2, -1))
    chi_defined = (a > 0).all(axis=(-2, -1))
    loga = np.zeros_like(a)
    np.log(a, out=loga, where=a > 0)
    chi = (loga[..., 0, 0] + loga[..., 1, 1]
           - loga[..., 0, 1] - loga[..., 1, 0])
    chi = np.where(chi_defined, chi, np.float64(0))

    row_error = np.abs(erased.sum(axis=-1) - rows)
    column_error = np.abs(erased.sum(axis=-2) - columns)
    marginal_tolerance = np.float64(64 * _K) * U64 * mass
    row_ok = (row_error <= marginal_tolerance[..., None]).all(axis=-1)
    column_ok = (column_error <= marginal_tolerance[..., None]).all(axis=-1)

    # Independent direct AV subtraction is compared with the factorized closed
    # form. Keep only per-head witnesses/counts, not all large per-coordinate deltas.
    max_error = np.empty(batch + (2,), dtype=np.float64)
    max_tolerance = np.empty_like(max_error)
    failure_count = np.empty(batch + (2,), dtype=np.int64)
    v0, v1 = v[..., 0, :], v[..., 1, :]
    for head in range(2):
        native = a[..., head, 0, None] * v0 + a[..., head, 1, None] * v1
        changed = (erased[..., head, 0, None] * v0
                   + erased[..., head, 1, None] * v1)
        direct = changed - native
        expected = coefficients[..., head, None] * difference
        error = np.abs(direct - expected)
        scale = ((a[..., head, 0, None] + erased[..., head, 0, None]) * np.abs(v0)
                 + (a[..., head, 1, None] + erased[..., head, 1, None]) * np.abs(v1))
        tolerance = np.float64(256 * _K) * U64 * scale
        max_error[..., head] = error.max(axis=-1)
        max_tolerance[..., head] = tolerance.max(axis=-1)
        failure_count[..., head] = np.count_nonzero(error > tolerance, axis=-1)
    direct_ok = (failure_count == 0).all(axis=-1)

    single_norm = np.abs(kappa) * np.linalg.norm(difference, axis=-1)
    head_norms = np.broadcast_to(single_norm[..., None], batch + (2,)).copy()
    concat_norm = np.sqrt(np.float64(2)) * single_norm
    nonnegative = (erased >= 0).all(axis=(-2, -1))
    finite = ((np.isfinite(erased).all(axis=(-2, -1)))
              & np.isfinite(kappa) & np.isfinite(mi) & np.isfinite(chi)
              & np.isfinite(concat_norm)
              & np.isfinite(max_error).all(axis=-1)
              & np.isfinite(max_tolerance).all(axis=-1))
    tuple_passed = finite & nonnegative & row_ok & column_ok & direct_ok
    gates = dict(finite=bool(finite.all()), nonnegative_erased=bool(nonnegative.all()),
                 row_marginals=bool(row_ok.all()), column_marginals=bool(column_ok.all()),
                 direct_av=bool(direct_ok.all()), passed=bool(tuple_passed.all()))
    return dict(native_attention=a, erased_attention=erased, mass=mass,
                row_masses=rows, column_masses=columns, kappa=kappa,
                mi=mi, mi_defined=~zero, chi=chi, chi_defined=chi_defined,
                zero_mass=zero, delta_coefficients=coefficients,
                value_difference=difference, head_norms=head_norms,
                concat_norm=concat_norm, row_error=row_error,
                column_error=column_error, marginal_tolerance=marginal_tolerance,
                max_direct_av_error=max_error, max_direct_av_tolerance=max_tolerance,
                direct_av_failure_count=failure_count, tuple_passed=tuple_passed,
                gates=gates, passed=gates['passed'])
