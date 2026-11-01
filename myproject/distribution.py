"""Conditional predictive calibration; no factuality calibration is assumed."""

import numpy as np
import torch


DISTRIBUTION_FIELDS = ('calibrated_gap', 'rank_gap', 'source_tv', 'source_tail_logp', 'distribution_gap',
                       'bridge_gap', 'source_nll', 'source_entropy')


def log_calibrator(log_u):
    """Log integral_0^1 a*u**(a-1) da, stable at u=1 and tiny u."""
    length = -log_u
    small = torch.log(.5 + length/6 + length.square()/24
                      + length.pow(3)/120 + length.pow(4)/720)
    safe = length.clamp_min(1e-3)
    large = safe + torch.log1p(-(1+safe)*torch.exp(-safe)) - 2*torch.log(safe)
    return torch.where(length < 1e-3, small, large)


def conditional_observations(source_logits, absent_logits, targets):
    """Reduce full-vocabulary logits to target scores, retaining complete ties.

    Q_rank(v) = p_source(v) h(u(v)) / E_source[h(u)], where u is the source-law
    upper-tail mass of log(p_absent/p_source). The final alternative mixes
    Q_rank with p_source using TV(p_source,p_absent), so vanishing source
    effects yield vanishing evidence at each fixed positive-probability atom.
    h is the Vovk--Wang (2021), Eq. (2), p-to-e mixture calibrator.
    All vocabulary atoms participate.
    Double precision normalizes both worlds consistently, independently of the
    historical FP32 target-likelihood arithmetic used by the other readouts.
    """
    if (source_logits.ndim != 2 or source_logits.shape != absent_logits.shape
            or source_logits.shape[1] == 0):
        raise ValueError('Expected matching nonempty vocabulary matrices')
    targets = torch.as_tensor(targets, device=source_logits.device)
    if (targets.shape != (len(source_logits),) or targets.dtype != torch.long
            or torch.any(targets < 0) or torch.any(targets >= source_logits.shape[1])):
        raise ValueError('Invalid target token axis')
    if not torch.isfinite(source_logits).all() or not torch.isfinite(absent_logits).all():
        raise ValueError('Logits must be finite')
    log_s = source_logits.double().log_softmax(-1)
    log_0 = absent_logits.double().log_softmax(-1)
    gap = log_0-log_s
    # The log-normalizer difference is constant across vocabulary atoms.
    # Order native differences directly: subtracting normalized log-probabilities
    # separately can split exact BF16/FP32 ties by a few FP64 rounding units.
    ordering = absent_logits.double()-source_logits.double()
    ordered, order = ordering.sort(dim=-1, descending=True, stable=True)
    sorted_s = log_s.gather(1, order)
    log_tail = sorted_s.logcumsumexp(-1)
    # Each tie receives the cumulative mass through its last member.
    end = torch.cat((ordered[:, 1:] != ordered[:, :-1],
                     torch.ones_like(ordered[:, :1], dtype=torch.bool)), dim=1)
    ranks = torch.arange(gap.shape[1], device=gap.device).expand_as(order)
    ends = torch.where(end, ranks, gap.shape[1]-1).flip(-1).cummin(-1).values.flip(-1)
    log_tail = log_tail.gather(1, ends).clamp_max(0)
    log_h = log_calibrator(log_tail)
    log_z = (sorted_s+log_h).logsumexp(-1)
    observed_rank = (order == targets[:, None]).long().argmax(-1)[:, None]
    rank_gap = log_h.gather(1, observed_rank)[:, 0]-log_z
    tv = (.5*(log_s.exp()-log_0.exp()).abs().sum(-1)).clamp(0, 1)
    gather = lambda array: array.gather(1, targets[:, None])[:, 0]
    return {
        'calibrated_gap': torch.logaddexp(torch.log1p(-tv), tv.log()+rank_gap),
        'rank_gap': rank_gap,
        'source_tv': tv,
        'source_tail_logp': log_tail.gather(1, observed_rank)[:, 0],
        'distribution_gap': gather(gap),
        'bridge_gap': .5*gather(gap)-(.5*(log_s+log_0)).logsumexp(-1),
        'source_nll': -gather(log_s),
        'source_entropy': -(log_s.exp()*log_s).sum(-1),
    }


class SourceDistributions:
    """Hold one answer's source logits on CPU; reduce when absent rows arrive."""

    def __init__(self, targets):
        self.targets = targets
        self.source = {}
        self.parts = {key: [] for key in DISTRIBUTION_FIELDS}
        self.present_rows = self.absent_rows = 0

    def present(self, start, logits):
        if start != self.present_rows or start+len(logits) > len(self.targets):
            raise ValueError('Source distribution row schedule changed')
        self.source[start] = logits.detach().cpu()
        self.present_rows += len(logits)

    def absent(self, start, logits):
        if start != self.absent_rows or start not in self.source:
            raise ValueError('Absent distribution row schedule changed')
        source = self.source.pop(start)
        if source.shape != logits.shape:
            raise ValueError('Source/absent distribution chunks differ')
        result = conditional_observations(source.to(logits.device), logits,
                                          self.targets[start:start+len(logits)])
        for key, value in result.items():
            self.parts[key].append(value.detach().cpu().numpy())
        self.absent_rows += len(logits)

    def finish(self):
        if self.source or self.present_rows != self.absent_rows or self.absent_rows != len(self.targets):
            raise ValueError('Incomplete source/absent distribution observation')
        return {key: np.concatenate(parts) for key, parts in self.parts.items()}
