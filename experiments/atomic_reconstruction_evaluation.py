"""One-conjunct qualification through the common independent CPU reader.

The primary matched comparison is the sealed H011 burden0 packet, preserved at
publication without changing these frozen predictions. Suffix duplicates are
compiler reuse, not new evidence of empirical burden invariance.
"""
from . import reconstruction_evaluation as shared
from . import atomic_reconstruction_qualification as observer

DEFAULT_PROTOCOL = dict(shared.DEFAULT_PROTOCOL, schema='atomic-reconstruction-evaluation-v1',
    purpose='known_single_conjunct_source_reconstruction_qualification',
    primary_control='sealed_H011_burden_zero_only', dangling_other_instruction_retained=True,
    non_target_conjunct_bundle_removed=True, suffix_invariance_is_by_construction=True,
    unique_internal_mechanism_claim=False)
_HELPERS = shared._HELPERS + ('experiments/masked_reconstruction_qualification.py',
    'experiments.research/masked_reconstruction.py', 'experiments/atomic_reconstruction_qualification.py',
    'experiments.research/atomic_reconstruction.py')


def evaluate(inputs, design, construction, observations, model_path, output, protocol):
    return shared.evaluate(inputs, design, construction, observations, model_path, output, protocol,
        profile=observer, evaluation_protocol=DEFAULT_PROTOCOL, helpers=_HELPERS)
