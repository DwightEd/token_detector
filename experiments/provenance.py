"""Small, shared source/runtime identity for retained experiment outputs."""

import importlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform

import numpy as np

from .data import file_hash


def research_identity():
    """Hash installed project modules/resources and record the numerical runtime."""
    code = {}
    distribution_root = Path(__file__).resolve().parents[1]
    for name in ['myproject', 'experiments', 'experiments.research']:
        location = importlib.import_module(name).__file__
        expected = distribution_root.joinpath(*name.split('.'))
        if location is None or Path(location).resolve().parent != expected:
            raise RuntimeError('Project imports are shadowed by an unrelated package; '
                               'run from token_detector or a clean installed working directory')
        root = Path(location).parent
        paths = list(root.glob('*.py')) + list((root / 'references').glob('*.npz'))
        code.update({f'{name}/{p.relative_to(root)}': file_hash(p) for p in sorted(paths)})
    dependencies = {'numpy': np.__version__}
    for name in ['scipy', 'torch', 'transformers']:
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        'source_sha256': code, 'python': platform.python_version(),
        'dependencies': dependencies,
        'threads': {name: os.environ.get(name) for name in
                    ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'numpy_build': np.show_config(mode='dicts'),
    }


def completed_identity(start):
    if research_identity() != start:
        raise RuntimeError('Source or numerical runtime changed during the research run')
    return start
