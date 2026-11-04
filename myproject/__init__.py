"""V3 exports; NumPy-only alignment imports do not initialize a model runtime."""

from importlib import import_module

__all__ = ["HallucinationDetector", "Observations", "WitnessDetector"]

_EXPORT_MODULES = {"HallucinationDetector": "api", "Observations": "observations",
                   "WitnessDetector": "witness"}


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("."+_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
