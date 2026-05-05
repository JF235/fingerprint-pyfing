"""PyTorch backend for pyfing.

This submodule is intentionally separate from the default Keras-backed API.
"""

from importlib import import_module

_EXPORTS = {
    "fingerprint_segmentation": ("pyfing.pytorch.simple_api", "fingerprint_segmentation"),
    "orientation_field_estimation": ("pyfing.pytorch.simple_api", "orientation_field_estimation"),
    "frequency_estimation": ("pyfing.pytorch.simple_api", "frequency_estimation"),
    "fingerprint_enhancement": ("pyfing.pytorch.simple_api", "fingerprint_enhancement"),
    "minutiae_extraction": ("pyfing.pytorch.simple_api", "minutiae_extraction"),
    "SufsTorch": ("pyfing.pytorch.algorithms", "SufsTorch"),
    "SnfoeTorch": ("pyfing.pytorch.algorithms", "SnfoeTorch"),
    "SnffeTorch": ("pyfing.pytorch.algorithms", "SnffeTorch"),
    "SnfenTorch": ("pyfing.pytorch.algorithms", "SnfenTorch"),
    "LeaderTorch": ("pyfing.pytorch.algorithms", "LeaderTorch"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
