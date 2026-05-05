"""pyfing - Fingerprint recognition in Python."""

import os
from importlib import import_module

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

__version__ = "0.6"

_EXPORTS = {
    "fingerprint_segmentation": ("pyfing.simple_api", "fingerprint_segmentation"),
    "orientation_field_estimation": ("pyfing.simple_api", "orientation_field_estimation"),
    "frequency_estimation": ("pyfing.simple_api", "frequency_estimation"),
    "fingerprint_enhancement": ("pyfing.simple_api", "fingerprint_enhancement"),
    "minutiae_extraction": ("pyfing.simple_api", "minutiae_extraction"),
    "SegmentationAlgorithm": ("pyfing._interfaces", "SegmentationAlgorithm"),
    "SegmentationParameters": ("pyfing._interfaces", "SegmentationParameters"),
    "Gmfs": ("pyfing.segmentation", "Gmfs"),
    "GmfsParameters": ("pyfing._interfaces", "GmfsParameters"),
    "Sufs": ("pyfing.segmentation", "Sufs"),
    "SufsParameters": ("pyfing._interfaces", "SufsParameters"),
    "OrientationEstimationAlgorithm": ("pyfing._interfaces", "OrientationEstimationAlgorithm"),
    "OrientationEstimationParameters": ("pyfing._interfaces", "OrientationEstimationParameters"),
    "Gbfoe": ("pyfing.orientations", "Gbfoe"),
    "GbfoeParameters": ("pyfing._interfaces", "GbfoeParameters"),
    "Snfoe": ("pyfing.orientations", "Snfoe"),
    "SnfoeParameters": ("pyfing._interfaces", "SnfoeParameters"),
    "FrequencyEstimationAlgorithm": ("pyfing._interfaces", "FrequencyEstimationAlgorithm"),
    "FrequencyEstimationParameters": ("pyfing._interfaces", "FrequencyEstimationParameters"),
    "Xsffe": ("pyfing.frequencies", "Xsffe"),
    "XsffeParameters": ("pyfing._interfaces", "XsffeParameters"),
    "Snffe": ("pyfing.frequencies", "Snffe"),
    "SnffeParameters": ("pyfing._interfaces", "SnffeParameters"),
    "EnhancementAlgorithm": ("pyfing._interfaces", "EnhancementAlgorithm"),
    "EnhancementParameters": ("pyfing._interfaces", "EnhancementParameters"),
    "Gbfen": ("pyfing.enhancement", "Gbfen"),
    "GbfenParameters": ("pyfing._interfaces", "GbfenParameters"),
    "Snfen": ("pyfing.enhancement", "Snfen"),
    "SnfenParameters": ("pyfing._interfaces", "SnfenParameters"),
    "EndToEndMinutiaExtractionAlgorithm": ("pyfing._interfaces", "EndToEndMinutiaExtractionAlgorithm"),
    "EndToEndMinutiaExtractionParameters": ("pyfing._interfaces", "EndToEndMinutiaExtractionParameters"),
    "Leader": ("pyfing.minutiae", "Leader"),
    "LeaderParameters": ("pyfing._interfaces", "LeaderParameters"),
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
