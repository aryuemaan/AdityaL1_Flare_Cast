"""Nowcasting: onset detection, GOES classification, soft/hard fusion."""
from aditya_flarecast.nowcast.catalogue import evaluate_catalogue, run_nowcast
from aditya_flarecast.nowcast.classifier import (
    classify_peak_flux,
    class_rank,
    meets_min_class,
)
from aditya_flarecast.nowcast.detector import FlareEvent, detect_hard, detect_soft
from aditya_flarecast.nowcast.fusion import fuse, to_dataframe

__all__ = [
    "run_nowcast",
    "evaluate_catalogue",
    "FlareEvent",
    "detect_soft",
    "detect_hard",
    "fuse",
    "to_dataframe",
    "classify_peak_flux",
    "class_rank",
    "meets_min_class",
]
