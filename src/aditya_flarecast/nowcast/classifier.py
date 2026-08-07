"""GOES-style flare classification from 1-8 A soft-X-ray peak flux.

The GOES scheme maps peak flux (W m^-2) to a letter class with a linear
sub-scale::

    A: 1e-8 .. 1e-7
    B: 1e-7 .. 1e-6
    C: 1e-6 .. 1e-5
    M: 1e-5 .. 1e-4
    X: >= 1e-4          (e.g. X2.7 == 2.7e-4)

SoLEXS observes an energy band close to the GOES 1-8 A band, so this mapping
gives directly comparable, operationally familiar labels.
"""
from __future__ import annotations

_BOUNDS = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
_ORDER = ["A", "B", "C", "M", "X"]


def classify_peak_flux(peak_flux: float) -> str:
    """Return the GOES class string (e.g. ``"M1.4"``) for a peak flux."""
    if peak_flux <= 0:
        return "sub-A"
    for cls in reversed(_ORDER):
        if peak_flux >= _BOUNDS[cls]:
            sub = peak_flux / _BOUNDS[cls]
            return f"{cls}{sub:.1f}"
    # Below A-class.
    sub = peak_flux / _BOUNDS["A"]
    return f"A{sub:.1f}" if sub >= 0.1 else "sub-A"


def class_letter(goes_class: str) -> str:
    return goes_class[0].upper()


def class_rank(goes_class: str) -> int:
    """Ordinal rank (A=0 .. X=4); sub-A -> -1."""
    letter = goes_class[0].upper()
    return _ORDER.index(letter) if letter in _ORDER else -1


def meets_min_class(goes_class: str, min_class: str) -> bool:
    return class_rank(goes_class) >= class_rank(min_class)
