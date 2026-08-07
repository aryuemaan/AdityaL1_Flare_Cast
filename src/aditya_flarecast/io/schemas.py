"""Canonical data contracts used across the pipeline.

We keep the *interface* small and explicit so that readers for real ISSDC
Level-1 products and the synthetic generator are interchangeable. Everything
downstream consumes a :class:`LightCurve` (a thin, validated wrapper around a
tidy :class:`pandas.DataFrame` indexed by UTC time).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LightCurve:
    """A time-indexed multi-column light curve for one instrument.

    Attributes
    ----------
    instrument:
        ``"solexs"`` or ``"hel1os"``.
    data:
        DataFrame indexed by a tz-aware ``DatetimeIndex`` (UTC). Columns are
        physical quantities (fluxes / count rates). A boolean ``quality``
        column may be present after preprocessing (True == good sample).
    meta:
        Free-form provenance dictionary (source file, level, cadence, ...).
    """

    instrument: str
    data: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("LightCurve.data must be indexed by a DatetimeIndex")
        if self.data.index.tz is None:
            # Normalise to UTC to avoid ambiguity between instruments.
            self.data.index = self.data.index.tz_localize("UTC")
        self.data = self.data.sort_index()

    # -- Convenience ------------------------------------------------------- #
    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def start(self) -> pd.Timestamp:
        return self.data.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.data.index[-1]

    @property
    def cadence_s(self) -> float:
        if len(self.data) < 2:
            return float("nan")
        # Resolution-safe: convert to naive-UTC ns then diff.
        ns = self.data.index.tz_convert("UTC").tz_localize(None).values.astype(
            "datetime64[ns]"
        ).astype("int64")
        return float(np.median(np.diff(ns)) / 1e9)

    def series(self, column: str) -> pd.Series:
        return self.data[column]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LightCurve(instrument={self.instrument!r}, n={len(self)}, "
            f"cadence={self.cadence_s:.1f}s, cols={self.columns}, "
            f"span={self.start}..{self.end})"
        )


# GOES class boundaries used for typing convenience elsewhere.
GOES_ORDER = ["A", "B", "C", "M", "X"]


def goes_class_geq(cls_a: str, cls_b: str) -> bool:
    """Return True if flare class ``cls_a`` is >= ``cls_b`` (ignoring sub-scale)."""
    return GOES_ORDER.index(cls_a[0].upper()) >= GOES_ORDER.index(cls_b[0].upper())
