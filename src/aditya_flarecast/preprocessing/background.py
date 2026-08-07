"""Quiescent-background estimation.

Flare detection is fundamentally an "excess above background" problem. We
estimate a slowly-varying background as a rolling low percentile of the signal
over a window several times longer than a typical flare, then smooth it. The
low percentile is robust to flares (which are rare, positive excursions) while
still tracking real drifts in the quiescent level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_percentile_background(
    series: pd.Series,
    window_min: float,
    percentile: float = 10.0,
    cadence_s: float | None = None,
    smooth_min: float | None = None,
) -> pd.Series:
    """Estimate background as a centred rolling percentile.

    Parameters
    ----------
    series:
        Time-indexed signal.
    window_min:
        Rolling window length in minutes.
    percentile:
        Percentile (0-100) taken within each window as the background.
    smooth_min:
        Optional final smoothing window (minutes). Defaults to
        ``window_min / 3``.
    """
    if cadence_s is None:
        from aditya_flarecast.timeutils import to_ns_array

        ns = to_ns_array(series.index)
        cadence_s = float(np.median(np.diff(ns)) / 1e9)
    win = max(3, int(round(window_min * 60.0 / cadence_s)))

    bg = series.rolling(window=win, center=True, min_periods=max(3, win // 4)).quantile(
        percentile / 100.0
    )
    bg = bg.bfill().ffill()

    smooth_min = smooth_min if smooth_min is not None else window_min / 3.0
    sw = max(1, int(round(smooth_min * 60.0 / cadence_s)))
    if sw > 1:
        bg = bg.rolling(window=sw, center=True, min_periods=1).mean()
    return bg


def subtract_background(
    series: pd.Series,
    window_min: float,
    percentile: float = 10.0,
    cadence_s: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(excess, background)`` where excess = signal - background."""
    bg = rolling_percentile_background(
        series, window_min, percentile, cadence_s=cadence_s
    )
    excess = (series - bg).clip(lower=0)
    return excess, bg
