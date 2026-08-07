"""Data-quality flagging.

Detects samples that should not be trusted by the detectors/forecaster:
telemetry spikes (single-sample outliers), saturation (flat-topped rails), and
non-finite values. These are combined into a single boolean ``quality`` mask
(True == good).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hampel_mask(series: pd.Series, window: int = 7, n_sigma: float = 6.0) -> pd.Series:
    """Flag single-sample spikes using a Hampel (rolling MAD) filter.

    Returns a boolean Series where True marks *good* (non-spike) samples.
    """
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    scaled_mad = 1.4826 * mad
    # Avoid div-by-zero on flat regions.
    good = (series - med).abs() <= (n_sigma * scaled_mad + 1e-30)
    return good.fillna(True)


def finite_mask(df: pd.DataFrame) -> pd.Series:
    return df.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def quality_flags(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    window: int = 7,
    n_sigma: float = 6.0,
) -> pd.Series:
    """Combine finite + spike checks over the given columns into one mask."""
    columns = columns or [c for c in df.columns if c != "quality"]
    good = finite_mask(df[columns])
    for col in columns:
        good &= hampel_mask(df[col], window=window, n_sigma=n_sigma)
    if "quality" in df.columns:
        good &= df["quality"].astype(bool)
    good.name = "quality"
    return good
