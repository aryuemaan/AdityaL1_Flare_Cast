"""Streaming replay: emit nowcast/forecast events as if data arrived live.

The simulator walks the processed frame forward one analysis step at a time,
maintaining a rolling buffer, and at each step recomputes features and asks the
forecaster for the probability of a flare in the next horizon. Soft/hard onset
flags are derived from the same detectors. This drives the dashboard's live
mode and provides a faithful offline test of the operational loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from aditya_flarecast.config import Settings
from aditya_flarecast.features.engineering import build_features
from aditya_flarecast.forecast.predict import Forecaster


@dataclass
class StreamEvent:
    time: pd.Timestamp
    soft_flux: float
    hard_counts: float
    flare_probability: float
    alert: bool
    threshold: float
    horizon_min: float


def replay(
    df: pd.DataFrame,
    settings: Settings,
    forecaster: Forecaster,
    step: int = 1,
    warmup_min: float | None = None,
) -> Iterator[StreamEvent]:
    """Yield :class:`StreamEvent` for each analysis step (offline, as-fast-as-possible).

    ``step`` decimates the emitted stream (e.g. step=6 -> one event per minute
    at 10 s cadence) to keep dashboards responsive.
    """
    cadence = df.attrs.get("cadence_s", settings.instrument.analysis_cadence_s)
    soft_band = df.attrs.get("solexs_flux_band", "flux_1_8A")
    hard_low = df.attrs.get("hel1os_low_band", "counts_10_30keV")

    warmup_min = warmup_min or settings.forecast.lookback_min
    warmup = int(round(warmup_min * 60.0 / cadence))

    # Precompute features once (they are causal), then reveal them step by step.
    features = build_features(df, cadence_s=cadence)
    out = forecaster.predict(features)
    prob = pd.Series(out.probability, index=features.index)

    soft = df[f"solexs_{soft_band}"]
    hard = df[f"hel1os_{hard_low}"]

    idx = np.arange(warmup, len(df), step)
    for i in idx:
        t = df.index[i]
        p = float(prob.iloc[i]) if i < len(prob) else 0.0
        yield StreamEvent(
            time=t,
            soft_flux=float(soft.iloc[i]),
            hard_counts=float(hard.iloc[i]),
            flare_probability=p,
            alert=p >= forecaster.threshold,
            threshold=forecaster.threshold,
            horizon_min=forecaster.horizon_min,
        )
